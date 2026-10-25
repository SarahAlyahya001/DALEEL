"""CPU-only offline detection evaluation runner. No Flask factory, worker, uploads, or database.

This runner reproduces the evaluation of the original pretrained configuration (eval/results/), with
the original extraction order and Dual-Model/Screening-Only labels (eval/legacy_pipeline.py).
"""
import csv
import ctypes
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from eval.detection_metrics import summarize, support_partition
from eval.legacy_pipeline import consensus, extract
from eval.visdrone import parse_annotations
from scripts.acquire_visdrone_val import raw_manifest, sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/stage6a/mirror/VisDrone2019-DET-val"
PROVENANCE = DATA.parent / "provenance.json"
ACCEPTED_MANIFEST = ROOT / "eval/results/dataset_manifest.json"
DATASET_RECORD = "dataset_provenance.json"
EVIDENCE_FILES = ("detection_metrics.json", "detection_metrics.csv", "per_image_metrics.jsonl",
                  "environment.json", "dataset_manifest.json")
PARAMETERS = {
    "protocol": "daleel-visdrone-two-class-v2", "images": 548, "device": "cpu", "imgsz": 1280,
    "ranking_conf": .001, "screen_conf": .30, "verify_conf": .50, "strict_confidence": True,
    "nms_iou": .7, "max_det": 300, "agnostic_nms": False, "augment": False, "half": False,
    "classes": [0, 2], "effective_batch": 1, "support_iou": .50,
    "matching_iou": .50, "ap_iou": [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95],
    "recall_points": 101, "gt_mapping": {"1": "person", "2": "person", "4": "car"},
    "region_union_coverage": .50, "score_zero_ignore": "same-class unmatched prediction IoA >= evaluation IoU",
    "nms_max_nms": 30000, "nms_max_time_img_seconds": .05, "nms_base_time_limit_seconds": 2.,
    "single_image_letterbox": "auto rectangular stride padding for PyTorch",
    "warmup_calls_per_model_per_mode": 5,
    "ranking_degenerate_policy": "finite nonpositive clipped area preserved and counted FP, bypass ignore/matching",
    "timing_gate": "wall > max(60 seconds,20*median(completed valid warmup/pilot calls)) aborts attempt",
    "interruption_gate": "wall > 5 seconds and wall > 20*process_cpu_seconds aborts attempt",
    "timing_units": "seconds; Ultralytics speed fields explicitly milliseconds",
    "disk_reserve_bytes": 1_000_000_000,
    "preflight_output_budget_bytes": 400_000_000,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def dataset_source(path=None):
    """Return the image folder and its acquisition record.

    By default the validated mirror under datasets/stage6a is used. A user-provided
    VisDrone2019-DET-val folder is accepted only when every raw image and annotation
    matches the recorded evaluation manifest (size and SHA-256), so the evaluated data,
    and therefore the protocol, stay the same.
    """
    if path is None:
        return DATA, json.loads(PROVENANCE.read_text(encoding="utf-8"))
    root = Path(path).resolve()
    accepted = json.loads(ACCEPTED_MANIFEST.read_text(encoding="utf-8"))["manifest"]
    manifest = json.loads(json.dumps(raw_manifest(root)))
    if manifest != accepted:
        raise ValueError("Dataset does not match the accepted VisDrone2019-DET-val manifest")
    # No local path is recorded; the manifest identifies the data.
    return root, {"source": "user-provided-copy", "reference": ACCEPTED_MANIFEST.relative_to(ROOT).as_posix(),
                  "comparison": {"identical": True, "differences": [],
                                 "scope": "all raw images and annotations against the accepted Stage 6A manifest"},
                  "manifest": manifest}


def configuration(data=None, provenance=None):
    if os.environ.get("DALEEL_IMGSZ", "1280") != "1280":
        raise ValueError("Application image-size environment differs from approved protocol")
    if data is None:
        data, provenance = dataset_source()
    manifest = json.loads(json.dumps(raw_manifest(data)))
    if not provenance["comparison"]["identical"] or manifest != provenance["manifest"]:
        raise ValueError("Dataset no longer matches validated acquisition")
    models = {name: {"path": f"weights/yolov8{name}.pt", "sha256": sha256(ROOT / f"weights/yolov8{name}.pt")}
              for name in ("n", "s")}
    sources = {p: sha256(ROOT / p) for p in ("eval/detection_run.py", "eval/detection_metrics.py", "eval/visdrone.py", "eval/legacy_pipeline.py")}
    packages = {name: version(name) for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "psutil")}
    return {"parameters": PARAMETERS, "models": models, "dataset_manifest_sha256": manifest["manifest_sha256"],
            "source_hashes": sources, "packages": packages}, provenance


def offline_guard(event, args):
    if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo"):
        raise RuntimeError("Network access prohibited during Stage 6A inference")


def no_subprocess(event, args):
    if event in ("subprocess.Popen", "os.system"):
        raise RuntimeError("Subprocess/download launch prohibited during inference")


class NMSWarnings(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.messages.append(record.getMessage())


def predictions(result, confidence, *, ranking=False, shape=None):
    if not ranking:
        extracted = extract(result)  # Strict canonical operational check.
        rows = [(cls, list(box), score) for cls, box, score in extracted if score > confidence]
    else:
        rows = []
        for box in result.boxes:
            category = int(box.cls.item())
            if category not in (0, 2):
                continue
            coords = [float(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf.item())
            if len(coords) != 4 or not np.isfinite(coords).all() or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Nonfinite/invalid ranking prediction")
            if score > confidence:
                rows.append(("person" if category == 0 else "car", coords, score))
        rows.sort(key=lambda row: (row[0], tuple(row[1]), -row[2]))
    data = []
    for identifier, (cls, box, score) in enumerate(rows):
        degenerate = box[2] <= box[0] or box[3] <= box[1]
        if shape is not None:
            height, width = shape[:2]
            degenerate |= min(width, box[2]) <= max(0, box[0]) or min(height, box[3]) <= max(0, box[1])
        if degenerate and not ranking:
            raise ValueError("Production-integrity defect: degenerate operational prediction")
        data.append({"id": identifier, "class": cls, "box": box, "score": score,
                     "geometry_status": "degenerate_after_clipping" if degenerate else "valid"})
    return data


class TimingAnomaly(RuntimeError):
    pass


def validate_timing(call, reference=(), *, warmup=False):
    if call.get("timing_unit") != "seconds" or call.get("speed_unit") != "milliseconds":
        raise TimingAnomaly("Checkpoint timing-unit mismatch")
    wall, cpu = call["predict_seconds"], call["process_cpu_seconds"]
    if not np.isfinite([wall, cpu]).all() or wall <= 0 or cpu < 0:
        raise TimingAnomaly("Invalid wall/process timing")
    components = list(call.get("speed_ms", {}).values())
    if not np.isfinite(components).all() or any(v < 0 for v in components):
        raise TimingAnomaly("Invalid diagnostic milliseconds")
    if sum(components) / 1000 > wall + max(.05, wall * .10):
        raise TimingAnomaly("Inconsistent checkpoint timing units/components")
    threshold = max(60., 20 * float(np.median(reference))) if reference else 60.
    if not warmup and wall > threshold:
        raise TimingAnomaly(f"Wall timing guard: {wall:.6f}s exceeds {threshold:.6f}s")
    if wall > 5. and wall > 20 * cpu:
        raise TimingAnomaly("Wall/process relationship indicates suspension or interruption")
    return threshold


def pilot_summary(records, wall_seconds, cpu_seconds):
    if len(records) != 10:
        raise TimingAnomaly("Pilot requires exactly ten complete images")
    calls = [r[mode][model] for r in records for mode in ("ranking", "operational") for model in ("n", "s")]
    for call in calls:
        validate_timing(call, call.get("timing_reference_seconds", []))
    if wall_seconds <= 0 or cpu_seconds < 0 or wall_seconds > 5 and wall_seconds > 20 * cpu_seconds:
        raise TimingAnomaly("Pilot interruption prevents full run")
    inference_sum = sum(c["predict_seconds"] for c in calls)
    if wall_seconds + .05 < inference_sum or wall_seconds - inference_sum > max(60., 20 * float(np.median([c["predict_seconds"] for c in calls]))):
        raise TimingAnomaly("Pilot gaps/units prevent credible runtime projection")
    return {"images": 10, "seconds": wall_seconds, "process_cpu_seconds": cpu_seconds,
            "estimated_full_seconds": wall_seconds / 10 * 548, "technical_only": True,
            "thresholds_unchanged": True, "passed": True,
            "wall_latency": latency([c["predict_seconds"] for c in calls]),
            "cpu_latency": latency([c["process_cpu_seconds"] for c in calls]),
            "per_pass_model": {mode: {model: {
                "wall": latency([r[mode][model]["predict_seconds"] for r in records]),
                "cpu": latency([r[mode][model]["process_cpu_seconds"] for r in records])}
                for model in ("n", "s")} for mode in ("ranking", "operational")}}


def require_passed_pilot(session):
    if not session.get("pilot", {}).get("passed"):
        raise TimingAnomaly("Failed pilot prevents full run")


def require_disk_headroom(free_bytes, *, preflight=False):
    required = PARAMETERS["disk_reserve_bytes"] + (PARAMETERS["preflight_output_budget_bytes"] if preflight else 0)
    if free_bytes < required:
        raise RuntimeError(f"Insufficient free disk headroom: {free_bytes} bytes; require {required}")


def assign_support(n, s):
    candidates = consensus([(p["class"], tuple(p["box"]), p["score"]) for p in n],
                           [(p["class"], tuple(p["box"]), p["score"]) for p in s])
    return [dict(p, support_label=c.support_label, verification_score=c.verification_score) for p, c in zip(n, candidates)]


def predict(model, image, confidence, warnings, *, mode="operational"):
    warnings.messages.clear()
    started_utc = utc()
    cpu_start = time.process_time()
    start = time.perf_counter()
    results = model.predict(image, conf=confidence, classes=[0, 2], verbose=False, device="cpu", imgsz=1280,
                            iou=.7, max_det=300, agnostic_nms=False, augment=False, half=False)
    elapsed = time.perf_counter() - start
    cpu_elapsed = time.process_time() - cpu_start
    if len(results) != 1 or not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Invalid prediction/timing result")
    data = predictions(results[0], confidence, ranking=mode == "ranking", shape=image.shape if image is not None else None)
    return {"predictions": data, "predict_seconds": elapsed, "process_cpu_seconds": cpu_elapsed,
            "started_utc": started_utc, "ended_utc": utc(), "timing_unit": "seconds", "speed_unit": "milliseconds",
            "speed_ms": results[0].speed, "warnings": list(warnings.messages),
            "max_det_cap_hit": len(results[0].boxes) >= 300}


def environment():
    import psutil
    power = None
    if os.name == "nt":
        result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, errors="replace")
        power = result.stdout.strip()
    battery = psutil.sensors_battery()
    return {"python": sys.version, "os": platform.platform(), "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total, "power_plan": power,
            "ac_power": battery.power_plugged if battery else None,
            "background_thermal_limitations": "Background workloads and temperature are uncontrolled; timing is laptop/session specific.",
            "opencv_threads": cv2.getNumThreads(), "offline_enforcement": "Python socket audit guard and subprocess guard during model loading/inference"}


def latency(values):
    return {"count": len(values), "mean_seconds": float(np.mean(values)), "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95))} if values else {"count": 0}


def read_checkpoints(run_dir, config):
    records = []
    for path in sorted(run_dir.glob("image-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["configuration_digest"] != fingerprint(config):
            raise ValueError("Checkpoint configuration mismatch")
        digest = record.pop("payload_sha256")
        if fingerprint(record) != digest:
            raise ValueError("Corrupted checkpoint")
        if config.get("parameters", {}).get("protocol") == "daleel-visdrone-two-class-v2":
            for mode in ("ranking", "operational"):
                for model in ("n", "s"):
                    validate_timing(record[mode][model], record[mode][model].get("timing_reference_seconds", []))
        records.append(record)
    return records


def aggregate(run_dir, output_dir):
    config = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
    records = read_checkpoints(run_dir, config)
    # A run on a user-provided dataset stores its own record; otherwise the mirror's is used.
    record = run_dir / DATASET_RECORD
    provenance = json.loads((record if record.exists() else PROVENANCE).read_text(encoding="utf-8"))
    expected = sorted(Path(f["path"]).stem for f in provenance["manifest"]["files"] if f["path"].startswith("images/"))
    if sorted(r["image_id"] for r in records) != expected or len(records) != 548:
        raise ValueError("Cannot publish incomplete/duplicate dataset results")
    annotations = {r["image_id"]: r["annotation"] for r in records}
    summary = {"configuration": config, "images": len(records), "ranking": {}, "operational": {}}
    curves = {}
    for mode in ("ranking", "operational"):
        for model, label in (("n", "A"), ("s", "B")):
            predictions_by_image = {r["image_id"]: r[mode][model]["predictions"] for r in records}
            metric, curve, assignments = summarize(predictions_by_image, annotations, ranking=mode == "ranking")
            summary[mode][label] = metric
            if mode == "ranking":
                curves[label] = curve
            elif model == "n":
                summary[mode]["fixed_support_partition"] = support_partition(assignments)
        summary[mode]["E"] = summary[mode]["A"]
    conditional = sum(bool(r["operational"]["n"]["predictions"]) for r in records)
    summary["invocations"] = {"per_mode_n": 548, "per_mode_s_standalone": 548, "total_measured_n": 1096,
                              "total_measured_s": 1096, "deployed_conditional_s": conditional,
                              "conditional_rate": conditional / 548, "estimated_avoided": 548 - conditional,
                              "estimated_avoided_fraction": (548 - conditional) / 548,
                              "note": "Both models physically ran on every image for comparison; deployed savings are counterfactual."}
    summary["timing"] = {mode: {m: latency([r[mode][m]["predict_seconds"] for r in records]) for m in ("n", "s")}
                         for mode in ("ranking", "operational")}
    summary["process_cpu_timing"] = {mode: {m: latency([r[mode][m]["process_cpu_seconds"] for r in records]) for m in ("n", "s")}
                                    for mode in ("ranking", "operational")}
    summary["degenerate_predictions"] = {mode: {m: {cls: sum(p.get("geometry_status") == "degenerate_after_clipping" and p["class"] == cls
        for r in records for p in r[mode][m]["predictions"]) for cls in ("person", "car")} for m in ("n", "s")} for mode in ("ranking", "operational")}
    summary["degenerate_by_image"] = [{"image_id": r["image_id"], "pass": mode, "model": m,
        "prediction_id": p["id"], "class": p["class"], "score": p["score"], "box": p["box"]}
        for r in records for mode in ("ranking", "operational") for m in ("n", "s")
        for p in r[mode][m]["predictions"] if p.get("geometry_status") == "degenerate_after_clipping"]
    summary["timing"]["derived_deployed_predict_sum"] = latency([r["operational"]["n"]["predict_seconds"] +
        (r["operational"]["s"]["predict_seconds"] if r["operational"]["n"]["predictions"] else 0) for r in records])
    summary["cap_hits"] = {mode: {m: sum(r[mode][m]["max_det_cap_hit"] for r in records) for m in ("n", "s")}
                           for mode in ("ranking", "operational")}
    summary["ignore_audit"] = {"region_excluded_gt": sum(a["region_excluded_gt"] for a in annotations.values()),
                               "score_zero_target_gt": sum(g["ignore"] for a in annotations.values() for g in a["targets"])}
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(EVIDENCE_FILES)
    if any((output_dir / name).exists() for name in names):
        raise ValueError("Refusing to overwrite existing evaluation evidence")
    atomic_json(output_dir / names[0], summary)
    atomic_json(output_dir / "dataset_manifest.json", provenance)
    sessions = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("session-*.json"))]
    atomic_json(output_dir / "environment.json", {"configuration": config, "sessions": sessions})
    with (output_dir / "per_image_metrics.jsonl").open("x", encoding="utf-8") as stream:
        for r in records:
            # Raw per-image predictions, annotations, support labels and timings remain reproducible.
            stream.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
    with (output_dir / "detection_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "output", "class", "gt", "predictions", "tp", "fp", "fn", "precision", "recall", "f1", "ap50", "map50_95"])
        for mode in ("ranking", "operational"):
            for label in ("A", "B", "E"):
                for cls, metric in summary[mode][label]["per_class"].items():
                    writer.writerow([mode, label, cls] + [metric.get(k) for k in ("ground_truth", "prediction_count", "tp", "fp", "fn", "precision", "recall", "f1", "ap50", "map50_95")])
    curve_dir = output_dir / "precision_recall_curves"
    curve_dir.mkdir(exist_ok=True)
    for label, classes in curves.items():
        for cls, points in classes.items():
            with (curve_dir / f"{label}_{cls}.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["recall", "precision"])
                writer.writeheader()
                writer.writerows(points)
            coordinates = " ".join(f'{40 + p["recall"] * 500:.4f},{340 - p["precision"] * 300:.4f}' for p in points)
            (curve_dir / f"{label}_{cls}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 390"><rect width="600" height="390" fill="white"/><text x="40" y="24">{label} {cls}: AP@0.50 precision envelope</text><path d="M40 40V340H540" fill="none" stroke="black"/><polyline points="{coordinates}" fill="none" stroke="#865d30"/><text x="260" y="380">Recall (0–1)</text><text x="45" y="55">Precision (0–1)</text></svg>', encoding="utf-8")
    print(json.dumps({"aggregated_images": len(records), "metrics_sha256": sha256(output_dir / names[0])}), flush=True)
    return summary


def run(dataset=None, output_dir=None):
    output_dir = ROOT / "eval/results" if output_dir is None else Path(output_dir).resolve()
    # Refuse before any dataset, model or inference work rather than after a full run.
    if any((output_dir / name).exists() for name in EVIDENCE_FILES):
        raise ValueError("Refusing to overwrite existing evaluation evidence")
    require_disk_headroom(shutil.disk_usage(ROOT).free, preflight=True)
    data, provenance = dataset_source(dataset)
    config, _ = configuration(data, provenance)
    digest = fingerprint(config)
    attempt_id = utc().replace(":", "").replace("+", "_") + "-" + uuid.uuid4().hex[:8]
    run_dir = ROOT / "runs/stage6a" / (digest[:12] + "-" + attempt_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run configuration mismatch")
    if not config_path.exists():
        atomic_json(config_path, config)
    if dataset is not None:
        atomic_json(run_dir / DATASET_RECORD, provenance)
    session = {"started_utc": utc(), "resumed_images": 0, "attempt_id": attempt_id,
               "environment": environment(), "configuration_digest": digest}
    session_path = run_dir / ("session-" + str(time.time_ns()) + ".json")
    atomic_json(session_path, session)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    platform.uname()
    sys.addaudithook(offline_guard)
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import torch
    import psutil
    import cpuinfo
    # py-cpuinfo uses a local helper process on Windows. Cache that hardware-only
    # discovery before blocking subprocesses; the model computations are unchanged.
    cpu_description = cpuinfo.get_cpu_info()
    session["cpu_brand"] = cpu_description.get("brand_raw")
    cpuinfo.get_cpu_info = lambda: dict(cpu_description)
    sys.addaudithook(no_subprocess)
    models = {name: YOLO(str(ROOT / value["path"])) for name, value in config["models"].items()}
    for model in models.values():
        if len(model.names) != 80 or model.names[0] != "person" or model.names[2] != "car":
            raise ValueError("Weights are not the configured COCO models")
    warnings = NMSWarnings()
    LOGGER.addHandler(warnings)
    images = sorted((data / "images").glob("*.jpg"))
    session["warmup_calls"] = 0
    process = psutil.Process()
    start = time.perf_counter()
    reference = []
    calls_path = run_dir / "calls.jsonl"

    def observed_call(name, frame, confidence, mode, image_id, stage):
        call = predict(models[name], frame, confidence, warnings, mode=mode)
        call["timing_reference_seconds"] = list(reference)
        # Preserve original raw outputs and timing even when the gate rejects this call.
        with calls_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"model": name, "pass": mode, "image_id": image_id, "stage": stage, **call}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        session["last_call"] = {"model": name, "pass": mode, "image_id": image_id, "stage": stage,
                                "predict_seconds": call["predict_seconds"], "process_cpu_seconds": call["process_cpu_seconds"],
                                "warnings": call["warnings"]}
        if call["warnings"]:
            raise RuntimeError("Model warning; attempt aborted: " + " | ".join(call["warnings"]))
        validate_timing(call, reference, warmup=stage == "warmup")
        if stage in ("warmup", "pilot"):
            reference.append(call["predict_seconds"])
        return call

    try:
        for mode in ("ranking", "operational"):
            for image in images[:5]:
                frame = cv2.imread(str(image))
                for name, model in models.items():
                    confidence = .001 if mode == "ranking" else (.30 if name == "n" else .50)
                    observed_call(name, frame, confidence, mode, image.stem, "warmup")
                    session["warmup_calls"] += 1
        session["torch_threads"] = torch.get_num_threads()
        session["torch_interop_threads"] = torch.get_num_interop_threads()
        session["warmup_seconds"] = time.perf_counter() - start
        atomic_json(session_path, session)
        print(json.dumps({"run_dir": str(run_dir), "warmup": session["warmup_seconds"], "threads": session["torch_threads"]}), flush=True)
        pilot_start = time.perf_counter()
        pilot_cpu_start = time.process_time()
        for index, path in enumerate(images):
            if index >= 10:
                require_passed_pilot(session)
            require_disk_headroom(shutil.disk_usage(ROOT).free)
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError("Unreadable validated image")
            height, width = frame.shape[:2]
            record = {"image_id": path.stem, "configuration_digest": digest, "session": session_path.name,
                      "annotation": parse_annotations(data / "annotations" / (path.stem + ".txt"), width, height)}
            for mode in ("ranking", "operational"):
                record[mode] = {}
                for name, model in models.items():
                    confidence = .001 if mode == "ranking" else (.30 if name == "n" else .50)
                    record[mode][name] = observed_call(name, frame, confidence, mode, path.stem, "pilot" if index < 10 else "full")
                if mode == "operational":
                    record[mode]["n"]["predictions"] = assign_support(record[mode]["n"]["predictions"], record[mode]["s"]["predictions"])
            record["payload_sha256"] = fingerprint(record)
            atomic_json(run_dir / ("image-" + path.stem + ".json"), record)
            if index == 9:
                pilot_seconds = time.perf_counter() - pilot_start
                pilot_cpu = time.process_time() - pilot_cpu_start
                session["pilot"] = pilot_summary(read_checkpoints(run_dir, config), pilot_seconds, pilot_cpu)
                atomic_json(session_path, session)
                print(json.dumps({"pilot": session["pilot"]}), flush=True)
            if (index + 1) % 10 == 0 or index == 547:
                memory = process.memory_info()
                print(json.dumps({"completed": index + 1, "total": 548, "rss_bytes": memory.rss}), flush=True)
        session["status"] = "complete"
    except BaseException as error:
        session["status"] = "failed"
        session["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        session["ended_utc"] = utc()
        session["elapsed_seconds"] = time.perf_counter() - start
        memory = process.memory_info()
        session["peak_working_set_bytes"] = getattr(memory, "peak_wset", None)
        session["final_rss_bytes"] = memory.rss
        atomic_json(session_path, session)
        LOGGER.removeHandler(warnings)
    aggregate(run_dir, output_dir)
