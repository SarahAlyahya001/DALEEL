"""Verify saved evaluation counts without weights, network access or inference."""
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.detection_metrics import match_image
from daleel.pipeline.detector import iou

INPUT = ROOT / "eval/results/per_image_metrics.jsonl"
BASELINE = ROOT / "eval/results/detection_metrics.json"
EXPECTED_INPUT_SHA256 = "91f752f1e14ca48bf63b5c6c52a5b645907e5a12f110fdf7342a967577ab62b1"
CLASSES = ("person", "car")
MIN_VALID = 0.05
SUPPORT_IOU = 0.50

class ValidationError(RuntimeError):
    """The inputs or re-scored counts do not reproduce the published evaluation baseline."""

def load():
    # The hash is checked on the uncompressed bytes, so both copies validate identically.
    data = INPUT.read_bytes() if INPUT.exists() else gzip.decompress(INPUT.with_suffix(".jsonl.gz").read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_INPUT_SHA256:
        raise ValidationError(f"{INPUT.name} SHA-256 is {digest}, expected {EXPECTED_INPUT_SHA256}.")
    images = []
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            images.append(dict(
                id=d["image_id"], seq=d["image_id"].split("_")[0], ann=d["annotation"],
                rn=[p for p in d["ranking"]["n"]["predictions"] if p["score"] > MIN_VALID],
                rs=[p for p in d["ranking"]["s"]["predictions"] if p["score"] > MIN_VALID],
                on=d["operational"]["n"]["predictions"], os=d["operational"]["s"]["predictions"],
                cap_n=min((p["score"] for p in d["ranking"]["n"]["predictions"]), default=0)
                if d["ranking"]["n"]["max_det_cap_hit"] else 0,
                cap_s=min((p["score"] for p in d["ranking"]["s"]["predictions"]), default=0)
                if d["ranking"]["s"]["max_det_cap_hit"] else 0,
                t_n=d["operational"]["n"]["predict_seconds"], t_s=d["operational"]["s"]["predict_seconds"]))
    return images

def gt_counts(images, subset=None):
    out = {c: 0 for c in CLASSES}
    for k, im in enumerate(images):
        if subset is not None and k not in subset:
            continue
        for g in im["ann"]["targets"]:
            if not g["ignore"] and g["class"] in out:
                out[g["class"]] += 1
    return out

def above(preds, t):
    return [p for p in preds if p["score"] > t]

def label_only(n_preds, s_preds, tn, tv):
    """DALEEL design: YOLOv8n candidates only; YOLOv8s supplies support labels."""
    N = above(n_preds, tn)
    S = above(s_preds, tv) if N else []
    pairs = sorted((-ov, i, j) for i, a in enumerate(N) for j, b in enumerate(S)
                   if a["class"] == b["class"] and (ov := iou(a["box"], b["box"])) >= SUPPORT_IOU)
    mi, mj = set(), set()
    for _, i, j in pairs:
        if i not in mi and j not in mj:
            mi.add(i), mj.add(j)
    return [dict(p, tier="Dual-Model" if i in mi else "Screening-Only") for i, p in enumerate(N)], bool(N)

def score_images(images, policy, subset=None, keep_preds=True):
    """Per-image TP/FP/candidate counts by class, and per-prediction statuses."""
    rows = []
    for k, im in enumerate(images):
        if subset is not None and k not in subset:
            continue
        preds, ran_s = policy(im)
        matched = match_image(preds, im["ann"])
        row = {"k": k, "ran_s": ran_s, "preds": matched if keep_preds else None}
        for c in CLASSES:
            row[c] = dict(cand=sum(p["class"] == c for p in preds),
                          tp=sum(p["class"] == c and p["status"] == "TP" for p in matched),
                          fp=sum(p["class"] == c and p["status"] == "FP" for p in matched))
        rows.append(row)
    return rows

def summarize(rows, gt):
    n_img = len(rows)
    out = {}
    for c in CLASSES:
        tp = sum(r[c]["tp"] for r in rows)
        fp = sum(r[c]["fp"] for r in rows)
        cand = sum(r[c]["cand"] for r in rows)
        fn = gt[c] - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        rcl = tp / gt[c] if gt[c] else 0.0
        out[c] = dict(gt=gt[c], tp=tp, fp=fp, fn=fn, precision=p, recall=rcl,
                      f1=2 * tp / (gt[c] + tp + fp) if gt[c] + tp + fp else 0.0,
                      candidates=cand, candidates_per_image=cand / n_img)
    tp = sum(out[c]["tp"] for c in CLASSES)
    fp = sum(out[c]["fp"] for c in CLASSES)
    g = sum(gt.values())
    cand = sum(out[c]["candidates"] for c in CLASSES)
    out["combined"] = dict(gt=g, tp=tp, fp=fp, fn=g - tp, precision=tp / (tp + fp) if tp + fp else 0.0,
                           recall=tp / g, f1=2 * tp / (g + tp + fp), candidates=cand,
                           candidates_per_image=cand / n_img)
    out["images"] = n_img
    out["images_running_yolov8s"] = sum(r["ran_s"] for r in rows)
    return out

def P_n(t):
    return lambda im: (above(im["rn"], t), False)

def P_s(t):
    return lambda im: (above(im["rs"], t), True)

def P_label(tn, tv):
    return lambda im: label_only(im["rn"], im["rs"], tn, tv)

def validate_baseline(images, gt, baseline):
    """Re-score the operational evaluation points and require exact equality with the
    published counts in eval/results/detection_metrics.json.

    A = YOLOv8n>0.30, B = YOLOv8s>0.50 (each checked from the threshold-filtered
    ranking pass AND from the re-scored operational predictions); C = DALEEL's
    Dual-Model/Screening-Only partition. Raises ValidationError listing
    every mismatch; returns {check: value} when all checks pass.
    """
    official = baseline["operational"]
    checks = {}
    fields = (("tp", "tp"), ("fp", "fp"), ("fn", "fn"), ("gt", "ground_truth"), ("candidates", "raw_predictions"))
    for run, pol, op_key in (("A", P_n(0.30), "on"), ("B", P_s(0.50), "os")):
        sources = {"ranking-filtered": pol, "operational-rescored": lambda im, k=op_key: (im[k], False)}
        for source, policy in sources.items():
            summary = summarize(score_images(images, policy, keep_preds=False), gt)
            for c in CLASSES:
                for field, ref_key in fields:
                    checks[f"{run}|{c}|{field}|{source}"] = (summary[c][field], official[run]["per_class"][c][ref_key])
    rows = score_images(images, P_label(0.30, 0.50))
    counts = defaultdict(lambda: {"TP": 0, "FP": 0, "raw": 0})
    for r in rows:
        for p in r["preds"]:
            counts[(p["tier"], p["class"])]["raw"] += 1
            if p["status"] in ("TP", "FP"):
                counts[(p["tier"], p["class"])][p["status"]] += 1
    for c in CLASSES:
        for label in ("Dual-Model", "Screening-Only"):
            ref = official["fixed_support_partition"][c][label]
            got = counts[(label, c)]
            checks[f"C|{c}|{label}|tp"] = (got["TP"], ref["tp"])
            checks[f"C|{c}|{label}|fp"] = (got["FP"], ref["fp"])
            checks[f"C|{c}|{label}|candidates"] = (got["raw"], ref["raw_predictions"])
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatches:
        raise ValidationError("Re-scoring does not reproduce the Stage 6A baseline: " + "; ".join(
            f"{k} got {a}, expected {b}" for k, (a, b) in sorted(mismatches.items())))
    return {k: v[0] for k, v in checks.items()}

def main():
    try:
        images = load()
        if len(images) != 548 or len({im["id"] for im in images}) != 548:
            raise ValidationError("Expected exactly 548 unique image records.")
        checks = validate_baseline(images, gt_counts(images), json.loads(BASELINE.read_text(encoding="utf-8")))
    except (ValidationError, OSError, ValueError, KeyError) as exc:
        print("Saved evaluation verification failed: " + str(exc), file=sys.stderr)
        return 1
    print(f"Verified {len(checks)}/{len(checks)} baseline counts across {len(images)} images. No inference ran; no files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
