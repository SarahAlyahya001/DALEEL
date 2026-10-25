import json
import pytest
import numpy as np
from types import SimpleNamespace
from eval import detection_run as run


def test_strict_confidence_boundaries(monkeypatch):
    monkeypatch.setattr(run, "extract", lambda _:[("person", (0, 0, 1, 1), .30), ("person", (1, 1, 2, 2), .3000001), ("car", (2, 2, 3, 3), .50), ("car", (3, 3, 4, 4), .5000001)])
    assert [p["score"] for p in run.predictions(None, .30)] == [.3000001, .50, .5000001]
    assert [p["score"] for p in run.predictions(None, .50)] == [.5000001]


def test_support_uses_n_box_one_to_one_same_class():
    n = [{"id": i, "class": "person", "box": [0, 0, 10, 10], "score": .4} for i in range(2)]
    s = [{"id": 0, "class": "person", "box": [0, 0, 5, 10], "score": .8},
         {"id": 1, "class": "car", "box": [0, 0, 10, 10], "score": .9}]
    out = run.assign_support(n, s)
    assert [p["support_label"] for p in out] == ["Dual-Model", "Screening-Only"]
    assert out[0]["box"] == n[0]["box"]
    assert len(out) == 2
    s[0]["box"][2] = 4.999
    assert all(p["support_label"] == "Screening-Only" for p in run.assign_support(n, s))


def test_checkpoint_roundtrip_and_corruption(tmp_path):
    config = {"test": True}
    record = {"configuration_digest": run.fingerprint(config), "image_id": "one"}
    record["payload_sha256"] = run.fingerprint(record)
    run.atomic_json(tmp_path / "image-one.json", record)
    assert run.read_checkpoints(tmp_path, config)[0]["image_id"] == "one"
    with pytest.raises(ValueError, match="configuration"):
        run.read_checkpoints(tmp_path, {})
    record["image_id"] = "tampered"
    run.atomic_json(tmp_path / "image-one.json", record)
    with pytest.raises(ValueError, match="Corrupted"):
        run.read_checkpoints(tmp_path, config)


@pytest.mark.parametrize("event", ["socket.connect", "socket.connect_ex", "socket.getaddrinfo"])
def test_offline_guard(event):
    with pytest.raises(RuntimeError, match="Network"):
        run.offline_guard(event, ())


def test_incomplete_results_not_published(tmp_path, monkeypatch):
    run.atomic_json(tmp_path / "configuration.json", {"test": True})
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"manifest": {"files": [{"path": "images/one.jpg"}]}}))
    monkeypatch.setattr(run, "PROVENANCE", provenance)
    with pytest.raises(ValueError, match="incomplete"):
        run.aggregate(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_complete_predict_arguments_and_timing():
    class Result:
        boxes = []
        speed = {"preprocess": 1, "inference": 2, "postprocess": 1}
    class Model:
        def predict(self, frame, **kwargs):
            assert kwargs["device"] == "cpu"
            assert kwargs["classes"] == [0, 2]
            assert kwargs["imgsz"] == 1280
            assert kwargs["iou"] == .7 and kwargs["max_det"] == 300
            return [Result()]
    out = run.predict(Model(), None, .30, run.NMSWarnings())
    assert out["predict_seconds"] > 0
    assert not out["predictions"]


def test_manifest_category_keys_normalize_json_roundtrip():
    # JSON object keys are strings; numeric sorting differs for categories 2 and 10.
    manifest = {"category_counts": {2: 3, 10: 4}}
    persisted = json.loads(json.dumps(manifest))
    assert json.loads(json.dumps(manifest)) == persisted


def result_box(coords, score=.01):
    box = SimpleNamespace(cls=np.array(2), conf=np.array(score), xyxy=np.array([coords]))
    return SimpleNamespace(boxes=[box])


@pytest.mark.parametrize("coords", [(10, 100, 20, 100), (20, 10, 10, 20), (10, 105, 20, 110)])
def test_ranking_degenerate_preserved_with_original_values(coords):
    output = run.predictions(result_box(coords), .001, ranking=True, shape=(100, 100, 3))
    assert output[0]["box"] == list(coords)
    assert output[0]["geometry_status"] == "degenerate_after_clipping"
    assert output[0]["class"] == "car" and output[0]["score"] == .01 and output[0]["id"] == 0


def test_nonfinite_ranking_is_fatal():
    with pytest.raises(ValueError, match="Nonfinite"):
        run.predictions(result_box((0, 0, float("nan"), 10)), .001, ranking=True, shape=(100, 100, 3))


def test_operational_degenerate_is_fatal():
    with pytest.raises(ValueError):
        run.predictions(result_box((10, 100, 20, 100), score=.7), .50, shape=(100, 100, 3))


def timing(wall=1., cpu=.9):
    return {"timing_unit": "seconds", "speed_unit": "milliseconds", "predict_seconds": wall,
            "process_cpu_seconds": cpu, "speed_ms": {"inference": wall * 900}}


def test_normal_timing_and_threshold_equality():
    assert run.validate_timing(timing(), [1, 2, 3]) == 60
    assert run.validate_timing(timing(60, 60), [1]) == 60
    assert run.validate_timing(timing(80, 80), [4]) == 80


def test_extreme_wall_aborts_instead_of_trimming():
    with pytest.raises(run.TimingAnomaly, match="Wall timing guard"):
        run.validate_timing(timing(60.000001, 60), [1])
    with pytest.raises(run.TimingAnomaly):
        run.validate_timing(timing(8694, 1), [1])


def test_process_interruption_below_absolute_limit():
    with pytest.raises(run.TimingAnomaly, match="suspension or interruption"):
        run.validate_timing(timing(10, .1), [1])


def test_checkpoint_timing_unit_mismatch():
    call = timing()
    call["timing_unit"] = "milliseconds"
    with pytest.raises(run.TimingAnomaly, match="unit"):
        run.validate_timing(call)
    call = timing()
    call["speed_ms"] = {"inference": 10000}
    with pytest.raises(run.TimingAnomaly, match="units/components"):
        run.validate_timing(call)


def test_failed_pilot_prevents_full_run():
    for session in ({}, {"pilot": {"passed": False}}):
        with pytest.raises(run.TimingAnomaly, match="prevents full run"):
            run.require_passed_pilot(session)
    run.require_passed_pilot({"pilot": {"passed": True}})


def test_pilot_summary_checks_all_calls_and_wall_cpu():
    records = [{mode: {model: timing() for model in ("n", "s")} for mode in ("ranking", "operational")} for _ in range(10)]
    summary = run.pilot_summary(records, 42, 38)
    assert summary["passed"]
    assert summary["estimated_full_seconds"] == pytest.approx(42 / 10 * 548)
    assert summary["wall_latency"]["count"] == 40
    with pytest.raises(run.TimingAnomaly, match="interruption"):
        run.pilot_summary(records, 800, 1)
    records[0]["ranking"]["n"]["timing_unit"] = "ms"
    with pytest.raises(run.TimingAnomaly, match="unit"):
        run.pilot_summary(records, 42, 38)


def test_disk_budget_checked_before_work_and_during_run():
    run.require_disk_headroom(1_400_000_000, preflight=True)
    run.require_disk_headroom(1_000_000_000)
    with pytest.raises(RuntimeError, match="disk"):
        run.require_disk_headroom(1_399_999_999, preflight=True)
    with pytest.raises(RuntimeError, match="disk"):
        run.require_disk_headroom(999_999_999)


def test_user_dataset_is_accepted_only_when_it_matches_the_accepted_manifest(tmp_path, monkeypatch):
    accepted = {"files": [{"path": "images/one.jpg", "bytes": 3, "sha256": "a" * 64, "rows": None}],
                "category_counts": {"1": 2}, "manifest_sha256": "b" * 64, "pairs": 1}
    reference = tmp_path / "eval/results/dataset_manifest.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({"source": "ultralytics-assets-mirror", "manifest": accepted}))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "ACCEPTED_MANIFEST", reference)
    # raw_manifest may return integer category keys; they compare after the JSON round trip.
    monkeypatch.setattr(run, "raw_manifest", lambda root: dict(accepted, category_counts={1: 2}))
    data, provenance = run.dataset_source(tmp_path / "VisDrone2019-DET-val")
    assert data == (tmp_path / "VisDrone2019-DET-val").resolve()
    assert provenance == {"source": "user-provided-copy", "reference": "eval/results/dataset_manifest.json",
                          "comparison": {"identical": True, "differences": [], "scope": provenance["comparison"]["scope"]},
                          "manifest": accepted}
    changed = dict(accepted, files=[dict(accepted["files"][0], sha256="c" * 64)])
    monkeypatch.setattr(run, "raw_manifest", lambda root: changed)
    with pytest.raises(ValueError, match="accepted"):
        run.dataset_source(tmp_path / "VisDrone2019-DET-val")


def test_default_dataset_source_is_the_validated_mirror(tmp_path, monkeypatch):
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"comparison": {"identical": True}, "manifest": {"pairs": 548}}))
    monkeypatch.setattr(run, "PROVENANCE", provenance)
    monkeypatch.setattr(run, "raw_manifest", lambda root: pytest.fail("The mirror is checked in configuration()"))
    assert run.dataset_source() == (run.DATA, {"comparison": {"identical": True}, "manifest": {"pairs": 548}})


def test_aggregate_uses_the_dataset_record_stored_with_the_run(tmp_path, monkeypatch):
    run.atomic_json(tmp_path / "configuration.json", {"test": True})
    run.atomic_json(tmp_path / run.DATASET_RECORD, {"manifest": {"files": [{"path": "images/one.jpg"}]}})
    monkeypatch.setattr(run, "PROVENANCE", tmp_path / "absent-mirror-record.json")
    with pytest.raises(ValueError, match="incomplete"):
        run.aggregate(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_run_refuses_existing_evidence_before_any_dataset_or_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "require_disk_headroom", lambda *a, **k: pytest.fail("No preflight work"))
    monkeypatch.setattr(run, "dataset_source", lambda *a: pytest.fail("No dataset or model work"))
    for name in run.EVIDENCE_FILES:
        output = tmp_path / name.replace(".", "_")
        output.mkdir()
        (output / name).write_text("existing evidence")
        with pytest.raises(ValueError, match="overwrite"):
            run.run(tmp_path / "dataset", output)
        assert (output / name).read_text() == "existing evidence"
        assert list(output.iterdir()) == [output / name]
    # The default output folder holds the accepted results, so a plain run stops at once.
    with pytest.raises(ValueError, match="overwrite"):
        run.run()
