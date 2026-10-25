import pytest
from eval import benchmark


def test_no_implicit_execution_or_legacy_flags(tmp_path):
    evidence = tmp_path / "historical.json"
    evidence.write_text("preserved")
    for arguments in ([], ["--json-out", str(evidence)]):
        with pytest.raises(SystemExit):
            benchmark.main(arguments)
    assert evidence.read_text() == "preserved"


def test_aggregate_dispatch_does_not_run_inference(monkeypatch, tmp_path):
    from eval import detection_run
    calls = []
    monkeypatch.setattr(detection_run, "aggregate", lambda *args: calls.append(args))
    monkeypatch.setattr(detection_run, "run", lambda: pytest.fail("Inference must not run"))
    benchmark.main(["aggregate", "--run-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert calls == [(tmp_path, tmp_path / "out")]


def test_run_dispatch_passes_dataset_and_output_folder(monkeypatch, tmp_path):
    from eval import detection_run
    calls = []
    monkeypatch.setattr(detection_run, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(detection_run, "aggregate", lambda *args: pytest.fail("No separate aggregation"))
    benchmark.main(["run", "--dataset", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out")])
    benchmark.main(["run"])
    assert calls == [(tmp_path / "data", tmp_path / "out"), (None, None)]
