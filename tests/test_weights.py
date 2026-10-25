from pathlib import Path

import daleel as daleel_package
from daleel import create_app
from scripts.setup_weights import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_sha256_is_stable(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"daleel")

    assert sha256_file(payload) == (
        "f3ca3b860d673ec979f797d0db2e243d04ad515c0a98fec1a6d8df14827d7885"
    )


def test_weights_use_local_defaults_with_optional_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("DALEEL_WEIGHTS_N", raising=False)
    monkeypatch.delenv("DALEEL_WEIGHTS_S", raising=False)
    app = create_app(str(tmp_path / "first"), {"TESTING": True})
    assert app.config["WEIGHTS_N"] == str(PROJECT_ROOT / "weights" / "yolov8n_sar_v1.pt")
    assert app.config["WEIGHTS_S"] == str(PROJECT_ROOT / "weights" / "yolov8s_sar_v1.pt")
    monkeypatch.setenv("DALEEL_WEIGHTS_N", str(tmp_path / "yolov8n.pt"))
    monkeypatch.setenv("DALEEL_WEIGHTS_S", str(tmp_path / "yolov8s.pt"))
    app = create_app(str(tmp_path / "second"), {"TESTING": True})
    assert app.config["WEIGHTS_N"] == str(tmp_path / "yolov8n.pt")
    assert "MIN_TRACK_FRAMES" not in app.config
    assert "CAMERA_HFOV_DEG" not in app.config

