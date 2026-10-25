import hashlib
import io
from pathlib import Path

import pytest

from scripts import setup_weights


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def spec(name, payload):
    return setup_weights.WeightSpec(
        filename=name,
        url=f"https://example.invalid/{name}",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_published_weight_metadata_is_exact():
    assert str(setup_weights.PROJECT_ROOT) in setup_weights.sys.path
    assert [(item.filename, item.size, item.sha256) for item in setup_weights.WEIGHT_SPECS] == [
        ("yolov8n_sar_v1.pt", 6_566_033,
         "21f16d2d72ec76bf020eeeddbc3b59836fa534af43fc88ec3723ce2b7af72e7d"),
        ("yolov8s_sar_v1.pt", 22_605_073,
         "90cda60aea2091c62a388954fe192104f9c5dbce96397bee00cbf4d650c40af4"),
    ]
    from daleel.pipeline.weights import RELEASE_REPOSITORY, RELEASE_TAG
    assert [item.url for item in setup_weights.WEIGHT_SPECS] == [
        f"https://github.com/{RELEASE_REPOSITORY}/releases/download/{RELEASE_TAG}/yolov8n_sar_v1.pt",
        f"https://github.com/{RELEASE_REPOSITORY}/releases/download/{RELEASE_TAG}/yolov8s_sar_v1.pt",
    ]
    assert all("ultralytics/assets" not in item.url for item in setup_weights.WEIGHT_SPECS)  # no pretrained fallback


def test_unconfigured_release_is_refused_without_network(tmp_path):
    item = setup_weights.WeightSpec("n.pt", "https://github.com/OWNER/REPOSITORY/releases/download/v2.0.0/n.pt",
                                    3, "0" * 64)
    with pytest.raises(setup_weights.WeightSetupError, match="not configured"):
        setup_weights.ensure_weights(tmp_path, (item,), lambda *_a, **_k: pytest.fail("network used"))


def test_valid_existing_weights_are_reused_without_network(tmp_path):
    specs = (spec("n.pt", b"n-model"), spec("s.pt", b"s-model"))
    for item, payload in zip(specs, (b"n-model", b"s-model")):
        (tmp_path / item.filename).write_bytes(payload)

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("network must not be used for valid existing weights")

    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    paths = setup_weights.ensure_weights(tmp_path, specs, network_forbidden)
    assert set(paths) == {"n.pt", "s.pt"}
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_verify_only_never_downloads(tmp_path, monkeypatch):
    item = spec("n.pt", b"expected")
    monkeypatch.setattr(setup_weights.urllib.request, "urlopen",
                        lambda *_a, **_k: pytest.fail("network used"))
    with pytest.raises(setup_weights.WeightSetupError, match="missing"):
        setup_weights.verify_weights(tmp_path, (item,))


def test_missing_weights_are_downloaded_and_verified(tmp_path):
    payloads = {"n.pt": b"n-model", "s.pt": b"s-model"}
    specs = tuple(spec(name, payload) for name, payload in payloads.items())
    requested = []

    def opener(request, timeout):
        requested.append((request.full_url, timeout))
        return Response(payloads[Path(request.full_url).name])

    setup_weights.ensure_weights(tmp_path, specs, opener)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == payloads
    assert len(requested) == 2


@pytest.mark.parametrize("payload", [b"short", b"wrong-model!!!"])
def test_mismatched_existing_file_is_never_replaced(tmp_path, payload):
    expected = b"expected-model"
    item = spec("n.pt", expected)
    destination = tmp_path / item.filename
    destination.write_bytes(payload)
    with pytest.raises(setup_weights.WeightSetupError, match="not replaced"):
        setup_weights.ensure_weights(tmp_path, (item,), lambda *_a, **_k: Response(expected))
    assert destination.read_bytes() == payload


def test_invalid_download_is_not_published_and_partial_is_removed(tmp_path):
    item = spec("n.pt", b"expected-model")
    with pytest.raises(setup_weights.WeightSetupError):
        setup_weights.ensure_weights(tmp_path, (item,), lambda *_a, **_k: Response(b"damaged"))
    assert not (tmp_path / item.filename).exists()
    assert list(tmp_path.glob("*.part")) == []


def test_interrupted_download_is_reported_and_partial_is_removed(tmp_path):
    item = spec("n.pt", b"expected-model")

    class Interrupted(Response):
        def read(self, _size=-1):
            raise OSError("synthetic interrupted transfer")

    with pytest.raises(setup_weights.WeightSetupError, match="interrupted transfer"):
        setup_weights.ensure_weights(tmp_path, (item,), lambda *_a, **_k: Interrupted(b""))
    assert not (tmp_path / item.filename).exists()
    assert list(tmp_path.glob("*.part")) == []


def test_one_valid_and_one_missing_weight(tmp_path):
    specs = (spec("n.pt", b"n-model"), spec("s.pt", b"s-model"))
    (tmp_path / "n.pt").write_bytes(b"n-model")
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(b"s-model")

    setup_weights.ensure_weights(tmp_path, specs, opener)
    assert calls == ["https://example.invalid/s.pt"]
    assert (tmp_path / "n.pt").read_bytes() == b"n-model"
    assert (tmp_path / "s.pt").read_bytes() == b"s-model"


def test_offline_load_check_invokes_runtime_detector_for_both_models(tmp_path):
    specs = (spec("n.pt", b"n-model"), spec("s.pt", b"s-model"))
    paths = {item.filename: tmp_path / item.filename for item in specs}
    calls = []

    def detector_factory(weights_n, weights_s, specs):
        calls.append((weights_n, weights_s))
        return object()

    setup_weights.load_models_offline(
        paths, specs, detector_factory=detector_factory, enforce_offline=False
    )
    assert calls == [(paths["n.pt"], paths["s.pt"])]


@pytest.mark.parametrize("failed_model", ["n.pt", "s.pt"])
def test_offline_load_check_fails_if_either_model_cannot_load(tmp_path, failed_model):
    specs = (spec("n.pt", b"n-model"), spec("s.pt", b"s-model"))
    paths = {item.filename: tmp_path / item.filename for item in specs}

    def detector_factory(_weights_n, _weights_s, specs):
        raise RuntimeError(f"synthetic failure loading {failed_model}")

    with pytest.raises(setup_weights.WeightSetupError, match=failed_model):
        setup_weights.load_models_offline(
            paths, specs, detector_factory=detector_factory, enforce_offline=False
        )
