"""The two fine-tuned detection models used by DALEEL, pinned by size and SHA-256.

The files are attached to a versioned GitHub Release of the repository and are downloaded and
verified by scripts/setup_weights.py. There is no fallback to other weights.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

# The repository that hosts the release, as "owner/name".
RELEASE_REPOSITORY = "SarahAlyahya001/DALEEL"
RELEASE_TAG = "v2.0.0"
UNCONFIGURED = "OWNER/REPOSITORY"


@dataclass(frozen=True)
class WeightSpec:
    filename: str
    url: str
    size: int
    sha256: str


def release_url(filename: str) -> str:
    return f"https://github.com/{RELEASE_REPOSITORY}/releases/download/{RELEASE_TAG}/{filename}"


YOLOV8N = WeightSpec("yolov8n_sar_v1.pt", release_url("yolov8n_sar_v1.pt"), 6_566_033,
                     "21f16d2d72ec76bf020eeeddbc3b59836fa534af43fc88ec3723ce2b7af72e7d")
YOLOV8S = WeightSpec("yolov8s_sar_v1.pt", release_url("yolov8s_sar_v1.pt"), 22_605_073,
                     "90cda60aea2091c62a388954fe192104f9c5dbce96397bee00cbf4d650c40af4")
WEIGHT_SPECS = (YOLOV8N, YOLOV8S)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_problem(path, spec: WeightSpec) -> str | None:
    """A description of why the file is not the pinned model, or None when it is."""
    path = Path(path)
    if not path.is_file():
        return f"Required model file is missing: {path}"
    size = path.stat().st_size
    if size != spec.size:
        return f"Model file {path} has {size} bytes; expected {spec.size} for {spec.filename}."
    actual = sha256_file(path)
    if actual != spec.sha256:
        return f"Model file {path} has SHA-256 {actual}; expected {spec.sha256} for {spec.filename}."
    return None
