"""Explicitly obtain or verify DALEEL's two fine-tuned model files from the project's GitHub Release."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daleel.pipeline.weights import UNCONFIGURED, WEIGHT_SPECS, WeightSpec, sha256_file  # noqa: E402


class WeightSetupError(RuntimeError):
    """Raised when a model file is absent, damaged, or unsafe to replace."""


def verify_weight(path: Path, spec: WeightSpec) -> None:
    if not path.is_file():
        raise WeightSetupError(f"Required model weight is missing: {path}")
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise WeightSetupError(f"Could not inspect model weight {path}: {exc}") from exc
    if actual_size != spec.size:
        raise WeightSetupError(
            f"Model weight {path} has {actual_size} bytes; expected {spec.size}. "
            "The existing file was not replaced."
        )
    try:
        actual_hash = sha256_file(path)
    except OSError as exc:
        raise WeightSetupError(f"Could not read model weight {path}: {exc}") from exc
    if actual_hash != spec.sha256:
        raise WeightSetupError(
            f"Model weight {path} has SHA-256 {actual_hash}; expected {spec.sha256}. "
            "The existing file was not replaced."
        )


def verify_weights(
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    specs: Iterable[WeightSpec] = WEIGHT_SPECS,
) -> dict[str, Path]:
    weights_dir = Path(weights_dir).resolve()
    verified: dict[str, Path] = {}
    for spec in specs:
        destination = weights_dir / spec.filename
        verify_weight(destination, spec)
        verified[spec.filename] = destination
    return verified


def _download_to_temporary_file(
    spec: WeightSpec,
    weights_dir: Path,
    opener: Callable = urllib.request.urlopen,
) -> Path:
    if UNCONFIGURED in spec.url:
        raise WeightSetupError("The release repository is not configured in daleel/pipeline/weights.py.")
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "DALEEL-weight-setup/2"},
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{spec.filename}.", suffix=".part", dir=weights_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            with opener(request, timeout=120) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
            target.flush()
            os.fsync(target.fileno())
        verify_weight(temporary, spec)
        return temporary
    except WeightSetupError:
        temporary.unlink(missing_ok=True)
        raise
    except (KeyboardInterrupt, SystemExit):
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise WeightSetupError(f"Could not download {spec.filename}: {exc}") from exc


def ensure_weights(
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    specs: Iterable[WeightSpec] = WEIGHT_SPECS,
    opener: Callable = urllib.request.urlopen,
) -> dict[str, Path]:
    weights_dir = Path(weights_dir).resolve()
    try:
        weights_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WeightSetupError(f"Could not create weight directory {weights_dir}: {exc}") from exc
    verified: dict[str, Path] = {}
    for spec in specs:
        destination = weights_dir / spec.filename
        if destination.exists():
            verify_weight(destination, spec)
            verified[spec.filename] = destination
            continue

        temporary = _download_to_temporary_file(spec, weights_dir, opener)
        try:
            # A hard-link publication is atomic and refuses to replace a file
            # that appeared while the download was in progress.
            os.link(temporary, destination)
        except FileExistsError:
            verify_weight(destination, spec)
        except OSError as exc:
            raise WeightSetupError(
                f"Could not publish verified model weight {destination}: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        verify_weight(destination, spec)
        verified[spec.filename] = destination
    return verified


def _deny_network(event: str, _args) -> None:
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyaddr", "socket.gethostbyname"}:
        raise RuntimeError(f"Network access is forbidden during the offline model load check ({event}).")


def load_models_offline(
    paths: dict[str, Path],
    specs: Iterable[WeightSpec] = WEIGHT_SPECS,
    detector_factory=None,
    enforce_offline: bool = True,
) -> None:
    """Construct both models through the same DualDetector path used at runtime."""
    specs = tuple(specs)
    if len(specs) != 2:
        raise WeightSetupError("The offline load check requires exactly two model specifications.")
    if enforce_offline:
        sys.addaudithook(_deny_network)
    if detector_factory is None:
        from daleel.pipeline.detector import DualDetector
        detector_factory = DualDetector
    try:
        detector_factory(paths[specs[0].filename], paths[specs[1].filename], specs=specs)
    except Exception as exc:
        raise WeightSetupError(
            f"Offline model load check failed for {specs[0].filename} or {specs[1].filename}: {exc}"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help="Destination directory (default: repository weights directory).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files without network access or modification.",
    )
    parser.add_argument(
        "--load-check",
        action="store_true",
        help="Load both verified models through DALEEL's runtime detector without inference or network access.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = (
            verify_weights(args.weights_dir)
            if args.verify_only
            else ensure_weights(args.weights_dir)
        )
    except WeightSetupError as exc:
        raise SystemExit(f"Weight setup failed: {exc}") from exc

    action = "Verified" if args.verify_only else "Prepared"
    for spec in WEIGHT_SPECS:
        print(f"{action} {paths[spec.filename]} ({spec.size} bytes, SHA-256 {spec.sha256})")
    if args.load_check:
        try:
            load_models_offline(paths)
        except WeightSetupError as exc:
            raise SystemExit(f"Weight setup failed: {exc}") from exc
        print("Offline runtime-path load check succeeded for "
              f"{WEIGHT_SPECS[0].filename} and {WEIGHT_SPECS[1].filename}; no inference ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
