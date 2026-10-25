"""Read-only, contained access to server-generated mission evidence."""
from io import BytesIO
from pathlib import Path, PureWindowsPath

from PIL import Image, UnidentifiedImageError


class EvidenceUnavailable(Exception):
    """Intentionally contains no filesystem details."""


def _relative(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EvidenceUnavailable()
    if PureWindowsPath(value).drive or value.startswith(("/", "\\")):
        raise EvidenceUnavailable()
    parts = value.replace("\\", "/").split("/")
    for part in parts:
        if (part in ("", ".", "..") or part.endswith((" ", "."))
                or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
                or part.split(".")[0].upper() in
                {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                 *[f"LPT{i}" for i in range(1, 10)]}):
            raise EvidenceUnavailable()
    return Path(*parts)


def load_evidence(mission, observation, instance_path, upload_root):
    """Return verified image bytes and MIME type; never use request paths.

    Resolve both persisted input paths and evidence beneath the configured
    upload root. Reading once prevents a later image-library reopen by path.
    """
    try:
        instance = Path(instance_path).resolve(strict=True)
        root = Path(upload_root).resolve(strict=True)
        root.relative_to(instance)
        video = (instance / _relative(mission.video_filename)).resolve(strict=True)
        srt = (instance / _relative(mission.srt_filename)).resolve(strict=True)
        folder = video.parent
        folder.relative_to(root)
        if folder == root or srt.parent != folder or not video.is_file() or not srt.is_file():
            raise EvidenceUnavailable()
        evidence = (folder / _relative(observation.frame_path)).resolve(strict=True)
        evidence.relative_to(folder)
        if not evidence.is_file() or evidence.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise EvidenceUnavailable()
        data = evidence.read_bytes()
        with Image.open(BytesIO(data)) as picture:
            if picture.format not in {"JPEG", "PNG"}:
                raise EvidenceUnavailable()
            mime = "image/jpeg" if picture.format == "JPEG" else "image/png"
            picture.verify()
        with Image.open(BytesIO(data)) as picture:
            picture.load()
        return data, mime
    except (OSError, ValueError, RuntimeError, UnidentifiedImageError,
            Image.DecompressionBombError, SyntaxError):
        raise EvidenceUnavailable() from None
