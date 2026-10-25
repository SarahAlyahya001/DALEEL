"""File validation and request-owned staging; no coordinate association."""

import math
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2

from .pipeline.srt_parser import SrtValidationError, parse_srt
from .mission_storage import display_basename, save_submission_names


class SubmissionValidationError(ValueError):
    """The submitted files are missing, unsupported, or invalid."""


class SubmissionStorageError(RuntimeError):
    """Temporary storage or cleanup failed; no mission can be created."""


# A deliberately limited MP4/ISO-BMFF compatibility check, not certification.
# Supported file validation cases and limits are exercised in tests/test_submission.py.
MP4_BRANDS = {b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"iso7",
              b"iso8", b"iso9", b"mp41", b"mp42", b"avc1", b"dash"}


def validate_file_pair(files):
    if set(files) - {"video", "srt"}:
        raise SubmissionValidationError("Unexpected file field: submit one MP4 video and one SRT file only.")
    for field, label, suffix in (("video", "MP4 video", ".mp4"), ("srt", "SRT telemetry", ".srt")):
        items = files.getlist(field)
        if not items or not items[0].filename:
            raise SubmissionValidationError(f"The {label} file is required.")
        if len(items) != 1:
            raise SubmissionValidationError(f"Exactly one {label} file is required.")
        name = display_basename(items[0].filename)
        if not name or Path(name).suffix.lower() != suffix:
            raise SubmissionValidationError(f"Unsupported {label} format: only {suffix} is accepted.")
    return files["video"], files["srt"]


def _save_bounded(upload, path, limit, label):
    total = 0
    # Server-generated path inside a newly created request directory.
    with path.open("xb") as target:
        while True:
            chunk = upload.stream.read(min(1024 * 1024, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise SubmissionValidationError(f"{label} exceeds the configured size limit.")
            target.write(chunk)
    if not total:
        raise SubmissionValidationError(f"{label} file is empty.")


def _validate_mp4_container(path):
    """Check top-level box bounds, MP4 branding, and movie/media presence.

    Nested metadata and codec validity are delegated to OpenCV below. This
    supports ordinary self-contained MP4 files, not arbitrary ISO-BMFF formats.
    """
    size = path.stat().st_size
    if size < 8:
        raise SubmissionValidationError("MP4 validation failed: empty or truncated container.")
    found = set()
    with path.open("rb") as source:
        offset = 0
        while offset < size:
            source.seek(offset)
            header = source.read(8)
            if len(header) != 8:
                raise SubmissionValidationError("MP4 validation failed: truncated box header.")
            box_size, kind = struct.unpack(">I4s", header)
            header_size = 8
            if box_size == 1:
                extended = source.read(8)
                if len(extended) != 8:
                    raise SubmissionValidationError("MP4 validation failed: truncated extended box header.")
                box_size = struct.unpack(">Q", extended)[0]
                header_size = 16
            elif box_size == 0:
                box_size = size - offset
            if box_size < header_size or offset + box_size > size:
                raise SubmissionValidationError("MP4 validation failed: invalid or truncated container box.")
            payload_size = box_size - header_size
            if kind == b"ftyp":
                if kind in found or payload_size < 8 or payload_size % 4:
                    raise SubmissionValidationError("MP4 validation failed: invalid file-type box.")
                major = source.read(4)
                source.read(4)  # minor version
                compatible = major in MP4_BRANDS
                for _ in range((payload_size - 8) // 4):
                    compatible = source.read(4) in MP4_BRANDS or compatible
                if not compatible:
                    raise SubmissionValidationError("MP4 validation failed: unsupported container brand.")
            if kind in (b"moov", b"mdat") and not payload_size:
                raise SubmissionValidationError("MP4 validation failed: empty movie or media box.")
            found.add(kind)
            offset += box_size
    if not {b"ftyp", b"moov", b"mdat"} <= found:
        raise SubmissionValidationError("MP4 validation failed: required MP4 container boxes are missing.")


def validate_mp4(path):
    path = Path(path)
    if path.suffix.lower() != ".mp4":
        raise SubmissionValidationError("Unsupported video format: only .mp4 is accepted.")
    try:
        _validate_mp4_container(path)
        # Create the handle before opening so finally also covers open failure.
        capture = cv2.VideoCapture()
        try:
            if not capture.open(str(path)) or not capture.isOpened():
                raise SubmissionValidationError("MP4 validation failed: video could not be opened.")
            reported_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            decoded_frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or frame.size == 0:
                    raise SubmissionValidationError("MP4 validation failed: unreadable video frame.")
                decoded_frames += 1
            if not decoded_frames:
                raise SubmissionValidationError("MP4 validation failed: no decodable video frames.")
            if (math.isfinite(reported_frames) and reported_frames > 0
                    and decoded_frames < int(reported_frames)):
                raise SubmissionValidationError("MP4 validation failed: video ended before all reported frames decoded.")
        finally:
            capture.release()
    except cv2.error as exc:
        raise SubmissionValidationError("MP4 validation failed: video decoder could not read the content.") from exc
    except OSError as exc:
        raise SubmissionValidationError("MP4 validation failed: file could not be read.") from exc


@contextmanager
def validated_submission(files, upload_root, video_max_bytes, request_max_bytes):
    """Yield valid temporary inputs, then remove only this request's staging."""
    video, srt = validate_file_pair(files)
    try:
        # No client filename becomes a disk path. Cleanup owns only this new,
        # uniquely named child of the configured upload directory.
        with tempfile.TemporaryDirectory(prefix="validation-", dir=upload_root) as temporary:
            root = Path(temporary)
            video_path, srt_path = root / "video.mp4", root / "telemetry.srt"
            _save_bounded(srt, srt_path, request_max_bytes, "SRT telemetry")
            try:
                parse_srt(str(srt_path))  # Parse records only; never call coord_at.
            except SrtValidationError as exc:
                raise SubmissionValidationError(f"SRT validation failed: {exc}") from exc
            _save_bounded(video, video_path, video_max_bytes, "MP4 video")
            validate_mp4(video_path)
            save_submission_names(root, video.filename, srt.filename)
            yield root
    except OSError as exc:
        raise SubmissionStorageError("Temporary upload storage or cleanup failed; no mission was created.") from exc


def validate_submission(files, upload_root, video_max_bytes, request_max_bytes):
    """Validation-only helper; no mission, worker, or retained files."""
    with validated_submission(files, upload_root, video_max_bytes, request_max_bytes):
        pass
