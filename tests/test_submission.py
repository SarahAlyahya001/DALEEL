import io
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
from werkzeug.datastructures import FileStorage, MultiDict

from daleel import submission
from daleel.pipeline import srt_parser
from daleel.pipeline.srt_parser import SrtValidationError, parse_srt
from daleel.submission import (SubmissionStorageError, SubmissionValidationError,
                               validate_file_pair, validate_mp4, validate_submission)


def cue(payload="[latitude: 24.0] [longitude: 46.0]", timing="00:00:00,000 --> 00:00:01,000", number="1"):
    return f"{number}\n{timing}\n{payload}\n"


def box(kind, payload, extended=False):
    if extended:
        return struct.pack(">I4sQ", 1, kind, len(payload) + 16) + payload
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def container():
    return (box(b"ftyp", b"isom" + b"\0" * 4 + b"mp42")
            + box(b"free", b"") + box(b"moov", b"synthetic metadata")
            + box(b"mdat", b"synthetic media"))


def pair(video=b"synthetic", srt=None, video_name="flight.mp4", srt_name="different-name.srt"):
    return MultiDict([
        ("video", FileStorage(stream=io.BytesIO(video), filename=video_name)),
        ("srt", FileStorage(stream=io.BytesIO(cue().encode() if srt is None else srt), filename=srt_name)),
    ])


class FakeCapture:
    def __init__(self, open_ok=True, frames=2, reported=2, fail=None):
        self.open_ok, self.frames, self.reported, self.fail = open_ok, frames, reported, fail
        self.released = False
        self.reads = 0

    def open(self, path):
        if self.fail == "open":
            raise cv2.error("synthetic open error")
        return self.open_ok

    def isOpened(self):
        return self.open_ok

    def get(self, prop):
        if self.fail == "metadata":
            raise cv2.error("synthetic metadata error")
        return self.reported

    def read(self):
        if self.fail == "read":
            raise cv2.error("synthetic decode error")
        self.reads += 1
        if self.reads > self.frames:
            return False, None
        if self.fail == "empty":
            return True, None
        return True, np.zeros((16, 16, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def readable_capture(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr(submission.cv2, "VideoCapture", lambda: capture)
    return capture


@pytest.mark.parametrize("missing", ["video", "srt"])
def test_missing_file_is_rejected(missing):
    files = pair()
    del files[missing]
    with pytest.raises(SubmissionValidationError, match="required"):
        validate_file_pair(files)


@pytest.mark.parametrize("field", ["video", "srt"])
def test_empty_filename_and_duplicate_parts_rejected(field):
    files = pair()
    files[field].filename = ""
    with pytest.raises(SubmissionValidationError, match="required"):
        validate_file_pair(files)
    files = pair()
    files.add(field, files[field])
    with pytest.raises(SubmissionValidationError, match="Exactly one"):
        validate_file_pair(files)


def test_unexpected_file_part_is_rejected():
    files = pair()
    files.add("extra", FileStorage(stream=io.BytesIO(b"extra"), filename="extra.mp4"))
    with pytest.raises(SubmissionValidationError, match="Unexpected"):
        validate_file_pair(files)


@pytest.mark.parametrize("video_name,srt_name", [
    ("video.avi", "t.srt"), ("video.mov", "t.srt"), ("video.mp4.exe", "t.srt"),
    ("video.mp4", "t.txt"), ("video", "t.srt"), ("video.mp4", "t"),
])
def test_unsupported_extensions_rejected(video_name, srt_name):
    with pytest.raises(SubmissionValidationError, match="Unsupported"):
        validate_file_pair(pair(video_name=video_name, srt_name=srt_name))


def test_different_filenames_and_uppercase_extensions_are_allowed():
    validate_file_pair(pair(video_name="recording.MP4", srt_name="telemetry.SRT"))


@pytest.mark.parametrize("name", ["desert flight.mp4", "رحلة البحث.mp4", r"C:\fakepath\رحلة.MP4"])
def test_original_display_name_kept_separate_from_storage(tmp_path, readable_capture, name):
    import json
    from daleel.mission_storage import display_basename
    with submission.validated_submission(pair(video=container(), video_name=name), tmp_path, 1000, 2000) as root:
        assert set(path.name for path in root.iterdir()) == {"video.mp4", "telemetry.srt", "submission.json"}
        names = json.loads((root / "submission.json").read_text(encoding="utf-8"))
        assert names == {"video_name": display_basename(name), "srt_name": "different-name.srt"}
        assert (root / "video.mp4").read_bytes() == container()
    assert not root.exists()


@pytest.mark.parametrize("data", [b"", b"not mp4", b"RIFF" + b"\0" * 4 + b"AVI " + b"junk",
    struct.pack(">I4s", 7, b"ftyp"), struct.pack(">I4s", 100, b"ftyp") + b"isom",
    struct.pack(">I4s", 1, b"ftyp") + b"short",
    box(b"ftyp", b"qt  " + b"\0" * 4) + box(b"moov", b"x") + box(b"mdat", b"x"),
    box(b"ftyp", b"isom" + b"\0" * 4), container() + b"x",
    box(b"ftyp", b"isom" + b"\0" * 4) + box(b"moov", b"") + box(b"mdat", b"x"),
])
def test_invalid_container_rejected_before_decoder(tmp_path, monkeypatch, data):
    video = tmp_path / "invalid.mp4"
    video.write_bytes(data)
    monkeypatch.setattr(submission.cv2, "VideoCapture", lambda: pytest.fail("decoder must not be opened"))
    with pytest.raises(SubmissionValidationError, match="MP4 validation failed"):
        validate_mp4(video)


def test_extended_boxes_compatible_brand_and_eof_box(tmp_path, readable_capture):
    video = tmp_path / "valid.mp4"
    video.write_bytes(box(b"ftyp", b"other"[:4] + b"\0" * 4 + b"mp42", extended=True)
                      + box(b"moov", b"x", extended=True)
                      + struct.pack(">I4s", 0, b"mdat") + b"media")
    validate_mp4(video)
    assert readable_capture.released
    assert readable_capture.reads == 3


@pytest.mark.parametrize("kwargs", [dict(open_ok=False), dict(frames=0), dict(frames=1, reported=2),
    dict(fail="open"), dict(fail="metadata"), dict(fail="read"), dict(fail="empty")])
def test_every_video_handle_is_released_on_failure(tmp_path, monkeypatch, kwargs):
    video = tmp_path / "video.mp4"
    video.write_bytes(container())
    capture = FakeCapture(**kwargs)
    monkeypatch.setattr(submission.cv2, "VideoCapture", lambda: capture)
    with pytest.raises(SubmissionValidationError, match="MP4 validation failed"):
        validate_mp4(video)
    assert capture.released


def test_video_read_error_is_specific(tmp_path):
    with pytest.raises(SubmissionValidationError, match="file could not be read"):
        validate_mp4(tmp_path / "missing.mp4")


def test_real_mp4_decodes_and_real_avi_renamed_mp4_is_rejected(tmp_path):
    # Local synthetic media only; no downloads or external encoder executable.
    for suffix, codec in (("mp4", "mp4v"), ("avi", "MJPG")):
        video = tmp_path / f"synthetic.{suffix}"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*codec), 5.0, (32, 32))
        try:
            assert writer.isOpened(), f"Installed OpenCV cannot write the {codec} test fixture"
            for _ in range(3):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        finally:
            writer.release()
        if suffix == "mp4":
            validate_mp4(video)
        else:
            renamed = tmp_path / "renamed.mp4"
            renamed.write_bytes(video.read_bytes())
            with pytest.raises(SubmissionValidationError, match="MP4 validation failed"):
                validate_mp4(renamed)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_utf8_bom_and_harmless_extra_telemetry(tmp_path, encoding):
    path = tmp_path / "telemetry.srt"
    path.write_text(cue('<font size="28">SrtCnt: 1 [latitude: 24.0] [longitude: 46.0]\n'
                        '[iso: 100] [rel_alt: 60] extra telemetry</font>'), encoding=encoding)
    points = parse_srt(path)
    assert len(points) == 1
    assert (points[0].lat, points[0].lon, points[0].alt) == (24.0, 46.0, 60.0)


@pytest.mark.parametrize("payload,expected", [
    ("GPS(46.0,24.0,60)", (24, 46, 60)), ("GPS(46,24)", (24, 46, None)),
    ("[latitude: -90] [longitude: 180]", (-90, 180, None)),
    ("[latitude: +90] [longitude: -180] [rel_alt: n/a]", (90, -180, None)),
])
def test_supported_gps_and_boundary_values(tmp_path, payload, expected):
    path = tmp_path / "t.srt"
    path.write_text(cue(payload), encoding="utf-8")
    point = parse_srt(path)[0]
    assert (point.lat, point.lon, point.alt) == expected


@pytest.mark.parametrize("payload", [
    "[latitude: nan] [longitude: 46]", "[latitude: inf] [longitude: 46]",
    "[latitude: 1e999] [longitude: 46]", "[latitude: 91] [longitude: 46]",
    "[latitude: 24] [longitude: -181]", "[latitude: 24]", "[longitude: 46]",
    "[latitude: ] [longitude: 46]", "[latitude: 24oops] [longitude: 46]",
    "GPS(46,bad,60)", "GPS(181,24,60)", "GPS(46,24", "GPS(46)",
    "[latitude: 24] [longitude: 46] [latitude: 95]",
    "[latitude: 24] [longitude: 46] GPS(181,24,60)",
    "[latitude: 24] [longitude: 46] GPS(47,24,60)",
])
def test_invalid_coordinates_not_silently_skipped(tmp_path, payload):
    path = tmp_path / "t.srt"
    path.write_text(cue() + "\n" + cue(payload, number="2"), encoding="utf-8")
    with pytest.raises(SrtValidationError):
        parse_srt(path)


@pytest.mark.parametrize("text", ["", "not subtitles", cue(number="bad"), cue(number="0"),
    cue(timing="00:60:00,000 --> 00:61:00,000"), cue(timing="00:00:60,000 --> 00:01:01,000"),
    cue(timing="00:00:02,000 --> 00:00:01,000"), cue(timing="00:00:01,000 --> 00:00:01,000"),
    cue(timing="00:00:00,000 -> 00:00:01,000"), cue(payload=""), cue(payload="extra data only"),
    cue() + "\nmalformed block", cue(payload="\x00"),
])
def test_malformed_srt_rejected(tmp_path, text):
    path = tmp_path / "t.srt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SrtValidationError):
        parse_srt(path)


def test_srt_unreadable_encoding_and_missing_file(tmp_path):
    path = tmp_path / "t.srt"
    with pytest.raises(SrtValidationError, match="could not be read"):
        parse_srt(path)
    path.write_bytes(cue().encode("utf-16"))
    with pytest.raises(SrtValidationError, match="UTF-8"):
        parse_srt(path)


def test_cues_without_coordinates_allowed_when_a_usable_record_exists(tmp_path):
    path = tmp_path / "t.srt"
    path.write_text(cue("harmless telemetry without GPS") + "\n" + cue(number="2"), encoding="utf-8")
    assert len(parse_srt(path)) == 1


def test_additional_blank_lines_and_crlf_are_harmless(tmp_path):
    path = tmp_path / "t.srt"
    path.write_bytes((cue() + "\n\n \n" + cue(number="2")).replace("\n", "\r\n").encode())
    assert len(parse_srt(path)) == 2


def test_unrepresentable_cue_timestamp_has_validation_error(tmp_path):
    path = tmp_path / "t.srt"
    path.write_text(cue(timing="9" * 1000 + ":00:00,000 --> 00:00:01,000"), encoding="utf-8")
    with pytest.raises(SrtValidationError, match="invalid timing"):
        parse_srt(path)


def test_valid_submission_cleans_files_without_association(tmp_path, readable_capture, monkeypatch):
    sentinel = tmp_path / "existing-upload.txt"
    sentinel.write_text("preserve")
    monkeypatch.setattr(srt_parser, "coord_at", lambda *args: pytest.fail("Stage 2 must not associate coordinates"))
    from daleel.pipeline import runner
    monkeypatch.setattr(runner, "start_processing", lambda *args: pytest.fail("no processor"))
    validate_submission(pair(video=container(), video_name="../../escape.MP4"), tmp_path, 1000, 2000)
    assert readable_capture.released
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve"
    assert "coord_at" not in submission.__dict__


@pytest.mark.parametrize("video,srt,limit,error", [
    (b"", None, 1000, "empty"), (container(), b"", 1000, "empty"),
    (container(), b"bad subtitles", 1000, "SRT validation failed"),
    (b"bad video", None, 1000, "MP4 validation failed"),
    (container(), None, 2, "size limit"),
])
def test_invalid_submissions_leave_no_files(tmp_path, video, srt, limit, error):
    with pytest.raises(SubmissionValidationError, match=error):
        validate_submission(pair(video=video, srt=srt), tmp_path, limit, 2000)
    assert not list(tmp_path.iterdir())


def test_exact_video_size_limit_is_accepted(tmp_path, readable_capture):
    video = container()
    validate_submission(pair(video=video), tmp_path, len(video), 2000)
    assert not list(tmp_path.iterdir())


def test_srt_size_limit_leaves_no_files(tmp_path):
    with pytest.raises(SubmissionValidationError, match="size limit"):
        validate_submission(pair(), tmp_path, 1000, 2)
    assert not list(tmp_path.iterdir())


def test_partial_save_failure_cleans_request_files(tmp_path):
    class BrokenStream(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("synthetic storage/read failure")
            return super().read(1)
    files = pair()
    files["video"].stream = BrokenStream(b"partial")
    with pytest.raises(SubmissionStorageError, match="no mission was created"):
        validate_submission(files, tmp_path, 1000, 2000)
    assert not list(tmp_path.iterdir())


def test_unexpected_validation_exception_also_cleans_files(tmp_path, monkeypatch):
    def fail(path):
        raise RuntimeError("synthetic validator error")
    monkeypatch.setattr(submission, "validate_mp4", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        validate_submission(pair(), tmp_path, 1000, 2000)
    assert not list(tmp_path.iterdir())


def test_cleanup_failure_is_not_reported_as_validation_success(tmp_path, monkeypatch):
    original = submission.tempfile.TemporaryDirectory.cleanup
    def cleanup_then_fail(self):
        original(self)
        raise OSError("synthetic cleanup failure")
    monkeypatch.setattr(submission.tempfile.TemporaryDirectory, "cleanup", cleanup_then_fail)
    monkeypatch.setattr(submission, "validate_mp4", lambda path: None)
    with pytest.raises(SubmissionStorageError, match="cleanup failed"):
        validate_submission(pair(), tmp_path, 1000, 2000)
    assert not list(tmp_path.iterdir())


def test_request_directories_are_unique_and_only_the_current_one_is_cleaned(tmp_path, monkeypatch):
    visited = []
    def nested_validation(path):
        visited.append(Path(path).parent)
        assert Path(path).parent.parent == tmp_path
        if len(visited) == 1:
            validate_submission(pair(), tmp_path, 1000, 2000)
            assert Path(path).exists()  # inner cleanup cannot delete outer media
    monkeypatch.setattr(submission, "validate_mp4", nested_validation)
    validate_submission(pair(), tmp_path, 1000, 2000)
    assert len(set(visited)) == 2
    assert not list(tmp_path.iterdir())


def test_validated_context_exposes_only_valid_staging_and_cleans_after_consumer_error(tmp_path, readable_capture):
    from daleel.submission import validated_submission
    root = None
    with pytest.raises(RuntimeError, match="consumer"):
        with validated_submission(pair(video=container()), tmp_path, 1000, 2000) as root:
            assert (root / "video.mp4").read_bytes() == container()
            assert (root / "telemetry.srt").is_file()
            raise RuntimeError("consumer failed before accepting mission")
    assert root is not None and not root.exists()
    assert list(tmp_path.iterdir()) == []
