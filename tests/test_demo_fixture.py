import re

import pytest
from werkzeug.datastructures import FileStorage, MultiDict

from daleel.pipeline.srt_parser import parse_srt
from daleel.submission import validate_submission
from scripts import make_fixture

CUE_TIMES = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})$", re.M)


def _milliseconds(hours, minutes, seconds, millis):
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(millis)


def test_generated_srt_has_one_positive_cue_per_video_second(tmp_path):
    srt = tmp_path / "fixture.srt"
    make_fixture.write_srt(srt)
    cues = CUE_TIMES.findall(srt.read_text(encoding="utf-8"))
    assert [_milliseconds(*cue[4:]) - _milliseconds(*cue[:4]) for cue in cues] == [1000] * make_fixture.SECONDS
    points = parse_srt(str(srt))
    assert [point.t for point in points] == [float(second) for second in range(make_fixture.SECONDS)]
    assert (points[0].lat, points[0].lon) == pytest.approx((make_fixture.BASE_LAT, make_fixture.BASE_LON))


def test_generated_demo_pair_passes_upload_validation(tmp_path):
    output, uploads = tmp_path / "fixtures", tmp_path / "uploads"
    uploads.mkdir()
    make_fixture.main(["--out", str(output)])
    with (output / "fixture.mp4").open("rb") as video, (output / "fixture.srt").open("rb") as srt:
        files = MultiDict([("video", FileStorage(video, filename="fixture.mp4")),
                           ("srt", FileStorage(srt, filename="fixture.srt"))])
        validate_submission(files, uploads, 500 * 1024 * 1024, 505 * 1024 * 1024)
    assert list(uploads.iterdir()) == []
