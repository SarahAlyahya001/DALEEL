"""The demonstration mission in demo/ must be accepted by the upload page as it is."""
from pathlib import Path

import cv2
from werkzeug.datastructures import FileStorage, MultiDict

from daleel.pipeline.srt_parser import parse_srt
from daleel.submission import validate_submission

DEMO = Path(__file__).resolve().parents[1] / "demo"
VIDEO, TELEMETRY = DEMO / "desert_search.mp4", DEMO / "desert_search.srt"


def test_demo_pair_passes_upload_validation(tmp_path):
    with VIDEO.open("rb") as video, TELEMETRY.open("rb") as srt:
        files = MultiDict([("video", FileStorage(video, filename=VIDEO.name)),
                           ("srt", FileStorage(srt, filename=TELEMETRY.name))])
        validate_submission(files, tmp_path, 500 * 1024 * 1024, 505 * 1024 * 1024)
    assert list(tmp_path.iterdir()) == []


def test_demo_telemetry_covers_the_video_inside_saudi_arabia():
    capture = cv2.VideoCapture(str(VIDEO))
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    points = parse_srt(str(TELEMETRY))
    assert points[0].t == 0 and duration - 1 < points[-1].t < duration
    assert all(16.0 <= point.lat <= 32.5 and 34.0 <= point.lon <= 56.0 for point in points)
