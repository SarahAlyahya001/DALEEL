"""Demo telemetry must follow the DJI GPS(longitude, latitude, altitude) order.

Reading the two values in the opposite order would place every demo observation
near 45.07 N, 21.42 E instead of 21.42 N, 45.07 E, outside the search area.
"""
from pathlib import Path

import pytest

from daleel.pipeline.srt_parser import parse_srt

DEMO = Path(__file__).parent / "data" / "desert_demo_telemetry.srt"
# Approximate bounding box of Saudi Arabia, the intended demo search area.
SAUDI_LATITUDE = (16.0, 32.5)
SAUDI_LONGITUDE = (34.0, 56.0)


def in_saudi_arabia(point):
    return (SAUDI_LATITUDE[0] <= point.lat <= SAUDI_LATITUDE[1]
            and SAUDI_LONGITUDE[0] <= point.lon <= SAUDI_LONGITUDE[1])


def test_demo_telemetry_places_every_cue_in_the_desert_search_area():
    points = parse_srt(DEMO)
    assert len(points) == 270
    assert (points[0].lat, points[0].lon, points[0].alt) == (21.4235, 45.0712, 312.0)
    assert all(in_saudi_arabia(point) for point in points)


@pytest.mark.parametrize("gps,expected,inside", [
    ("GPS (45.0712, 21.4235, 312)", (21.4235, 45.0712, 312.0), True),   # DJI order
    ("GPS (21.4235, 45.0712, 312)", (45.0712, 21.4235, 312.0), False),  # latitude first: outside the area
])
def test_demo_style_gps_line_is_read_longitude_first(tmp_path, gps, expected, inside):
    path = tmp_path / "telemetry.srt"
    path.write_text("1\n00:00:00,000 --> 00:00:01,000\n"
                    f"F/2.8, SS 500, ISO 100, EV 0, {gps}, D 0.00m, H 50.00m, HS 0.00m/s, VS 0.00m/s\n",
                    encoding="utf-8")
    [point] = parse_srt(path)
    assert (point.lat, point.lon, point.alt) == expected
    assert in_saudi_arabia(point) is inside
