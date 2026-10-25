"""Offline export tests: no network, inference, real uploads, or real databases."""
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from lxml import etree
from PIL import Image
from pypdf import PdfReader

from daleel.exports import (GPX_NAMESPACE, build_gpx, build_pdf, coordinate_pair,
                            format_score, format_video_clock, format_video_time)


SCHEMA = Path(__file__).parent / "schemas/gpx-1.1.xsd"
SCHEMA_HASH = "9e4d1988b862edbe556305b130f8f6f1b29864fefd0dc02d5dab04ccdd1f34d6"


class NoExternalResources(etree.Resolver):
    def resolve(self, url, public_id, context):
        raise AssertionError("External XML resources are forbidden in tests.")


@pytest.fixture
def schema():
    assert sha256(SCHEMA.read_bytes()).hexdigest() == SCHEMA_HASH
    parser = etree.XMLParser(no_network=True, resolve_entities=False, load_dtd=False)
    parser.resolvers.add(NoExternalResources())
    return etree.XMLSchema(etree.fromstring(SCHEMA.read_bytes(), parser))


def validate(schema, data):
    parser = etree.XMLParser(no_network=True, resolve_entities=False, load_dtd=False)
    parser.resolvers.add(NoExternalResources())
    document = etree.fromstring(data, parser)
    schema.assertValid(document)
    return document


def mission():
    return SimpleNamespace(mission_id=42, submitted_at=datetime(2026, 9, 12, 10, 0),
                           status="Complete", yolov8n_frames=17, yolov8s_frames=17, elapsed_seconds=2.75,
                           video_filename="uploads/mission-test/video.mp4",
                           srt_filename="uploads/mission-test/telemetry.srt")


def observation(identifier=1, **values):
    defaults = dict(observation_id=identifier, video_time_ms=1500.0, first_frame_index=15,
                    last_frame_index=45, frame_count=3, target_class="person",
                    agreement="Both models", yolov8n_score=0.81, yolov8s_score=0.72,
                    latitude=24.5, longitude=46.5, review_status="Confirmed",
                    frame_path="frames/evidence.jpg")
    defaults.update(values)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize("latitude,longitude,expected_lat,expected_lon", [
    (0, -180, "0", "-180"), (0, 180, "0", "-180"),
    (90, 179.99999999999997, "90", "179.99999999999997"),
    (-90, -179.99999999999997, "-90", "-179.99999999999997"),
    (1e-12, -1e-15, "0.000000000001", "-0.000000000000001"),
    (Decimal("0.10000"), Decimal("179.999999999999999999999"), "0.1", "179.999999999999999999999"),
])
def test_gpx_decimal_boundaries(schema, latitude, longitude, expected_lat, expected_lon):
    item = observation(latitude=latitude, longitude=longitude)
    data = build_gpx(mission(), [item]).getvalue()
    root = validate(schema, data)
    point = root.find(f"{{{GPX_NAMESPACE}}}wpt")
    assert point.attrib == {"lat": expected_lat, "lon": expected_lon}
    assert item.longitude == longitude
    assert Decimal("-180") <= Decimal(point.attrib["lon"]) < Decimal("180")
    assert root.attrib == {"version": "1.1", "creator": "DALEEL"}


@pytest.mark.parametrize("latitude,longitude", [
    (None, 1), (1, None), (None, None), (float("nan"), 1), (1, float("nan")),
    (float("inf"), 1), (1, float("-inf")), (90.01, 1), (-90.01, 1),
    (1, 180.01), (1, -180.01), ("invalid", 1), (True, 1),
])
def test_gpx_invalid_coordinates_excluded(schema, latitude, longitude):
    assert coordinate_pair(latitude, longitude) is None
    root = validate(schema, build_gpx(mission(), [observation(latitude=latitude, longitude=longitude)]).getvalue())
    assert len(root) == 0


def test_gpx_filter_order_escaping_and_no_fabrication(schema):
    items = [observation(8, video_time_ms=200), observation(3, video_time_ms=100),
             observation(2, video_time_ms=100, target_class='person & <synthetic> "quoted"'),
             observation(4, review_status="Unreviewed"), observation(5, review_status="Rejected"),
             observation(6, longitude=None)]
    data = build_gpx(mission(), items).getvalue()
    root = validate(schema, data)
    ns = {"g": GPX_NAMESPACE}
    assert [n.text for n in root.findall("g:wpt/g:name", ns)] == ["Observation 1", "Observation 2", "Observation 3"]
    assert "&amp;" in data.decode() and "&lt;synthetic&gt;" in data.decode()
    assert 'person & <synthetic> "quoted"' in root.find("g:wpt/g:desc", ns).text
    for point in root:
        assert [child.tag.split("}")[1] for child in point] == ["name", "desc"]
        assert "Drone-Position Approximation" in point.find("g:desc", ns).text
    assert root.find(".//g:ele", ns) is None and root.find(".//g:time", ns) is None


@pytest.mark.parametrize("items", [[], [observation(review_status="Unreviewed")],
                                  [observation(review_status="Rejected")]])
def test_zero_waypoints(schema, items):
    root = validate(schema, build_gpx(mission(), items).getvalue())
    assert len(root) == 0


def test_gpx_number_comes_from_full_mission_order_not_filtered_waypoint_order(schema):
    items = [observation(50, video_time_ms=100, review_status="Unreviewed"),
             observation(99, video_time_ms=200, review_status="Confirmed")]
    root = validate(schema, build_gpx(mission(), items).getvalue())
    name = root.find(f"{{{GPX_NAMESPACE}}}wpt/{{{GPX_NAMESPACE}}}name")
    assert name.text == "Observation 2"


def test_schema_rejects_wrong_namespace_order_and_unormalized_longitude(schema):
    good = build_gpx(mission(), [observation()]).getvalue()
    for bad in [good.replace(GPX_NAMESPACE.encode(), b"https://invalid.example/gpx"),
                good.replace(b'lon="46.5"', b'lon="180"'),
                good.replace(b"</wpt>", b"<name>Wrong ordering</name></wpt>")]:
        with pytest.raises(etree.DocumentInvalid):
            validate(schema, bad)


def test_external_entities_are_not_resolved(schema, tmp_path):
    # Parser does not substitute an external file entity; schema rejects the unresolved node.
    secret = tmp_path / "synthetic.txt"
    secret.write_text("must-not-appear", encoding="utf-8")
    parser = etree.XMLParser(no_network=True, resolve_entities=False, load_dtd=False)
    parser.resolvers.add(NoExternalResources())
    xml = f'<!DOCTYPE gpx [<!ENTITY external SYSTEM "{secret.as_uri()}">]><gpx xmlns="{GPX_NAMESPACE}" version="1.1" creator="DALEEL"><metadata><name>&external;</name></metadata></gpx>'
    root = etree.fromstring(xml.encode(), parser)
    assert b"must-not-appear" not in etree.tostring(root)


def pdf_text(pdf):
    """Extracted PDF text with whitespace normalized, so table cells read as one line."""
    return " ".join("\n".join(page.extract_text() for page in PdfReader(pdf).pages).split())


def test_pdf_contains_all_states_scores_metadata_and_missing_evidence(tmp_path):
    items = [observation(1, review_status="Unreviewed"), observation(2, review_status="Confirmed"),
             observation(3, review_status="Rejected", target_class="car", agreement="One model",
                         yolov8s_score=None, latitude=None)]
    pdf = build_pdf(mission(), items, tmp_path, tmp_path / "uploads",
                    names={"video_name": "desert_search.mp4", "srt_name": "desert_search.srt"},
                    analyst="Synthetic Analyst")
    text_content = pdf_text(pdf)
    for expected in ("Mission Summary", "Mission 42", "M-042", "STATUS Complete",
                     "VIDEO FILE desert_search.mp4", "TELEMETRY FILE desert_search.srt",
                     "ANALYST Synthetic Analyst", "SUBMITTED 12 Sep 2026, 1:00 PM AST",
                     "PROCESSING TIME 3 seconds", "17 sampled frames by YOLOv8n, 17 by YOLOv8s",
                     "3 OBSERVATIONS", "1 CONFIRMED", "1 REJECTED", "1 UNREVIEWED",
                     "Observation 1 · Person", "Observation 2 · Person", "Observation 3 · Vehicle",
                     "Model agreement Both models", "Model agreement One model",
                     "YOLOv8n confidence 81%", "YOLOv8s confidence 72%", "YOLOv8s confidence —", "supporting information",
                     "Video time 00:00:01.500", "Seen in 3 sampled frames (frames 15 to 45)",
                     "Drone-Position Approximation", "Drone location 24.500000, 46.500000",
                     "Drone location Location unavailable",
                     "Review status Unreviewed", "Review status Confirmed", "Review status Rejected",
                     "Representative evidence unavailable", "Page 1 of"):
        assert expected in text_content
    # Storage paths are never shown to the reader.
    assert "uploads/" not in text_content and "telemetry.srt" not in text_content


@pytest.mark.parametrize("image_kind", ["valid", "corrupt", "missing"])
def test_pdf_evidence_truthful_without_dropping_observation(tmp_path, image_kind):
    folder = tmp_path / "uploads/mission-test"
    (folder / "frames").mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"synthetic")
    (folder / "telemetry.srt").write_bytes(b"synthetic")
    if image_kind == "valid":
        Image.new("RGB", (100, 50), "white").save(folder / "frames/evidence.jpg")
    elif image_kind == "corrupt":
        (folder / "frames/evidence.jpg").write_bytes(b"corrupt")
    document = PdfReader(build_pdf(mission(), [observation()], tmp_path, tmp_path / "uploads"))
    text_content = "\n".join(page.extract_text() for page in document.pages)
    assert "Observation 1" in text_content
    assert ("Representative evidence unavailable" in text_content) == (image_kind != "valid")
    if image_kind == "valid":
        assert any(page.images for page in document.pages)


def test_pdf_order_and_dynamic_markup_escaped(tmp_path):
    items = [observation(3, video_time_ms=200), observation(2, video_time_ms=100),
             observation(1, video_time_ms=100)]
    names = {"video_name": 'synthetic & <b>not markup</b>.mp4', "srt_name": "flight.srt"}
    result = pdf_text(build_pdf(mission(), items, tmp_path, tmp_path / "uploads", names=names))
    assert result.index("Observation 1") < result.index("Observation 2") < result.index("Observation 3")
    assert "synthetic & <b>not markup</b>.mp4" in result


def test_presentation_formatting_removes_float_noise_and_limits_score_precision():
    assert format_video_time(3500.0000000000005) == "3500 ms"
    assert format_video_time(1234.56789) == "1234.568 ms"
    assert format_score(0.4114939272403717) == "0.411"
    assert format_score(None) == "Not available"
    assert format_video_clock(15000.000000000002) == "00:15"
    assert format_video_clock(1500) == "00:01.5"
    assert format_video_clock(3_661_250) == "01:01:01.25"


@pytest.mark.parametrize("value,expected", [
    (0, "00:00:00.000"), (2500, "00:00:02.500"), (60_000, "00:01:00.000"),
    (3_661_250, "01:01:01.250"), (3_599_999.6, "01:00:00.000"),
    (None, "Unavailable"), (float("nan"), "Unavailable"),
    (float("inf"), "Unavailable"), (-1, "Unavailable"),
])
def test_full_video_clock_has_explicit_units_and_safe_rounding(value, expected):
    assert format_video_clock(value, full=True) == expected
