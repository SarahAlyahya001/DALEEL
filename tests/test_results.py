"""Results, review and export routes for owned missions, using temporary databases
and media and no model loading."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from sqlalchemy import text
from werkzeug.datastructures import MultiDict
from werkzeug.security import generate_password_hash

from daleel import create_app, initialize_database
from daleel.models import Mission, Observation, User, db


@pytest.fixture
def env(tmp_path):
    app = create_app(str(tmp_path / "instance"), {"TESTING": True})
    folder = Path(app.config["UPLOAD_DIR"]) / "mission-test"
    (folder / "frames").mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"synthetic input; never decoded")
    (folder / "telemetry.srt").write_bytes(b"synthetic input; never parsed")
    Image.new("RGB", (32, 24), "white").save(folder / "frames/evidence.jpg")
    with app.app_context():
        initialize_database()
        users = [User(username=name, password_hash=generate_password_hash("synthetic-test-only"), role=role)
                 for name, role in (("owner", "analyst"), ("other", "analyst"),
                                    ("admin", "administrator"))]
        db.session.add_all(users)
        db.session.flush()
        mission = Mission(analyst_id=users[0].user_id, status="Complete",
                          video_filename="uploads/mission-test/video.mp4",
                          srt_filename="uploads/mission-test/telemetry.srt",
                          yolov8n_frames=5, yolov8s_frames=5, elapsed_seconds=1.5)
        db.session.add(mission)
        db.session.flush()
        observations = []
        for index, state in enumerate(("Unreviewed", "Confirmed", "Rejected")):
            observation = Observation(
                mission_id=mission.mission_id, video_time_ms=100 if index else 200,
                first_frame_index=0, last_frame_index=30, frame_count=3,
                frame_path="frames/evidence.jpg", target_class="person" if index else "car",
                agreement="Both models", yolov8n_score=0.8, yolov8s_score=0.7,
                latitude=24 if index != 2 else None, longitude=46, review_status=state)
            db.session.add(observation)
            observations.append(observation)
        db.session.commit()
        result = SimpleNamespace(app=app, folder=folder, mission=mission.mission_id,
                                 ids=[o.observation_id for o in observations],
                                 users=[u.user_id for u in users])
    result.client = app.test_client()
    with result.client.session_transaction() as session:
        session["user_id"], session["_csrf_token"] = result.users[0], "synthetic-token"
    yield result
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def review(env, decision="Confirmed", observation=None):
    return env.client.post(f"/observations/{observation or env.ids[0]}/review",
                           data={"_csrf_token": "synthetic-token", "review_status": decision})


@pytest.mark.parametrize("kind", ["results", "frame", "pdf", "gpx", "review"])
@pytest.mark.parametrize("user_index,code", [(1, 404), (2, 403)])
def test_owner_only(env, kind, user_index, code):
    with env.client.session_transaction() as session:
        session["user_id"] = env.users[user_index]
    paths = {"results": f"/results/{env.mission}", "frame": f"/frame/{env.ids[0]}",
             "pdf": f"/export/{env.mission}/pdf", "gpx": f"/export/{env.mission}/gpx"}
    response = review(env) if kind == "review" else env.client.get(paths[kind])
    assert response.status_code == code
    assert b"mission-test" not in response.data
    assert b"yolov8n_score" not in response.data
    assert response.headers["Cache-Control"] == "private, no-store"


@pytest.mark.parametrize("state", ["Processing", "Failed"])
def test_noncomplete_gates(env, state):
    with env.app.app_context():
        db.session.get(Mission, env.mission).status = state
        db.session.commit()
    for endpoint in (f"/results/{env.mission}", f"/export/{env.mission}/pdf",
                     f"/export/{env.mission}/gpx"):
        response = env.client.get(endpoint)
        assert response.status_code == 302
        assert response.location.endswith(f"/status/{env.mission}")
    assert env.client.get(f"/frame/{env.ids[0]}").status_code == 404
    assert review(env).status_code == 409
    assert state.encode() in env.client.get(f"/status/{env.mission}").data


def test_results_cards_order_map_and_safe_payload(env):
    response = env.client.get(f"/results/{env.mission}")
    assert response.status_code == 200
    html = response.data.decode()
    expected = [env.ids[1], env.ids[2], env.ids[0]]
    assert [int(x) for x in re.findall(r'<article class="observation-card" id="observation-(\d+)"', html)] == expected
    assert re.findall(r'<p class="observation-number">Observation (\d+)</p>', html) == ["1", "2", "3"]
    for label in ("Mission Results", "Observations", "Unreviewed", "Confirmed", "Rejected",
                  "Video timestamp", "Supporting frames", "Person", "Vehicle",
                  "Show on map", "Evidence map"):
        assert label in html
    for technical_label in ("Model support", "Support label", "Screening score", "Verification score",
                            "First source-frame index", "Last source-frame index"):
        assert technical_label not in html
    payload = json.loads(re.search(r'<script id="map-data"[^>]*>(.*?)</script>', html, re.S).group(1))
    assert [o["observation_number"] for o in payload] == [1, 3]
    assert all(set(o) == {"observation_number", "anchor", "latitude", "longitude", "target_class",
                          "review_status", "review_label"} for o in payload)
    assert '/static/vendor/leaflet/leaflet.js' in html and "unpkg.com" not in html
    script = env.client.get("/static/map.js").data.decode()
    assert "L.circleMarker" in script and "L.circle(" not in script
    assert "missionMap.flyTo" in script and "bindPopup" not in script
    assert "marker.setRadius(active ? 12 : 8)" in script
    assert "frame_path" not in html
    assert 'name="_csrf_token"' in html


def test_review_only_changes_status_and_fresh_exports(env):
    with env.app.app_context():
        original = db.session.get(Observation, env.ids[0])
        before = {c.name: getattr(original, c.name) for c in Observation.__table__.columns}
    for decision in ("Confirmed", "Confirmed", "Rejected", "Confirmed"):
        assert review(env, decision).status_code == 303
        with env.app.app_context():
            item = db.session.get(Observation, env.ids[0])
            after = {c.name: getattr(item, c.name) for c in Observation.__table__.columns}
            assert after == {**before, "review_status": decision}
            assert db.session.get(Mission, env.mission).yolov8n_frames == 5
        from pypdf import PdfReader
        from io import BytesIO
        document = PdfReader(BytesIO(env.client.get(f"/export/{env.mission}/pdf").data))
        text_content = " ".join("\n".join(page.extract_text() for page in document.pages).split())
        section = text_content.split("Observation 3 ·")[1]
        assert f"Review status {decision}" in section


@pytest.mark.parametrize("values", [[], ["Unreviewed"], ["confirmed"], ["bogus"], ["Confirmed", "Rejected"],
                                  ["Confirmed", "Confirmed"], [""]])
def test_review_invalid_decisions(env, values):
    data = MultiDict([("_csrf_token", "synthetic-token")] + [("review_status", x) for x in values])
    assert env.client.post(f"/observations/{env.ids[0]}/review", data=data).status_code == 400
    with env.app.app_context():
        assert db.session.get(Observation, env.ids[0]).review_status == "Unreviewed"


def test_review_csrf_forgery_method_and_extra_fields(env):
    endpoint = f"/observations/{env.ids[0]}/review"
    assert env.client.get(endpoint).status_code == 405
    assert env.client.post(endpoint, data={"review_status": "Confirmed"}).status_code == 400
    assert env.client.post(endpoint, data={"_csrf_token": "wrong", "review_status": "Confirmed"}).status_code == 400
    assert review(env, observation=999999).status_code == 404
    assert env.client.post(endpoint, data={"_csrf_token": "synthetic-token",
           "review_status": "Confirmed", "yolov8n_score": "1"}).status_code == 400


def test_review_rollback(env, monkeypatch):
    def fail():
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(db.session, "commit", fail)
    assert review(env).status_code == 500
    with env.app.app_context():
        assert db.session.get(Observation, env.ids[0]).review_status == "Unreviewed"


@pytest.mark.parametrize("path", ["../video.mp4", "/outside.jpg", "C:/outside.jpg", "C:outside.jpg",
    "\\\\server/share/a.jpg", "frames//evidence.jpg", "frames/./evidence.jpg", "frames/../evidence.jpg",
    "frames/a\x00.jpg", "frames/NUL.jpg", "frames/a:stream.jpg", "frames/a?.jpg",
    "frames/evidence.jpg.", "frames/evidence.jpg ", "frames/missing.jpg", "video.mp4"])
def test_evidence_rejects_invalid_stored_paths(env, path):
    with env.app.app_context():
        db.session.execute(text("UPDATE observations SET frame_path=:path WHERE observation_id=:id"),
                           {"path": path, "id": env.ids[0]})
        db.session.commit()
    response = env.client.get(f"/frame/{env.ids[0]}")
    assert response.status_code == 404
    assert str(env.folder).encode() not in response.data
    assert env.client.get(f"/export/{env.mission}/pdf").status_code == 200


def test_evidence_success_ignores_request_path_and_corrupt_file(env):
    response = env.client.get(f"/frame/{env.ids[0]}?path=C:/private.jpg")
    assert response.status_code == 200 and response.mimetype == "image/jpeg"
    (env.folder / "frames/evidence.jpg").write_bytes(b"not an image")
    assert env.client.get(f"/frame/{env.ids[0]}").status_code == 404


@pytest.mark.parametrize("field,value", [
    ("video_filename", "../outside.mp4"),
    ("srt_filename", "uploads/another/telemetry.srt"),
    ("video_filename", "C:/private.mp4"),
])
def test_evidence_rejects_invalid_mission_storage(env, field, value):
    with env.app.app_context():
        db.session.execute(text(f"UPDATE missions SET {field}=:value WHERE mission_id=:id"),
                           {"value": value, "id": env.mission})
        db.session.commit()
    assert env.client.get(f"/frame/{env.ids[0]}").status_code == 404


def test_evidence_resolved_symlink_escape(env, tmp_path, monkeypatch):
    # Simulate OS resolution deterministically even on Windows without symlink privilege.
    original = Path.resolve
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (10, 10)).save(outside)
    def resolve(path, *args, **kwargs):
        if path.name == "evidence.jpg":
            return original(outside, *args, **kwargs)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    assert env.client.get(f"/frame/{env.ids[0]}").status_code == 404


def test_zero_observations(env):
    with env.app.app_context():
        Observation.query.delete()
        db.session.commit()
    response = env.client.get(f"/results/{env.mission}")
    assert response.status_code == 200 and b"No observations" in response.data
    assert env.client.get(f"/export/{env.mission}/pdf").status_code == 200
    assert env.client.get(f"/export/{env.mission}/gpx").status_code == 200


def test_map_includes_rejected_and_excludes_incomplete_coordinates(env):
    with env.app.app_context():
        db.session.get(Observation, env.ids[2]).latitude = -90
        db.session.get(Observation, env.ids[0]).longitude = None
        db.session.commit()
    html = env.client.get(f"/results/{env.mission}").data.decode()
    payload = json.loads(re.search(r'<script id="map-data"[^>]*>(.*?)</script>', html, re.S).group(1))
    assert [o["observation_number"] for o in payload] == [1, 2]
    assert payload[1]["review_status"] == "Rejected"
    assert "Map point unavailable" in html


def test_map_payload_escapes_script_text_and_revalidates_coordinates(env, monkeypatch):
    from daleel import routes
    with env.app.app_context():
        records = [SimpleNamespace(**{c.name: getattr(item, c.name) for c in Observation.__table__.columns})
                   for item in Observation.query.all()]
    records[0].target_class = "</script><script>synthetic</script>"
    records[1].latitude = float("inf")
    records[2].latitude, records[2].longitude = 10, 181
    monkeypatch.setattr(routes, "_observations", lambda mission: records)
    html = env.client.get(f"/results/{env.mission}").data.decode()
    raw = re.search(r'<script id="map-data"[^>]*>(.*?)</script>', html, re.S).group(1)
    assert "<script>" not in raw
    payload = json.loads(raw)
    assert len(payload) == 1 and payload[0]["target_class"] == records[0].target_class


def test_gpx_tracks_review_and_does_not_change_stored_boundary(env):
    from xml.etree import ElementTree as ET
    with env.app.app_context():
        db.session.get(Observation, env.ids[0]).longitude = 180
        db.session.commit()
    for state, present in (("Confirmed", True), ("Rejected", False)):
        assert review(env, state).status_code == 303
        root = ET.fromstring(env.client.get(f"/export/{env.mission}/gpx").data)
        points = {point.find("{http://www.topografix.com/GPX/1/1}name").text: point for point in root}
        assert ("Observation 3" in points) is present
        if present:
            assert points["Observation 3"].attrib["lon"] == "-180"
        with env.app.app_context():
            assert db.session.get(Observation, env.ids[0]).longitude == 180
