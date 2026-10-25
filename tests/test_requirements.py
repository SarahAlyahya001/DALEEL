"""Requirement-based tests used in Chapter 6 of the project report.

Each main functional requirement is treated as one testing unit. The unit
tests (ut) check one function or class on its own; the integration tests
(it) send real requests to the web application and check the database and
the returned pages or files.
Run with:  python -m pytest tests/test_requirements.py -v
"""
import io
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pypdf import PdfReader

from daleel import create_app, initialize_database
from daleel.account_security import (generate_temporary_password,
                                     password_error)
from daleel.exports import build_gpx, build_pdf
from daleel.models import Mission, Observation, User, db
from daleel.pipeline import runner
from daleel.pipeline.detector import Candidate, FrameDetections, fuse
from daleel.pipeline.srt_parser import (SrtValidationError, TelemetryPoint,
                                        coord_at, parse_srt)
from daleel.pipeline.tracker import Tracker
from test_runner import Capture

TOKEN = "test-token"
GOOD_PASSWORD = "River dunes orbit lantern 82!"
VALID_SRT = b"1\n00:00:00,000 --> 00:00:09,000\nGPS(46,24)\n"
NAMES = {"video_name": "desert_search.mp4",
         "srt_name": "desert_search.srt"}


def sample_observation(number, review="Confirmed", target="person"):
    return SimpleNamespace(
        observation_id=number, video_time_ms=1000.0 * number,
        first_frame_index=0, last_frame_index=30, frame_count=3,
        target_class=target, agreement="Both models",
        yolov8n_score=0.80, yolov8s_score=0.70,
        latitude=21.4235, longitude=45.0712, review_status=review,
        frame_path="frames/track-1.jpg")


def pdf_text(data):
    pages = PdfReader(data).pages
    text = "\n".join(page.extract_text() for page in pages)
    return " ".join(text.split())


# ======================================================================
# Unit tests
# ======================================================================

def test_ut1_login_accepts_only_the_correct_password():
    """FR-01, NFR-06: only the correct password is accepted."""
    user = User(username="analyst", role="analyst")
    user.set_password("Correct-Password-2026")
    assert user.check_password("Correct-Password-2026") is True
    assert user.check_password("Wrong-Password") is False
    # only a hash of the password is stored
    assert user.password_hash != "Correct-Password-2026"


def test_ut2_temporary_password_is_long_and_unique():
    """FR-02: each new account gets a long, unique password."""
    first = generate_temporary_password()
    second = generate_temporary_password()
    assert len(first) >= 40
    assert first != second


def test_ut3_invalid_srt_file_is_rejected(tmp_path):
    """FR-04: a file that is not a valid SRT log is refused."""
    srt = tmp_path / "broken.srt"
    srt.write_text("this is not a subtitle file", encoding="utf-8")
    with pytest.raises(SrtValidationError):
        parse_srt(str(srt))


def test_ut4_agreement_label_depends_on_both_models():
    """FR-06: Both models only when both models report the target."""
    yolov8n = [("person", (0, 0, 10, 10), 0.70),
               ("car", (50, 50, 80, 70), 0.60)]
    # YOLOv8s reports the same person and a person YOLOv8n missed
    yolov8s = [("person", (1, 1, 11, 11), 0.85),
               ("person", (200, 200, 210, 220), 0.55)]
    labels = [(c.target_class, c.agreement) for c in fuse(yolov8n, yolov8s)]
    assert labels == [("person", "Both models"), ("person", "One model"),
                      ("car", "One model")]


def test_ut5_observation_gets_nearest_srt_coordinate():
    """FR-07: the nearest telemetry record is used."""
    points = [TelemetryPoint(t=0.0, lat=21.4235, lon=45.0712,
                             alt=None),
              TelemetryPoint(t=1.0, lat=21.4236, lon=45.0713,
                             alt=None)]
    # 0.2 s is closer to 0 s; 0.8 s is closer to 1 s
    assert coord_at(points, 0.2) == (21.4235, 45.0712)
    assert coord_at(points, 0.8) == (21.4236, 45.0713)


def test_ut6_same_person_in_several_frames_is_one_observation():
    """FR-08: repeated detections become one observation."""
    tracker = Tracker()
    person = Candidate("person", (100, 100, 120, 140), 0.70)
    tracker.update(0, 0, [person])
    tracker.update(15, 500, [person])
    tracker.update(30, 1000, [person])
    observations = tracker.finalize()
    assert len(observations) == 1
    assert observations[0].frame_count == 3


def test_ut7_gpx_contains_only_confirmed_observations():
    """FR-14: the GPX file contains Confirmed observations only."""
    mission = SimpleNamespace(mission_id=1)
    observations = [sample_observation(1, "Confirmed"),
                    sample_observation(2, "Rejected"),
                    sample_observation(3, "Unreviewed")]
    gpx = build_gpx(mission, observations).getvalue().decode()
    assert gpx.count("<wpt ") == 1
    assert "Observation 1" in gpx


def test_ut8_pdf_summary_lists_each_observation(tmp_path):
    """FR-15: the PDF summary shows every observation."""
    mission = SimpleNamespace(
        mission_id=7, submitted_at=datetime(2026, 10, 15, 7, 0),
        status="Complete", yolov8n_frames=37, yolov8s_frames=37,
        elapsed_seconds=42.0,
        video_filename="uploads/mission-x/video.mp4",
        srt_filename="uploads/mission-x/telemetry.srt")
    observations = [sample_observation(1, "Confirmed"),
                    sample_observation(2, "Rejected", "car")]
    pdf = build_pdf(mission, observations, tmp_path, tmp_path,
                    names=NAMES)
    text = pdf_text(pdf)
    assert "VIDEO FILE desert_search.mp4" in text
    assert "Observation 1 · Person" in text
    assert "Review status Confirmed" in text
    assert "Observation 2 · Vehicle" in text
    assert "Review status Rejected" in text


def test_ut9_weak_personal_password_is_refused():
    """FR-16: a weak personal password is refused."""
    user = SimpleNamespace(username="analyst",
                           email="analyst@daleel.example",
                           display_name="Demo Analyst")
    assert password_error("short", user) is not None
    assert password_error(GOOD_PASSWORD, user) is None


# ======================================================================
# Integration tests (web pages + database + files)
# ======================================================================

@pytest.fixture
def site(tmp_path):
    """Test copy of DALEEL: one administrator, two analysts and one
    finished mission owned by the analyst Layla."""
    app = create_app(str(tmp_path / "instance"), {"TESTING": True})
    folder = Path(app.config["UPLOAD_DIR"]) / "mission-test"
    (folder / "frames").mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"test video")
    (folder / "telemetry.srt").write_bytes(b"test telemetry")
    (folder / "submission.json").write_text(json.dumps(NAMES))
    Image.new("RGB", (64, 36), "tan").save(folder / "frames/track-1.jpg")
    with app.app_context():
        initialize_database()
        accounts = {}
        for username, role in (("admin", "administrator"),
                               ("layla", "analyst"),
                               ("omar", "analyst")):
            user = User(username=username, role=role,
                        full_name=username.title(),
                        email=f"{username}@daleel.example")
            user.set_password(GOOD_PASSWORD)
            db.session.add(user)
            accounts[username] = user
        db.session.flush()
        mission = Mission(
            analyst_id=accounts["layla"].user_id, status="Complete",
            video_filename="uploads/mission-test/video.mp4",
            srt_filename="uploads/mission-test/telemetry.srt",
            yolov8n_frames=3, yolov8s_frames=3, elapsed_seconds=5.0)
        db.session.add(mission)
        db.session.flush()
        observation = Observation(
            mission_id=mission.mission_id, video_time_ms=500,
            first_frame_index=0, last_frame_index=30, frame_count=3,
            frame_path="frames/track-1.jpg", target_class="person",
            agreement="Both models", yolov8n_score=0.8,
            yolov8s_score=0.7, latitude=21.4235,
            longitude=45.0712)
        db.session.add(observation)
        db.session.commit()
        ids = SimpleNamespace(
            mission=mission.mission_id,
            observation=observation.observation_id,
            **{name: user.user_id for name, user in accounts.items()})
    yield SimpleNamespace(app=app, ids=ids)
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def signed_in(site, username):
    """A browser-like client that is signed in as the given user."""
    client = site.app.test_client()
    with client.session_transaction() as session:
        session["_csrf_token"] = TOKEN
    client.post("/login", data={"_csrf_token": TOKEN,
                                "username": username,
                                "password": GOOD_PASSWORD})
    with client.session_transaction() as session:
        session["_csrf_token"] = TOKEN
    return client


def test_it1_new_analyst_must_change_temporary_password(site):
    """FR-01, FR-02, FR-16: the temporary password is replaced."""
    admin = signed_in(site, "admin")
    page = admin.post("/admin/accounts", data={
        "_csrf_token": TOKEN, "full_name": "Sara Ahmed",
        "email": "sara@daleel.example", "username": "sara",
        "role": "analyst"})
    html = page.get_data(as_text=True)
    marker = 'id="temporary-password" value="'
    temporary = html.split(marker)[1].split('"')[0]
    sara = site.app.test_client()
    with sara.session_transaction() as session:
        session["_csrf_token"] = TOKEN
    first = sara.post("/login", data={
        "_csrf_token": TOKEN, "username": "sara",
        "password": temporary})
    assert first.location.endswith("/change-password")
    # after saving a personal password, Sara continues to New Mission
    with sara.session_transaction() as session:
        session["_csrf_token"] = TOKEN
    saved = sara.post("/change-password", data={
        "_csrf_token": TOKEN, "password": GOOD_PASSWORD,
        "confirm_password": GOOD_PASSWORD})
    assert saved.location.endswith("/upload")
    assert sara.get("/upload").status_code == 200


def test_it2_invalid_upload_is_rejected(site):
    """FR-03, FR-04: an invalid upload creates no mission."""
    analyst = signed_in(site, "layla")
    response = analyst.post("/upload", data={
        "_csrf_token": TOKEN,
        "video": (io.BytesIO(b"not really a video"), "flight.mp4"),
        "srt": (io.BytesIO(b"not a subtitle file"), "flight.srt")})
    assert response.status_code == 400
    page = response.get_data(as_text=True)
    assert "SRT validation failed" in page
    with site.app.app_context():
        # only the prepared mission exists
        assert Mission.query.count() == 1


def test_it3_valid_upload_is_analyzed_to_complete(
        site, tmp_path, monkeypatch):
    """FR-03, FR-05, FR-09: a valid upload becomes Complete."""
    class FakeDetector:  # stands in for the two YOLO models
        def __init__(self, *args):
            pass

        def detect(self, frame, on_call):
            on_call("yolov8n")
            on_call("yolov8s")
            person = Candidate("person", (1, 2, 12, 15), 0.6, 0.8)
            return FrameDetections([person])

    for key in ("WEIGHTS_N", "WEIGHTS_S"):
        (tmp_path / f"{key}.pt").write_bytes(b"test weight")
        site.app.config[key] = str(tmp_path / f"{key}.pt")
    monkeypatch.setattr(runner, "DualDetector", FakeDetector)
    monkeypatch.setattr(runner.cv2, "VideoCapture",
                        lambda: Capture())
    monkeypatch.setattr("daleel.submission.validate_mp4",
                        lambda path: None)
    processor = runner.initialize_processor(site.app)
    try:
        analyst = signed_in(site, "layla")
        response = analyst.post("/upload", data={
            "_csrf_token": TOKEN,
            "video": (io.BytesIO(b"video"), "flight.mp4"),
            "srt": (io.BytesIO(VALID_SRT), "flight.srt")})
        # wait for the analysis to finish
        processor.future.result(timeout=10)
        mission_id = int(response.location.rsplit("/", 1)[1])
        state = analyst.get(f"/api/status/{mission_id}").json
        assert state["status"] == "Complete"
    finally:
        processor.shutdown()


def test_it4_analyst_reviews_an_observation(site):
    """FR-10, FR-12: the analyst confirms an observation."""
    analyst = signed_in(site, "layla")
    page = analyst.get(f"/results/{site.ids.mission}")
    html = page.get_data(as_text=True)
    assert "Observation 1" in html and "Both models" in html
    analyst.post(f"/observations/{site.ids.observation}/review",
                 data={"_csrf_token": TOKEN,
                       "review_status": "Confirmed"})
    with site.app.app_context():
        saved = db.session.get(Observation, site.ids.observation)
        assert saved.review_status == "Confirmed"


def test_it5_analyst_cannot_open_another_analysts_mission(site):
    """FR-13, NFR-05: each analyst sees only their own missions."""
    omar = signed_in(site, "omar")
    page = omar.get(f"/results/{site.ids.mission}")
    assert page.status_code == 404
    history = omar.get("/history").get_data(as_text=True)
    assert "desert_search.mp4" not in history
    layla = signed_in(site, "layla")
    history = layla.get("/history").get_data(as_text=True)
    assert "desert_search.mp4" in history


def test_it6_exports_use_the_reviewed_results(site):
    """FR-14, FR-15: the exports reflect the review decisions."""
    analyst = signed_in(site, "layla")
    analyst.post(f"/observations/{site.ids.observation}/review",
                 data={"_csrf_token": TOKEN,
                       "review_status": "Confirmed"})
    gpx = analyst.get(f"/export/{site.ids.mission}/gpx")
    assert gpx.get_data(as_text=True).count("<wpt ") == 1
    pdf = analyst.get(f"/export/{site.ids.mission}/pdf")
    text = pdf_text(io.BytesIO(pdf.data))
    assert "VIDEO FILE desert_search.mp4" in text
    assert "ANALYST Layla" in text
    assert "Review status Confirmed" in text
    assert "uploads/" not in text  # no storage paths


def test_it7_mission_is_deleted_only_after_confirmation(site):
    """FR-18: a mission is deleted only after confirmation."""
    analyst = signed_in(site, "layla")
    address = f"/missions/{site.ids.mission}/delete"
    refused = analyst.post(address, data={"_csrf_token": TOKEN})
    assert refused.status_code == 400
    analyst.post(address, data={"_csrf_token": TOKEN,
                                "confirm": "delete"})
    with site.app.app_context():
        assert db.session.get(Mission, site.ids.mission) is None


def test_it8_suspended_account_cannot_sign_in(site):
    """FR-17: a suspended account can no longer sign in."""
    admin = signed_in(site, "admin")
    admin.post(f"/admin/accounts/{site.ids.omar}/edit", data={
        "_csrf_token": TOKEN, "full_name": "Omar",
        "email": "omar@daleel.example", "username": "omar",
        "role": "analyst", "account_status": "Suspended"})
    omar = signed_in(site, "omar")
    # the user is sent back to the sign-in page
    assert omar.get("/upload").status_code == 302
