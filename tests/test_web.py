import io
import re
from html.parser import HTMLParser

import pytest

from werkzeug.datastructures import MultiDict
from werkzeug.security import generate_password_hash

from daleel import create_app, initialize_database
from daleel.models import Mission, Observation, User, db


TOKEN_RE = re.compile(rb'name="_csrf_token" value="([^"]+)"')


def csrf_from(response):
    match = TOKEN_RE.search(response.data)
    assert match, "CSRF token not rendered"
    return match.group(1).decode()


def make_app(tmp_path):
    app = create_app(str(tmp_path / "instance"))
    app.config.update(TESTING=True)
    with app.app_context():
        initialize_database()
        db.session.add(
            User(
                username="admin",
                password_hash=generate_password_hash("correct-password"),
                role="administrator",
            )
        )
        db.session.add(User(username="analyst", password_hash=generate_password_hash("correct-password"), role="analyst"))
        db.session.commit()
    return app


def login(client, username="analyst"):
    token = csrf_from(client.get("/login"))
    return client.post(
        "/login",
        data={
            "_csrf_token": token,
            "username": username,
            "password": "correct-password",
        },
    )


def valid_srt_bytes():
    return b"""1
00:00:00,000 --> 00:00:00,999
[latitude: 24.0] [longitude: 46.0] [rel_alt: 60.0]

2
00:00:01,000 --> 00:00:01,999
[latitude: 24.0001] [longitude: 46.0] [rel_alt: 60.0]
"""


def test_secret_is_persistent_and_not_the_historical_default(tmp_path):
    instance = tmp_path / "instance"
    first = create_app(str(instance))
    second = create_app(str(instance))

    assert first.secret_key == second.secret_key
    assert first.secret_key != "dev-key-change-in-production"
    assert (instance / "secret_key").exists()


def test_post_requires_csrf_token(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/login",
        data={"username": "admin", "password": "correct-password"},
    )

    assert response.status_code == 400
    assert b"Invalid or missing CSRF token" in response.data


def test_logout_is_post_only_and_csrf_protected(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()
    login(client)

    assert client.get("/logout").status_code == 405
    assert client.post("/logout").status_code == 400

    page = client.get("/upload")
    response = client.post(
        "/logout",
        data={"_csrf_token": csrf_from(page)},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Sign in" in response.data


def test_security_headers_and_script_nonce(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()

    login(client)
    response = client.get("/upload")
    policy = response.headers["Content-Security-Policy"]

    nonce_match = re.search(r"'nonce-([^']+)'", policy)
    assert nonce_match
    nonce = nonce_match.group(1)
    assert f"'nonce-{nonce}'" in policy
    assert b'src="/static/interface.js" defer' in response.data
    assert "object-src 'none'" in policy
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Permissions-Policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )


def test_login_rate_limit_is_generic_and_scoped(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()
    token = csrf_from(client.get("/login"))

    for _ in range(app.config["LOGIN_MAX_ATTEMPTS"]):
        response = client.post(
            "/login",
            data={
                "_csrf_token": token,
                "username": "admin",
                "password": "wrong-password",
            },
        )
        assert response.status_code == 200
        assert b"Invalid username or password" in response.data

    limited = client.post(
        "/login",
        data={
            "_csrf_token": token,
            "username": "admin",
            "password": "wrong-password",
        },
    )
    assert limited.status_code == 429
    assert b"Too many login attempts" in limited.data
    assert limited.headers["Retry-After"] == "300"


def test_invalid_srt_creates_no_session_or_pending_file(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()
    assert login(client).status_code == 302
    token = csrf_from(client.get("/upload"))

    response = client.post(
        "/upload",
        data={
            "_csrf_token": token,
            "confidence_threshold": "0.5",
            "iou_threshold": "0.5",
            "inference_imgsz": "1280",
            "video": (io.BytesIO(b"not reached"), "mission.mp4"),
            "srt": (io.BytesIO(b"not telemetry"), "mission.srt"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 400
    assert b"SRT validation failed" in response.data
    with app.app_context():
        assert Mission.query.count() == 0
    assert not list((tmp_path / "instance" / "uploads").iterdir())


def test_corrupt_video_creates_no_session_or_pending_file(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()
    login(client)
    token = csrf_from(client.get("/upload"))

    response = client.post(
        "/upload",
        data={
            "_csrf_token": token,
            "confidence_threshold": "0.5",
            "iou_threshold": "0.5",
            "inference_imgsz": "1280",
            "video": (io.BytesIO(b"not a video"), "mission.mp4"),
            "srt": (io.BytesIO(valid_srt_bytes()), "mission.srt"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 400
    assert b"MP4 validation failed" in response.data
    with app.app_context():
        assert Mission.query.count() == 0
    assert not list((tmp_path / "instance" / "uploads").iterdir())


def test_analyst_cannot_access_another_users_mission(tmp_path):
    app = make_app(tmp_path)
    with app.app_context():
        other = User(
            username="other",
            password_hash=generate_password_hash("other-password"),
            role="analyst",
        )
        db.session.add(other)
        db.session.flush()
        mission = Mission(
            analyst_id=other.user_id,
            video_filename="private.mp4",
            srt_filename="private.srt",
            status="Complete",
        )
        db.session.add(mission)
        db.session.commit()
        mission_id = mission.mission_id

    client = app.test_client()
    login(client)

    assert client.get(f"/results/{mission_id}").status_code == 404

class FormAccessibilityParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.labels = set()
        self.controls = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "label" and attributes.get("for"):
            self.labels.add(attributes["for"])
        if tag in {"input", "select"} and attributes.get("type") != "hidden":
            if attributes.get("id"):
                self.controls.add(attributes["id"])


def assert_controls_have_labels(response):
    parser = FormAccessibilityParser()
    parser.feed(response.data.decode("utf-8"))
    assert parser.controls
    assert parser.controls <= parser.labels


def test_forms_have_explicit_labels_skip_link_and_keyboard_targets(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()

    login_page = client.get("/login")
    assert_controls_have_labels(login_page)
    assert b'class="skip-link"' in login_page.data

    login(client)
    upload_page = client.get("/upload")
    assert_controls_have_labels(upload_page)
    assert b'id="submit-mission"' in upload_page.data
    assert b'aria-describedby="video-hint"' in upload_page.data
    for obsolete in (b".avi", b'name="confidence_threshold"', b'name="iou_threshold"', b'name="inference_imgsz"'):
        assert obsolete not in upload_page.data


def seed_mission(app):
    with app.app_context():
        owner = User.query.filter_by(username="analyst").one()
        mission = Mission(analyst=owner, video_filename="mission.mp4", srt_filename="mission.srt")
        item = Observation(
            mission=mission, video_time_ms=0, first_frame_index=0, last_frame_index=0,
            frame_count=1, frame_path="frames/evidence.jpg", target_class="person",
            agreement="One model", yolov8n_score=0.3,
        )
        db.session.add(item)
        db.session.commit()
        return mission.mission_id, item.observation_id


@pytest.mark.parametrize("endpoint", [
    "/results/{mission_id}",
    "/export/{mission_id}/gpx", "/export/{mission_id}/pdf", "/frame/{observation_id}",
])
def test_incomplete_results_are_unavailable_without_mutation(tmp_path, endpoint):
    app = make_app(tmp_path)
    mission_id, observation_id = seed_mission(app)
    client = app.test_client()
    login(client)
    response = client.get(endpoint.format(mission_id=mission_id, observation_id=observation_id))
    assert response.status_code == (404 if endpoint.startswith("/frame/") else 302)
    assert "Content-Disposition" not in response.headers
    with app.app_context():
        assert db.session.get(Mission, mission_id).status == "Processing"
        assert Mission.query.count() == 1
        assert Observation.query.count() == 1
        assert db.session.get(Observation, observation_id).review_status == "Unreviewed"


@pytest.mark.parametrize("endpoint", [
    "/status/{mission_id}", "/api/status/{mission_id}", "/results/{mission_id}",
    "/export/{mission_id}/gpx", "/export/{mission_id}/pdf", "/frame/{observation_id}",
])
def test_guards_preserve_authentication_and_ownership(tmp_path, endpoint):
    app = make_app(tmp_path)
    mission_id, observation_id = seed_mission(app)
    url = endpoint.format(mission_id=mission_id, observation_id=observation_id)
    client = app.test_client()
    assert client.get(url).status_code == 302
    with app.app_context():
        other = User(username="other", role="analyst")
        other.set_password("other-password")
        db.session.add(other)
        db.session.commit()
        other_id = other.user_id
    with client.session_transaction() as session:
        session["user_id"] = other_id
    assert client.get(url).status_code == 404


def test_valid_submission_guard_creates_no_records_or_leftover_media(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    client = app.test_client()
    login(client)
    monkeypatch.setattr("daleel.submission.validate_mp4", lambda path: None)
    response = client.post("/upload", data={
        "_csrf_token": csrf_from(client.get("/upload")),
        "video": (io.BytesIO(b"synthetic validated video"), "mission.mp4"),
        "srt": (io.BytesIO(valid_srt_bytes()), "mission.srt"),
    }, content_type="multipart/form-data")
    assert response.status_code == 503
    assert b"No mission was created" in response.data
    with app.app_context():
        assert Mission.query.count() == Observation.query.count() == 0
    assert not list((tmp_path / "instance" / "uploads").iterdir())
    assert not list((tmp_path / "instance" / "artifacts").iterdir())


def test_history_reads_canonical_fields_and_is_owner_scoped(tmp_path):
    app = make_app(tmp_path)
    seed_mission(app)
    with app.app_context():
        other = User(username="other", role="analyst")
        other.set_password("synthetic-password")
        db.session.add(Mission(analyst=other, video_filename="other-private.mp4", srt_filename="other.srt"))
        db.session.commit()
    client = app.test_client()
    login(client)
    response = client.get("/history")
    assert response.status_code == 200
    assert b"Mission history" in response.data
    assert b"Processing" in response.data
    assert b"Observations" in response.data
    assert b"other-private.mp4" not in response.data


def test_account_form_collects_identity_and_no_employee_password(tmp_path):
    client = make_app(tmp_path).test_client()
    login(client, "admin")
    page = client.get("/admin/accounts")
    assert_controls_have_labels(page)
    for field in (b'name="full_name"', b'name="email"', b'name="username"', b'name="role"'):
        assert field in page.data
    assert b'Administrator only' not in page.data
    assert b'name="password"' not in page.data
    assert b'Created (UTC)' not in page.data


@pytest.mark.parametrize("username,destination", [("analyst", "/upload"), ("admin", "/admin/accounts")])
def test_role_specific_login_and_landing(tmp_path, username, destination):
    client = make_app(tmp_path).test_client()
    assert login(client, username).location.endswith(destination)
    assert client.get("/login").location.endswith(destination)
    assert client.get("/").location.endswith(destination)


@pytest.mark.parametrize("username,password", [("unknown", "correct-password"), ("analyst", "wrong"), ("", "")])
def test_login_failure_is_generic_and_creates_no_session(tmp_path, username, password):
    client = make_app(tmp_path).test_client()
    response = client.post("/login", data={"_csrf_token": csrf_from(client.get("/login")),
                                          "username": username, "password": password})
    assert b"Invalid username or password" in response.data
    with client.session_transaction() as session:
        assert "user_id" not in session


@pytest.mark.parametrize("user_id", [999999, "1", True, None])
def test_stale_or_invalid_session_cannot_access_pages(tmp_path, user_id):
    client = make_app(tmp_path).test_client()
    with client.session_transaction() as session:
        session["user_id"] = user_id
    assert client.get("/upload").status_code == 302
    assert client.get("/admin/accounts").status_code == 302


def test_account_management_requires_administrator_and_csrf(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()
    token = csrf_from(client.get("/login"))
    assert client.get("/admin/accounts").status_code == 302
    assert client.post("/admin/accounts", data={"_csrf_token": token}).status_code == 302
    login(client)
    token = csrf_from(client.get("/upload"))
    assert client.get("/admin/accounts").status_code == 403
    assert client.post("/admin/accounts", data={"_csrf_token": token}).status_code == 403
    with app.app_context():
        assert User.query.count() == 2
    admin = app.test_client()
    login(admin, "admin")
    assert admin.post("/admin/accounts", data={"username": "new", "password": "new-password"}).status_code == 400
    assert admin.get("/register").status_code == 404


@pytest.mark.parametrize("roles", [["admin"], ["unknown"], ["analyst", "analyst"], ["analyst", "administrator"]])
def test_account_role_cannot_be_forged(tmp_path, roles):
    app = make_app(tmp_path)
    client = app.test_client()
    login(client, "admin")
    data = MultiDict([("_csrf_token", csrf_from(client.get("/admin/accounts"))),
                      ("username", "new"), ("full_name", "New Employee"),
                      ("email", "employee@example.org")])
    for role in roles:
        data.add("role", role)
    response = client.post("/admin/accounts", data=data)
    assert response.status_code == 400
    with app.app_context():
        assert User.query.count() == 2


@pytest.mark.parametrize("endpoint", ["/upload", "/history", "/status/{mission_id}",
    "/api/status/{mission_id}", "/results/{mission_id}", "/export/{mission_id}/gpx",
    "/export/{mission_id}/pdf", "/frame/{observation_id}"])
def test_administrator_has_no_mission_access_override(tmp_path, endpoint):
    app = make_app(tmp_path)
    mission_id, observation_id = seed_mission(app)
    client = app.test_client()
    login(client, "admin")
    response = client.get(endpoint.format(mission_id=mission_id, observation_id=observation_id))
    assert response.status_code == 403
    response = client.post("/upload", data={"_csrf_token": csrf_from(client.get("/admin/accounts"))})
    assert response.status_code == 403
    with app.app_context():
        assert Mission.query.count() == Observation.query.count() == 1
    assert not list((tmp_path / "instance" / "uploads").iterdir())


def assert_no_submission_side_effects(app, tmp_path):
    with app.app_context():
        assert Mission.query.count() == Observation.query.count() == 0
    assert not list((tmp_path / "instance" / "uploads").iterdir())
    assert not list((tmp_path / "instance" / "artifacts").iterdir())


@pytest.mark.parametrize("case,message", [
    ("missing_video", b"MP4 video file is required"), ("missing_srt", b"SRT telemetry file is required"),
    ("duplicate_video", b"Exactly one MP4"), ("duplicate_srt", b"Exactly one SRT"),
    ("extra", b"Unexpected file field"), ("avi", b"only .mp4"), ("txt", b"only .srt"),
    ("empty", b"file is empty"), ("coordinates", b"out-of-range coordinate"),
])
def test_invalid_multipart_submission_has_specific_error_and_no_side_effects(tmp_path, case, message):
    app = make_app(tmp_path)
    client = app.test_client()
    login(client)
    data = MultiDict([("_csrf_token", csrf_from(client.get("/upload")))])
    if case != "missing_video":
        data.add("video", (io.BytesIO(b"not decoded"), "flight.avi" if case == "avi" else "flight.mp4"))
    if case != "missing_srt":
        content = b"" if case == "empty" else valid_srt_bytes()
        if case == "coordinates":
            content = content.replace(b"24.0]", b"91.0]")
        data.add("srt", (io.BytesIO(content), "data.txt" if case == "txt" else "different.srt"))
    if case.startswith("duplicate_"):
        field = case.removeprefix("duplicate_")
        data.add(field, (io.BytesIO(b"duplicate"), "extra.mp4" if field == "video" else "extra.srt"))
    if case == "extra":
        data.add("other", (io.BytesIO(b"extra"), "extra.srt"))
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert message in response.data
    assert b"No mission was created; temporary files were removed." not in response.data
    assert_no_submission_side_effects(app, tmp_path)


def test_submission_limits_and_storage_errors_are_not_processing_unavailability(tmp_path, monkeypatch):
    from daleel.submission import SubmissionStorageError

    app = make_app(tmp_path)
    client = app.test_client()
    login(client)
    token = csrf_from(client.get("/upload"))
    def submit():
        return client.post("/upload", data={"_csrf_token": token,
            "video": (io.BytesIO(b"oversized"), "video.mp4"),
            "srt": (io.BytesIO(valid_srt_bytes()), "data.srt")}, content_type="multipart/form-data")
    app.config["VIDEO_MAX_BYTES"] = 4
    response = submit()
    assert response.status_code == 400
    assert b"MP4 video exceeds" in response.data
    assert_no_submission_side_effects(app, tmp_path)
    app.config["MAX_CONTENT_LENGTH"] = 100
    assert submit().status_code == 413
    assert_no_submission_side_effects(app, tmp_path)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    def unavailable_storage(*args, **kwargs):
        raise SubmissionStorageError("Temporary upload storage or cleanup failed; no mission was created.")
    monkeypatch.setattr("daleel.routes.validated_submission", unavailable_storage)
    response = submit()
    assert response.status_code == 500
    assert b"storage or cleanup failed" in response.data
    assert_no_submission_side_effects(app, tmp_path)


def test_real_mp4_and_bom_srt_return_503_without_processing_or_association(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    def forbidden(*args, **kwargs):
        pytest.fail("Stage 2 must not process, start workers, or associate coordinates")
    monkeypatch.setattr("daleel.pipeline.srt_parser.coord_at", forbidden)
    monkeypatch.setattr("daleel.pipeline.runner.start_processing", forbidden)
    monkeypatch.setattr("threading.Thread.start", forbidden)
    app = make_app(tmp_path)
    client = app.test_client()
    login(client)
    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
    try:
        assert writer.isOpened(), "Existing OpenCV must support the test MP4 codec"
        for _ in range(3):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    original_limit = app.config["VIDEO_MAX_BYTES"]
    response = client.post("/upload", data={"_csrf_token": csrf_from(client.get("/upload")),
        "video": (io.BytesIO(video.read_bytes()), "flight.MP4"),
        "srt": (io.BytesIO(b"\xef\xbb\xbf" + valid_srt_bytes()), "unrelated.SRT"),
        "confidence_threshold": "invalid", "iou_threshold": "invalid", "inference_imgsz": "invalid",
    }, content_type="multipart/form-data")
    assert response.status_code == 503
    assert b"Processing is not initialized. No mission was created" in response.data
    assert app.config["VIDEO_MAX_BYTES"] == original_limit
    assert_no_submission_side_effects(app, tmp_path)
