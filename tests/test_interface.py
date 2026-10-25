"""Interface presentation tests and an isolated fixture for rendered inspection."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from PIL import Image, ImageDraw
from flask import redirect, session, Response, request

from daleel import create_app, initialize_database
from daleel.models import Mission, Observation, User, db
from daleel.mission_storage import save_submission_names


def fixture_app(instance):
    app = create_app(str(instance), {"TESTING": True})
    with app.app_context():
        initialize_database()
        users = [
            User(username="example_analyst_" + "a" * 84, role="analyst"),
            User(username="example_admin", role="administrator", full_name="Sample Administrator", email="admin@example.org"),
            User(username="empty_analyst", role="analyst"),
            User(username="invited_employee", full_name="Invited Employee", email="invite@example.org",
                 role="analyst", account_status="Active", password_initialized=True, must_change_password=True),
        ]
        for user in users:
            user.set_password("temporary-fixture-only")
        db.session.add_all(users)
        db.session.flush()
        ids = [user.user_id for user in users]
        for number, state in enumerate(("Complete", "Processing", "Failed", "Complete"), 1):
            folder = Path(app.config["UPLOAD_DIR"]) / f"mission-{number}"
            (folder / "frames").mkdir(parents=True)
            video = "video.mp4"
            save_submission_names(folder, "desert_search_" + "long_filename_" * 10 + ".mp4", "flight.srt")
            (folder / video).write_bytes(b"synthetic input; not decoded")
            (folder / "telemetry.srt").write_bytes(b"synthetic input; not parsed")
            mission = Mission(analyst_id=ids[0], status=state,
                              video_filename=f"uploads/mission-{number}/{video}",
                              srt_filename=f"uploads/mission-{number}/telemetry.srt",
                              yolov8n_frames=17, yolov8s_frames=17,
                              elapsed_seconds=2.75)
            db.session.add(mission)
            db.session.flush()
            if number != 1:
                continue
            picture = Image.new("RGB", (960, 540), "#d8c9aa")
            draw = ImageDraw.Draw(picture)
            draw.rectangle((420, 180, 540, 360), outline="#1b2a4a", width=4)
            draw.text((20, 20), "SYNTHETIC TEST EVIDENCE", fill="#1b2a4a")
            picture.save(folder / "frames/evidence.jpg")
            for index, decision in enumerate(("Unreviewed", "Confirmed", "Rejected")):
                db.session.add(Observation(
                    mission_id=mission.mission_id, video_time_ms=1500.0 + index * 500,
                    first_frame_index=15, last_frame_index=45, frame_count=3,
                    frame_path="frames/evidence.jpg" if index != 2 else "frames/missing.jpg",
                    target_class="person" if index != 1 else "car",
                    agreement="Both models" if index != 2 else "One model",
                    yolov8n_score=0.81, yolov8s_score=0.72 if index != 2 else None,
                    latitude=26.05 + index * .003 if index != 2 else None,
                    longitude=45.05 + index * .003 if index != 2 else None,
                    review_status=decision))
        db.session.commit()
    # Inspection-only session setup, never registered by the production factory.
    @app.get("/_ui/<role>")
    def fixture_role(role):
        session.clear()
        session["user_id"] = ids[{"analyst": 0, "admin": 1, "empty": 2}[role]]
        return redirect("/admin/accounts" if role == "admin" else "/history")
    @app.get("/_ui/reference")
    def visual_reference():
        # Test-only read-only reference, never a production route or template.
        source = Path(__file__).resolve().parents[1] / "tests/data/interface_reference.html"
        return Response(source.read_text(encoding="utf-8"), mimetype="text/html")
    @app.get("/_ui/password-setup")
    def password_setup_fixture():
        session.clear()
        user = User.query.filter_by(username="invited_employee").one()
        session["user_id"] = user.user_id
        session["session_version"] = user.session_version
        return redirect("/change-password")
    @app.after_request
    def reference_policy(response):
        if request.path == "/_ui/reference":
            response.headers["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; connect-src 'none'"
        return response
    app.after_request_funcs[None].insert(0, app.after_request_funcs[None].pop())
    return app


@pytest.fixture
def interface_app(tmp_path):
    app = fixture_app(tmp_path / "instance")
    yield app
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


@pytest.mark.parametrize("role,expected,excluded", [
    ("analyst", b"Mission History", b"Account Management"),
    ("admin", b"Account Management", b"New Mission"),
])
def test_role_navigation(interface_app, role, expected, excluded):
    client = interface_app.test_client()
    response = client.get(f"/_ui/{role}", follow_redirects=True)
    assert expected in response.data and excluded not in response.data
    assert b'aria-current="page"' in response.data
    assert b"Session History" not in response.data
    assert b'class="skip-link"' in response.data


@pytest.mark.parametrize("path", ["/upload", "/history", "/status/2", "/status/3", "/results/1", "/results/4"])
def test_canonical_pages_and_native_forms(interface_app, path):
    client = interface_app.test_client()
    client.get("/_ui/analyst")
    response = client.get(path)
    assert response.status_code == 200
    assert b'id="main-content"' in response.data
    for obsolete in (b"Session History", b"HIGH", b"ALERT", b"startProcessingSimulation",
                     b"confidence_threshold", b"inference_imgsz", b".avi", b"forgot-password"):
        assert obsolete not in response.data
    assert b'src="/static/interface.js" defer' in response.data


def test_password_controls_are_optional_and_do_not_submit(interface_app):
    client = interface_app.test_client()
    pages = [client.get("/login")]
    pages.append(client.get("/_ui/password-setup", follow_redirects=True))
    for response in pages:
        html = response.data.decode()
        assert 'type="button" class="password-toggle"' in html
        assert 'aria-label="Show password" aria-pressed="false" hidden' in html
        assert 'type="password"' in html
        assert 'name="_csrf_token"' in html
        assert 'method="post"' in html
        assert 'readonly' not in html
    script = client.get("/static/interface.js").data.decode()
    password_script = script.split("const historySearch")[0]
    assert "input.value" not in password_script
    assert "localStorage" not in password_script and "sessionStorage" not in script
    assert 'localStorage.setItem(storageKey, String(collapsed))' in script
    assert 'const storageKey = "daleel.sidebar.collapsed"' in script
    assert "fetch(" not in script
    assert 'window.addEventListener("pageshow", reset)' in script


def test_results_fields_review_and_mobile_dom_order(interface_app):
    client = interface_app.test_client()
    client.get("/_ui/analyst")
    html = client.get("/results/1").data.decode()
    assert html.index('class="observation-list"') < html.index('id="map"')
    for label in ("Mission Results", "Unreviewed", "Video timestamp", "Supporting frames",
                  "Show on map", "Evidence map", "Confirmed", "Rejected"):
        assert label in html
    for technical_label in ("Support label", "Screening score", "Verification score",
                            "First source-frame index", "Last source-frame index"):
        assert technical_label not in html
    assert html.count('class="review-actions"') == 3
    assert html.count('aria-pressed="true"') == 3  # Two decisions plus the default All filter.
    assert 'src="/frame/1"' in html and 'href="/frame/1"' not in html
    assert "Open image" not in html and "frame-location-label" not in html
    assert 'name="review_status" value="Unreviewed"' not in html
    assert 'An internet connection is required for map tiles' in html
    assert 'method="post" action="/observations/1/review"' in html


def test_empty_and_status_states(interface_app):
    client = interface_app.test_client()
    client.get("/_ui/empty")
    assert b"No missions" in client.get("/history").data
    client.get("/_ui/analyst")
    assert b"No observations" in client.get("/results/4").data
    assert b'Download confirmed map points' in client.get("/results/4").data
    assert b'data-mission-state="Processing"' in client.get("/status/2").data
    assert b'data-mission-state="Failed"' in client.get("/status/3").data
    assert b'processing-indicator' in client.get("/status/2").data


def test_css_has_motion_and_layout_safeguards(interface_app):
    css = interface_app.test_client().get("/static/style.css").data.decode()
    assert "prefers-reduced-motion: reduce" in css
    assert ":focus-visible" in css
    assert "[hidden]" in css
    assert "overflow-x: auto" in css
    assert "object-fit: contain" in css
    for obsolete in ("tier-high", "tier-alert", "--high:", "--alert:", ".progress-inner"):
        assert obsolete not in css


def test_gp1_visual_tokens_and_real_shell(interface_app):
    client = interface_app.test_client()
    css = client.get("/static/style.css").data.decode()
    for token in ("#1C1A17", "#F5EFE3", "#FBF8F1", "#D9D2C2", "#C9A96E",
                  "--font-mono:", "grid-template-rows: auto 1fr"):
        assert token in css
    client.get("/_ui/analyst")
    html = client.get("/upload").data.decode()
    assert 'class="topbar"' in html and "Operations" in html
    assert 'class="dropzone__icon file-icon"' in html
    assert "data-jump" not in html and "renderSidebar" not in html
    assert "App.history" not in html
    history = client.get("/history").data.decode()
    assert 'class="table mission-table"' in history
    assert 'data-label="Status"' in history
    assert 'class="mission-actions"' in history


def test_reference_components_and_native_evidence_dialog(interface_app):
    client = interface_app.test_client()
    login = client.get("/login").data.decode()
    for component in ('class="login"', 'class="login__brand"', 'class="login__form-wrap"',
                      'class="login__form"', 'class="field__input"'):
        assert component in login
    client.get("/_ui/analyst")
    results = client.get("/results/1").data.decode()
    for component in ('class="results-shell"', 'class="results-list"',
                      'class="results-toolbar"', 'class="detection"',
                      '<dialog class="frame-modal"', 'class="frame-scores"'):
        assert component in results
    for field in ("Both models", "One model", "Model agreement",
                  "YOLOv8n", "YOLOv8s", "First frame", "Last frame", "Unreviewed"):
        assert field in results
    assert results.count('data-open-observation') == 3
    assert 'data-agreement-filter="Both models"' in results
    assert 'data-agreement-filter="One model"' in results
    assert "Dual-Model" not in results and "Screening-Only" not in results
    script = client.get("/static/interface.js").data.decode()
    assert "dialog.showModal()" in script
    assert 'dialog.addEventListener("close"' in script
    assert 'zone.addEventListener("drop"' in script
    assert 'historySearch.addEventListener("input"' in script
    assert "card.dataset.agreement" in script


def test_reference_css_component_rules_are_preserved(interface_app):
    import re
    source = (Path(__file__).resolve().parents[1] /
              "tests/data/interface_reference.html").read_text(encoding="utf-8")
    css = interface_app.test_client().get("/static/style.css").data.decode()
    for selector in (".shell", ".sidebar", ".topbar", ".login", ".login__brand",
                     ".login__form", ".dropzone", ".processing-grid", ".gauge",
                     ".results-shell", ".results-list", ".detection", ".create-form"):
        pattern = re.escape(selector) + r"\s*\{([^}]+)\}"
        reference = re.search(pattern, source).group(1)
        actual = re.search(pattern, css).group(1)
        assert actual.strip() == reference.strip(), selector


if __name__ == "__main__":
    import sys
    import threading
    from werkzeug.serving import make_server
    with TemporaryDirectory(prefix="daleel-stage5-browser-") as root:
        app = fixture_app(Path(root) / "instance")
        server = make_server("127.0.0.1", 0, app)
        def stop_on_input():
            sys.stdin.readline()
            server.shutdown()
        threading.Thread(target=stop_on_input, daemon=True).start()
        print(f"UI_FIXTURE=http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
            with app.app_context():
                db.session.remove()
                db.engine.dispose()


def test_login_copy_is_calm_and_account_help_is_one_statement(interface_app):
    html = interface_app.test_client().get("/login").data.decode()
    assert "Search and rescue" in html
    assert "Supporting the search for missing people through careful review of aerial evidence." in html
    assert html.count('class="login__access"') == 1
    assert "For account access or password help, contact your administrator." in html
    for obsolete in ("Desert SAR", "UAV Detection Platform", "Upload UAV footage and telemetry.",
                     "Accounts are provisioned", "operations supervisor", 'class="login__recovery"'):
        assert obsolete not in html


@pytest.mark.parametrize("mission_id", [1, 4])
def test_delete_action_is_history_only_and_exports_remain(interface_app, mission_id):
    client = interface_app.test_client()
    client.get("/_ui/analyst")
    results = client.get(f"/results/{mission_id}").data.decode()
    assert "Delete mission" not in results
    assert "delete-link" not in results
    assert f"/missions/{mission_id}/delete" not in results
    assert "Export GPX" in results and "Generate PDF" in results
    history = client.get("/history").data.decode()
    assert "delete-link" in history and f'Delete Mission {mission_id}' in history

def test_password_guidance_is_neutral_and_keeps_minimum(interface_app):
    client = interface_app.test_client()
    html = client.get("/_ui/password-setup", follow_redirects=True).data.decode()
    assert "Use at least 15 characters. Avoid common passwords and personal details." in html
    assert html.count('minlength="15"') == 2
    assert "Try a few unrelated words, with spaces." not in html
    assert "Choose a long, unique passphrase unrelated to your account." not in html


@pytest.mark.parametrize("category,message,role", [
    ("success", "Password saved.", "status"),
    ("error", "Invalid username or password.", "alert"),
    ("info", "Please sign in to continue.", "status"),
])
def test_login_feedback_is_inside_form_accessible_and_consumed_once(interface_app, category, message, role):
    client = interface_app.test_client()
    with client.session_transaction() as state:
        state["_flashes"] = [(category, message)]
    html = client.get("/login").data.decode()
    form = html.split('<form class="login__form"', 1)[1].split('</form>', 1)[0]
    assert form.index('id="login-heading"') < form.index(message) < form.index('for="username"')
    assert html.count(message) == 1
    assert f'role="{role}" aria-atomic="true"' in form
    assert 'class="login__notice' in form
    assert 'class="flash ' not in html
    assert message not in client.get("/login").data.decode()


def test_login_feedback_remains_escaped(interface_app):
    client = interface_app.test_client()
    with client.session_transaction() as state:
        state["_flashes"] = [("error", '<script>alert("test")</script>')]
    html = client.get("/login").data.decode()
    assert '<script>alert("test")</script>' not in html
    assert '&lt;script&gt;alert(&#34;test&#34;)&lt;/script&gt;' in html
