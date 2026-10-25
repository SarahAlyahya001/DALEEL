"""Offline account flow. All credentials are synthetic and remain in memory."""
from html import unescape
import re

import pytest
from sqlalchemy.exc import IntegrityError

from daleel.account_security import generate_temporary_password, password_error
from daleel.models import Mission, Observation, User, db
from test_web import make_app, login, csrf_from, seed_mission

STRONG = "River dunes orbit lantern 82!"


@pytest.fixture
def env(tmp_path):
    app = make_app(tmp_path)
    with app.app_context():
        for username, name in (("admin", "Initial Administrator"), ("analyst", "Existing Analyst")):
            user = User.query.filter_by(username=username).one()
            user.full_name = name
            user.email = username + "@example.org"
        db.session.commit()
    admin = app.test_client()
    login(admin, "admin")
    yield app, admin
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def create(client, **changes):
    data = dict(username="employee", full_name="New Employee",
                email="employee@example.org", role="analyst")
    data.update(changes)
    data["_csrf_token"] = csrf_from(client.get("/admin/accounts"))
    return client.post("/admin/accounts", data=data)


def temporary(response):
    match = re.search(rb'id="temporary-password" value="([^"]+)"', response.data)
    if not match:
        pytest.fail("One-time credential response missing", pytrace=False)
    return unescape(match.group(1).decode())


def sign_in(client, username, password):
    return client.post("/login", data={"_csrf_token": csrf_from(client.get("/login")),
                                      "username": username, "password": password})


def change_password(client, password=STRONG, confirm=None):
    return client.post("/change-password", data={
        "_csrf_token": csrf_from(client.get("/change-password")),
        "password": password, "confirm_password": password if confirm is None else confirm})


def reset(client, user_id):
    return client.post(f"/admin/accounts/{user_id}/reset-password", data={
        "_csrf_token": csrf_from(client.get("/admin/accounts"))})


def edit_data(app, username="analyst"):
    with app.app_context():
        user = User.query.filter_by(username=username).one()
        return user.user_id, dict(username=user.username, full_name=user.display_name,
                                 email=user.email, role=user.role, account_status=user.account_status)


def edit(client, user_id, data):
    return client.post(f"/admin/accounts/{user_id}/edit", data=dict(
        data, _csrf_token=csrf_from(client.get(f"/admin/accounts/{user_id}/edit"))))


@pytest.mark.parametrize("role", ["analyst", "administrator"])
def test_create_first_sign_in_and_normal_role_access(env, role):
    app, admin = env
    response = create(admin, role=role)
    assert response.status_code == 200
    secret = temporary(response)
    with app.app_context():
        user = User.query.filter_by(username="employee").one()
        assert user.role == role and user.must_change_password
        assert user.account_status == "Active" and user.password_initialized
        assert user.check_password(secret)
        assert user.password_hash != secret
    employee = app.test_client()
    assert sign_in(employee, "employee", secret).location.endswith("/change-password")
    landing = "/admin/accounts" if role == "administrator" else "/upload"
    # Saving the personal password continues to the role's page without a second sign-in.
    assert change_password(employee).location.endswith(landing)
    assert employee.get("/admin/accounts").status_code == (200 if role == "administrator" else 403)
    assert employee.get("/upload").status_code == (403 if role == "administrator" else 200)
    employee.post("/logout", data={"_csrf_token": csrf_from(employee.get(landing))})
    assert employee.get(landing).location.endswith("/login")
    assert b"Invalid username or password" in sign_in(employee, "employee", secret).data
    assert sign_in(employee, "employee", STRONG).location.endswith(landing)
    with app.app_context():
        user = User.query.filter_by(username="employee").one()
        assert not user.must_change_password and user.check_password(STRONG)
    if role == "administrator":
        page = employee.get("/admin/accounts")
        assert b"Initial Administrator" in page.data and b"Existing Analyst" in page.data


@pytest.mark.parametrize("endpoint", [
    "/", "/login", "/upload", "/history", "/results/1", "/mission/1/status",
    "/export/1/pdf", "/export/1/gpx", "/frame/1", "/admin/accounts",
    "/admin/accounts/1/edit", "/forgot-password", "/missing-page",
])
def test_temporary_user_cannot_access_other_routes(env, endpoint):
    app, admin = env
    secret = temporary(create(admin))
    client = app.test_client()
    sign_in(client, "employee", secret)
    result = client.get(endpoint)
    assert result.status_code == 302 and result.location.endswith("/change-password")
    page = client.get("/change-password")
    assert page.status_code == 200
    assert b"primary-sidebar" not in page.data and b"Account management" not in page.data
    assert client.get("/static/style.css").status_code == 200


@pytest.mark.parametrize("endpoint", ["/upload", "/admin/accounts", "/admin/accounts/1/reset-password", "/missing"])
def test_post_route_restriction_and_csrf(env, endpoint):
    app, admin = env
    client = app.test_client()
    sign_in(client, "employee", temporary(create(admin)))
    assert client.post(endpoint).status_code == 400
    result = client.post(endpoint, data={"_csrf_token": csrf_from(client.get("/change-password"))})
    assert result.status_code == 302 and result.location.endswith("/change-password")


@pytest.mark.parametrize("password,confirm", [
    ("short1!", "short1!"), (STRONG, "different confirmation"),
    ("password123456789", "password123456789"),
    ("employeeemployee123", "employeeemployee123"),
    ("invisible\u200b strong password", "invisible\u200b strong password"),
])
def test_invalid_new_password_keeps_restriction(env, password, confirm):
    app, admin = env
    client = app.test_client()
    secret = temporary(create(admin))
    sign_in(client, "employee", secret)
    assert change_password(client, password, confirm).status_code == 400
    assert client.get("/history").location.endswith("/change-password")
    with app.app_context():
        user = User.query.filter_by(username="employee").one()
        assert user.must_change_password and user.check_password(secret)


def test_temporary_password_cannot_be_reused_as_final(env):
    app, admin = env
    secret = temporary(create(admin))
    client = app.test_client()
    sign_in(client, "employee", secret)
    assert change_password(client, secret).status_code == 400
    assert b"different from your temporary password" in client.get("/change-password").data or client.get("/history").status_code == 302


def test_change_only_once_and_restricted_logout(env):
    app, admin = env
    secret = temporary(create(admin))
    first, second = app.test_client(), app.test_client()
    sign_in(first, "employee", secret)
    sign_in(second, "employee", secret)
    token = csrf_from(second.get("/change-password"))
    assert change_password(first).status_code == 302
    replay = second.post("/change-password", data={"_csrf_token": token,
                         "password": "Another unique desert passphrase", "confirm_password": "Another unique desert passphrase"})
    assert replay.location.endswith("/login")
    assert second.get("/upload").location.endswith("/login")  # the other session was revoked
    assert first.get("/upload").status_code == 200  # the session that changed the password continues
    with app.app_context():
        assert User.query.filter_by(username="employee").one().check_password(STRONG)
    assert first.get("/login").location.endswith("/upload")  # still signed in, no second sign-in
    assert first.get("/change-password").location.endswith("/upload")
    restricted = app.test_client()
    with app.app_context():
        uid = User.query.filter_by(username="employee").one().user_id
    new_secret = temporary(reset(admin, uid))
    sign_in(restricted, "employee", new_secret)
    assert restricted.post("/logout", data={"_csrf_token": csrf_from(restricted.get("/change-password"))}).location.endswith("/login")


def test_one_time_display_no_plaintext_database_files_or_logs(env, tmp_path, caplog):
    app, admin = env
    response = create(admin)
    secret = temporary(response)
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert b"data-copy-credentials" in response.data
    with app.app_context():
        user = User.query.filter_by(username="employee").one()
        uid = user.user_id
    for url in ("/admin/accounts", f"/admin/accounts/{uid}/edit", "/login"):
        assert secret.encode() not in admin.get(url).data
    with admin.session_transaction() as cookie:
        assert secret not in repr(dict(cookie))
    # Generated files, SQLite pages and captured logs must not contain it.
    with app.app_context():
        db.session.remove()
        db.engine.dispose()
    assert all(secret.encode() not in file.read_bytes() for file in tmp_path.rglob("*") if file.is_file())
    assert secret not in caplog.text
    assert admin.get(f"/admin/accounts/{uid}/reset-password").status_code == 405


def test_admin_reset_revokes_final_password_old_sessions_and_previous_temporary(env):
    app, admin = env
    client = app.test_client()
    login(client)
    uid, _ = edit_data(app)
    response = reset(admin, uid)
    first = temporary(response)
    assert client.get("/history").location.endswith("/login")
    assert b"Invalid username or password" in sign_in(client, "analyst", "correct-password").data
    assert sign_in(client, "analyst", first).location.endswith("/change-password")
    second = temporary(reset(admin, uid))
    assert second != first
    assert client.get("/change-password").location.endswith("/login")
    assert b"Invalid username or password" in sign_in(client, "analyst", first).data
    assert sign_in(client, "analyst", second).location.endswith("/change-password")
    assert change_password(client).location.endswith("/upload")


def test_self_reset_discloses_once_and_signs_out(env):
    app, admin = env
    uid, _ = edit_data(app, "admin")
    response = reset(admin, uid)
    secret = temporary(response)
    assert b"Sign in" in response.data
    assert admin.get("/admin/accounts").location.endswith("/login")
    assert sign_in(admin, "admin", secret).location.endswith("/change-password")


def test_reset_keeps_suspension_and_explicitly_recovers_legacy_invited_account(env):
    app, admin = env
    uid, fields = edit_data(app)
    fields["account_status"] = "Suspended"
    assert edit(admin, uid, fields).status_code == 302
    secret = temporary(reset(admin, uid))
    client = app.test_client()
    assert b"Invalid username or password" in sign_in(client, "analyst", secret).data
    with app.app_context():
        user = User(username="pending", full_name="Pending Employee", email="pending@example.org",
                    role="analyst", account_status="Invited", password_initialized=False, password_hash=None)
        db.session.add(user)
        db.session.commit()
        pending_id = user.user_id
    assert b"Invalid username or password" in sign_in(client, "pending", "unused-password").data
    secret = temporary(reset(admin, pending_id))
    assert sign_in(client, "pending", secret).location.endswith("/change-password")


@pytest.mark.parametrize("field,value", [
    ("full_name", ""), ("full_name", "a"*151), ("username", ""), ("username", "a"*101),
    ("email", "not-email"), ("email", "Name <a@example.org>"),
    ("email", "a@example.org\nBcc: x@example.org"), ("role", "owner"),
    ("username", "analyst"), ("email", "ANALYST@example.org"),
])
def test_invalid_or_duplicate_identity_does_not_create(env, field, value):
    app, admin = env
    assert create(admin, **{field: value}).status_code == 400
    with app.app_context():
        assert User.query.count() == 2


@pytest.mark.parametrize("change", [{"role": "analyst"}, {"account_status": "Suspended"}])
def test_last_active_administrator_protection(env, change):
    app, admin = env
    uid, data = edit_data(app, "admin")
    data.update(change)
    result = edit(admin, uid, data)
    assert result.status_code == 400 and b"last active administrator" in result.data
    with app.app_context():
        assert User.query.filter_by(username="admin").one().is_admin


def test_account_forms_have_no_password_confirmation(env):
    app, admin = env
    uid, _ = edit_data(app, "analyst")
    for path in ("/admin/accounts", f"/admin/accounts/{uid}/edit"):
        page = admin.get(path)
        assert b'name="admin_password"' not in page.data
        assert b'type="password"' not in page.data
        for dead in (b"Development email preview", b"Generate reset link", b"Delivery status", b"Create and invite"):
            assert dead not in page.data


def test_edit_preserves_credentials_missions_observations(env):
    app, admin = env
    seed_mission(app)
    uid, fields = edit_data(app)
    fields["full_name"] = "Updated Analyst"
    assert edit(admin, uid, fields).status_code == 302
    with app.app_context():
        user = User.query.filter_by(username="analyst").one()
        assert user.check_password("correct-password") and not user.must_change_password
        assert Mission.query.count() == Observation.query.count() == 1
        assert Mission.query.one().analyst_id == uid


def test_analysts_and_anonymous_users_cannot_manage_accounts(env):
    app, admin = env
    analyst = app.test_client()
    login(analyst)
    for path in ("/admin/accounts", "/admin/accounts/1/reset-password"):
        token = csrf_from(analyst.get("/upload"))
        assert analyst.post(path, data={"_csrf_token": token}).status_code == 403
    anonymous = app.test_client()
    assert anonymous.get("/admin/accounts").location.endswith("/login")


@pytest.mark.parametrize("path", ["/forgot-password", "/set-password/unused", "/admin/accounts/1/send-link",
                                  "/admin/accounts/1/deliveries/unused/preview"])
def test_retired_routes_are_unavailable(env, path):
    app, admin = env
    assert admin.get(path).status_code == 404
    assert admin.post(path, data={"_csrf_token": csrf_from(admin.get("/admin/accounts"))}).status_code == 404


def test_temporary_password_entropy_and_account_modes_are_absent(env):
    app, admin = env
    values = {generate_temporary_password() for _ in range(100)}
    assert len(values) == 100 and all(len(value) >= 43 for value in values)
    assert "daleel_mail" not in app.extensions
    assert not any(key.startswith("SMTP_") or key == "APP_MODE" for key in app.config)

def test_concurrent_administrators_cannot_remove_last_active_admin(env):
    from concurrent.futures import ThreadPoolExecutor
    app, first = env
    secret = temporary(create(first, role="administrator"))
    second = app.test_client()
    sign_in(second, "employee", secret)
    change_password(second)  # continues signed in
    first_id, first_values = edit_data(app, "admin")
    second_id, second_values = edit_data(app, "employee")
    first_values["role"] = second_values["role"] = "analyst"
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda task: edit(*task).status_code,
                       [(first, first_id, first_values), (second, second_id, second_values)]))
    assert sorted(results) == [302, 400]
    with app.app_context():
        assert User.query.filter_by(role="administrator", account_status="Active").count() == 1


def test_commit_failure_does_not_reveal_temporary_password(env, monkeypatch, caplog):
    app, admin = env
    def reject_commit():
        raise IntegrityError("synthetic conflict", None, Exception("safe synthetic conflict"))
    with app.app_context():
        session_type = type(db.session())
    monkeypatch.setattr(session_type, "commit", lambda self: reject_commit())
    response = create(admin)
    assert response.status_code == 409 and b"temporary-password" not in response.data
    assert "Temporary password:" not in caplog.text
    with app.app_context():
        assert User.query.count() == 2


def test_database_errors_hide_sensitive_parameters(env):
    app, admin = env
    with app.app_context():
        assert db.engine.hide_parameters is True

@pytest.mark.parametrize("password,accepted", [
    ("cobalt ivy mos", False),
    ("cobalt ivy moss", True),
    ("cobalt meadow lantern river", True),
], ids=["below-minimum", "exact-minimum", "long-passphrase"])
def test_passphrase_length_without_number_or_symbol_requirements(password, accepted):
    user = User(username="sample_operator", full_name="Sample Operator", role="analyst")
    assert (password_error(password, user) is None) is accepted
