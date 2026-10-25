"""Launcher, explicit recovery and verified migration; never starts a server."""
import io
from pathlib import Path
import sqlite3
from unittest.mock import Mock

import pytest
from sqlalchemy import text

import run
from daleel import create_app, initialize_database
from daleel.account_migration import upgrade_accounts, _snapshot
from daleel.account_runtime import (
    bootstrap_administrator, recover_administrator, recovery_available, resolve_instance_path,
)
from daleel.models import User, db
from test_web import make_app, seed_mission
from test_accounts import sign_in, change_password, STRONG


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture(autouse=True)
def isolated_instance_root(tmp_path, monkeypatch):
    """The launcher's archive step must never look at a real instance folder during tests."""
    root = tmp_path / "launcher-instance"
    monkeypatch.setattr(run, "INSTANCE_ROOT", root)
    return root


def credentials(output):
    values = {}
    for line in output.getvalue().splitlines():
        if line.startswith(("Username:", "Temporary password:")):
            key, value = line.split(": ", 1)
            values[key] = value
    if len(values) != 2:
        pytest.fail("Expected one terminal credential display", pytrace=False)
    return values["Username"], values["Temporary password"]


@pytest.fixture
def empty_app(tmp_path):
    app = create_app(str(tmp_path / "instance"), {"TESTING": True})
    with app.app_context():
        initialize_database()
    yield app
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def test_fresh_bootstrap_password_change_and_restart_preservation(empty_app, tmp_path, caplog):
    app = empty_app
    output = Terminal()
    with app.app_context():
        assert bootstrap_administrator(output)
        username, secret = credentials(output)
        user = User.query.one()
        assert user.is_admin and user.must_change_password and user.check_password(secret)
        uid, hashed = user.user_id, user.password_hash
        silent = Terminal()
        assert not bootstrap_administrator(silent)
        assert silent.getvalue() == ""
        assert User.query.one().password_hash == hashed
    client = app.test_client()
    assert sign_in(client, username, secret).location.endswith("/change-password")
    assert change_password(client).location.endswith("/admin/accounts")  # no second sign-in
    assert client.get("/admin/accounts").status_code == 200
    with app.app_context():
        db.session.remove()
        db.engine.dispose()
    restarted = create_app(app.instance_path, {"TESTING": True})
    with restarted.app_context():
        silent = Terminal()
        assert not bootstrap_administrator(silent)
        assert silent.getvalue() == ""
        user = User.query.one()
        assert user.user_id == uid and not user.must_change_password and user.check_password(STRONG)
        db.session.remove()
        db.engine.dispose()
    assert all(secret.encode() not in file.read_bytes() for file in tmp_path.rglob("*") if file.is_file())
    assert secret not in caplog.text


def test_noninteractive_output_refuses_secret_creation(empty_app):
    with empty_app.app_context():
        with pytest.raises(RuntimeError, match="interactive terminal"):
            bootstrap_administrator(io.StringIO())
        db.session.rollback()
        assert User.query.count() == 0


def test_existing_accounts_do_not_bootstrap_even_without_administrator(tmp_path):
    app = make_app(tmp_path)
    seed_mission(app)
    with app.app_context():
        before = [(user.user_id, user.password_hash, user.role) for user in User.query.all()]
        assert not bootstrap_administrator(io.StringIO())
        assert [(user.user_id, user.password_hash, user.role) for user in User.query.all()] == before
        User.query.filter_by(role="administrator").delete()
        db.session.commit()
        assert not bootstrap_administrator(io.StringIO())
        assert User.query.count() == 1
        db.session.remove()
        db.engine.dispose()


def test_recovery_explicit_once_revokes_sessions_preserves_domain(tmp_path, caplog):
    app = make_app(tmp_path)
    seed_mission(app)
    client = app.test_client()
    from test_web import login
    login(client, "admin")
    with app.app_context():
        user = recovery_available()
        uid, previous = user.user_id, user.password_hash
        assert recovery_available().password_hash == previous
        with pytest.raises(RuntimeError, match="interactive terminal"):
            recover_administrator(io.StringIO())
        db.session.rollback()
        output = Terminal()
        recover_administrator(output)
        username, secret = credentials(output)
        user = db.session.get(User, uid)
        assert user.must_change_password and user.session_version == 1
        assert user.check_password(secret) and not user.check_password("correct-password")
        assert not bootstrap_administrator(Terminal())
    assert client.get("/admin/accounts").location.endswith("/login")
    assert sign_in(client, username, secret).location.endswith("/change-password")
    assert change_password(client).location.endswith("/admin/accounts")
    with app.app_context():
        from daleel.models import Mission, Observation
        assert Mission.query.count() == Observation.query.count() == 1
        db.session.remove()
        db.engine.dispose()
    assert all(secret.encode() not in f.read_bytes() for f in tmp_path.rglob("*") if f.is_file())
    assert secret not in caplog.text


def test_recovery_cannot_create_an_administrator_and_requires_disambiguation(empty_app):
    with empty_app.app_context():
        with pytest.raises(RuntimeError, match="No matching"):
            recovery_available()
        for name in ("first", "second"):
            user = User(username=name, role="administrator")
            user.set_password("synthetic-only password")
            db.session.add(user)
        db.session.commit()
        with pytest.raises(RuntimeError, match="More than one"):
            recovery_available()
        assert recovery_available("second").username == "second"


@pytest.mark.parametrize("location", ["app", "review/current", "stage1"])
def test_dual_model_version_never_opens_earlier_storage(tmp_path, location):
    root = tmp_path / "instance"
    existing = root / location
    existing.mkdir(parents=True)
    database = existing / "daleel.db"
    database.write_bytes(b"synthetic sentinel")
    assert resolve_instance_path(root) == root.resolve() / "v2"
    assert database.read_bytes() == b"synthetic sentinel"


def test_launcher_normal_command_bootstraps_once_without_running_server(empty_app, monkeypatch):
    import daleel.pipeline.runner as runner
    monkeypatch.setattr(run, "create_app", lambda: empty_app)
    processor = Mock()
    initialize = Mock(return_value=processor)
    monkeypatch.setattr(runner, "initialize_processor", initialize)
    server = Mock()
    monkeypatch.setattr(run, "serve", server)
    monkeypatch.delenv("DALEEL_HOST", raising=False)
    monkeypatch.delenv("DALEEL_PORT", raising=False)
    first, second = Terminal(), Terminal()
    assert run.main([], first) == 0
    assert run.main([], second) == 0
    assert first.getvalue().count("Temporary password:") == 1
    assert "Temporary password:" not in second.getvalue()
    assert "DALEEL is running at http://127.0.0.1:5000" in second.getvalue()
    assert initialize.call_count == 2 and server.call_count == 2
    assert processor.shutdown.call_count == 2
    assert server.call_args.args == (empty_app, "127.0.0.1", 5000)


def test_launcher_serves_on_the_configured_address(empty_app, monkeypatch):
    import daleel.pipeline.runner as runner
    monkeypatch.setattr(run, "create_app", lambda: empty_app)
    monkeypatch.setattr(runner, "initialize_processor", Mock(return_value=Mock()))
    server = Mock()
    monkeypatch.setattr(run, "serve", server)
    monkeypatch.setenv("DALEEL_HOST", "0.0.0.0")
    monkeypatch.setenv("DALEEL_PORT", "8000")
    assert run.main([], Terminal()) == 0
    assert server.call_args.args == (empty_app, "0.0.0.0", 8000)


@pytest.mark.parametrize("port", ["0", "65536", "http", "-1", ""])
def test_launcher_rejects_invalid_port_before_opening_storage(monkeypatch, capsys, port):
    factory = Mock(side_effect=AssertionError("Storage must not be opened"))
    monkeypatch.setattr(run, "create_app", factory)
    monkeypatch.setenv("DALEEL_PORT", port)
    assert run.main([], Terminal()) == 1
    assert "DALEEL_PORT" in capsys.readouterr().err
    factory.assert_not_called()


def test_serve_uses_the_waitress_production_server(monkeypatch):
    import waitress
    calls = Mock()
    monkeypatch.setattr(waitress, "serve", calls)
    app = object()
    run.serve(app, "127.0.0.1", 5000)
    calls.assert_called_once_with(app, host="127.0.0.1", port=5000)


def test_init_creates_the_first_administrator_once_without_serving(empty_app, monkeypatch):
    import daleel.pipeline.runner as runner
    monkeypatch.setattr(run, "create_app", lambda: empty_app)
    monkeypatch.setattr(runner, "initialize_processor", Mock(side_effect=AssertionError("No processing")))
    monkeypatch.setattr(run, "serve", Mock(side_effect=AssertionError("Server must never start")))
    first, second = Terminal(), Terminal()
    assert run.main(["init"], first) == 0
    assert run.main(["init"], second) == 0
    assert first.getvalue().count("Temporary password:") == 1
    assert "already initialized" in second.getvalue() and "Temporary password:" not in second.getvalue()
    with empty_app.app_context():
        assert User.query.filter_by(role="administrator").count() == 1


def test_init_without_terminal_creates_no_account(empty_app, monkeypatch, capsys):
    monkeypatch.setattr(run, "create_app", lambda: empty_app)
    output = io.StringIO()
    assert run.main(["init"], output) == 1
    assert output.getvalue() == ""
    assert "interactive terminal" in capsys.readouterr().err
    with empty_app.app_context():
        assert User.query.count() == 0


def test_recovery_check_does_not_start_server_or_reset(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr(run, "create_app", lambda: app)
    server = Mock(side_effect=AssertionError("Server must never start"))
    monkeypatch.setattr(run, "serve", server)
    with app.app_context():
        previous = User.query.filter_by(role="administrator").one().password_hash
    output = io.StringIO()
    assert run.main(["recover-admin", "--check"], output) == 0
    assert "No credentials were changed" in output.getvalue()
    with app.app_context():
        assert User.query.filter_by(role="administrator").one().password_hash == previous
        db.session.remove()
        db.engine.dispose()
    server.assert_not_called()


def test_launcher_help_has_no_database_side_effects(monkeypatch):
    factory = Mock(side_effect=AssertionError("Help must not open database"))
    monkeypatch.setattr(run, "create_app", factory)
    with pytest.raises(SystemExit) as result:
        run.main(["--help"])
    assert result.value.code == 0
    factory.assert_not_called()


def pre_migration_database(tmp_path):
    app = make_app(tmp_path)
    seed_mission(app)
    with app.app_context():
        database = Path(db.engine.url.database)
        db.session.remove()
        db.engine.dispose()
    # Only synthetic fixtures: construct a users table without must_change_password.
    with sqlite3.connect(database) as connection:
        sql = connection.execute("SELECT sql FROM sqlite_master WHERE name='users'").fetchone()[0]
        import re
        sql = re.sub(r"\s*must_change_password BOOLEAN[^,]*,", "", sql)
        sql = re.sub(r",?\s*CONSTRAINT ck_users_must_change_password CHECK \(must_change_password IN \(0, 1\)\)", "", sql)
        connection.execute(sql.replace("CREATE TABLE users", "CREATE TABLE users_before"))
        names = [row[1] for row in connection.execute("PRAGMA table_info(users_before)")]
        columns = ", ".join(names)
        connection.execute(f"INSERT INTO users_before ({columns}) SELECT {columns} FROM users")
        connection.execute("DROP TABLE users")
        connection.execute("ALTER TABLE users_before RENAME TO users")
        connection.execute("CREATE TABLE account_tokens (token_hash TEXT PRIMARY KEY, user_id INTEGER)")
        connection.execute("CREATE TABLE account_deliveries (token_hash TEXT PRIMARY KEY, recipient_id INTEGER)")
        connection.execute("INSERT INTO account_tokens VALUES ('synthetic-digest', 2)")
        connection.execute("INSERT INTO account_deliveries VALUES ('synthetic-digest', 2)")
    return app, database


def test_single_column_migration_verified_backup_preserves_all_data(tmp_path):
    app, database = pre_migration_database(tmp_path)
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(users)")]
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        before = {table: _snapshot(connection, table) for table in tables}
    with app.app_context():
        backup = upgrade_accounts()
        assert backup.is_file()
        assert upgrade_accounts() is None
        assert all(not user.must_change_password for user in User.query.all())
        assert all(user.check_password("correct-password") for user in User.query.all())
        db.session.remove()
        db.engine.dispose()
    with sqlite3.connect(database) as connection, sqlite3.connect(backup) as recovery:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        for table in tables:
            assert _snapshot(recovery, table) == before[table]
            assert _snapshot(connection, table, columns if table == "users" else None) == before[table]
        assert set(row[1] for row in connection.execute("PRAGMA table_info(users)")) - set(columns) == {"must_change_password"}


def test_failed_backup_verification_leaves_database_unchanged(tmp_path, monkeypatch):
    import daleel.account_migration as migration
    app, database = pre_migration_database(tmp_path)
    before = database.read_bytes()
    def reject(_database):
        raise RuntimeError("Recovery backup verification failed")
    monkeypatch.setattr(migration, "_verified_backup", reject)
    with app.app_context():
        with pytest.raises(RuntimeError, match="verification failed"):
            upgrade_accounts()
        db.session.remove()
        db.engine.dispose()
    assert database.read_bytes() == before


def test_documented_single_startup_and_no_credential_receipt_writers():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert r".\.venv\Scripts\python.exe run.py" in readme  # one launcher, no activation script needed
    assert "Activate.ps1" not in readme and "Set-ExecutionPolicy" not in readme
    assert "setup_review.ps1" not in readme and "run_review.ps1" not in readme
    runtime = (root / "daleel/account_runtime.py").read_text(encoding="utf-8")
    assert "review_credentials.txt" not in runtime
    assert ".write_text" not in runtime and "open(" not in runtime


def test_migration_rolls_back_if_post_change_data_verification_fails(tmp_path, monkeypatch):
    import daleel.account_migration as migration
    app, database = pre_migration_database(tmp_path)
    before = database.read_bytes()
    original = migration._snapshot
    def mismatch(connection, table, columns=None):
        if table == "users" and columns is not None:
            # Break only post-ALTER verification, not backup verification.
            has_new_column = any(row[1] == "must_change_password"
                                 for row in connection.execute("PRAGMA table_info(users)"))
            if has_new_column:
                return b"synthetic mismatch"
        return original(connection, table, columns)
    monkeypatch.setattr(migration, "_snapshot", mismatch)
    with app.app_context():
        with pytest.raises(RuntimeError, match="differs"):
            upgrade_accounts()
        db.session.remove()
        db.engine.dispose()
    assert database.read_bytes() == before
    assert list((database.parent / "backups").glob("*.sqlite3"))


def test_launcher_missing_column_fails_without_bootstrap_or_server(tmp_path, monkeypatch, capsys):
    app, database = pre_migration_database(tmp_path)
    monkeypatch.setattr(run, "create_app", lambda: app)
    server = Mock(side_effect=AssertionError("Server must never start"))
    monkeypatch.setattr(run, "serve", server)
    output = Terminal()
    assert run.main([], output) == 1
    assert not output.getvalue()
    assert "migrate-accounts" in capsys.readouterr().err
    server.assert_not_called()

def test_recovery_multiple_admins_gives_safe_username_guidance(tmp_path, monkeypatch, capsys):
    app = make_app(tmp_path)
    with app.app_context():
        user = User(username="another_admin", role="administrator")
        user.set_password("synthetic-only password")
        db.session.add(user)
        db.session.commit()
    monkeypatch.setattr(run, "create_app", lambda: app)
    assert run.main(["recover-admin", "--check"], Terminal()) == 1
    assert "--username" in capsys.readouterr().err


def test_launcher_never_logs_unexpected_exception_secret(empty_app, monkeypatch, capsys):
    from daleel.account_security import generate_temporary_password
    secret = generate_temporary_password()
    def failure():
        raise ValueError(secret)
    monkeypatch.setattr(run, "create_app", failure)
    assert run.main([], Terminal()) == 1
    assert secret not in capsys.readouterr().err
