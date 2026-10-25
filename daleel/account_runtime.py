"""Offline provisioning and explicit recovery; no delivery service or modes."""
import secrets
from pathlib import Path

from sqlalchemy import inspect, text

from .account_security import generate_temporary_password
from .models import User, db


class AccountFlowError(RuntimeError):
    """Safe operator guidance containing no account or credential values."""


CURRENT_LAYOUT = "v2"


def resolve_instance_path(instance_root):
    """The dual-model version keeps its own database; earlier layouts are archived, never opened."""
    return Path(instance_root).resolve() / CURRENT_LAYOUT


def require_terminal(output):
    if not getattr(output, "isatty", lambda: False)():
        raise AccountFlowError("Temporary credentials must be displayed in an interactive terminal, not redirected to a file.")


def verify_account_schema():
    inspector = inspect(db.engine)
    if "users" not in inspector.get_table_names():
        raise AccountFlowError("Account database is not initialized.")
    if {column["name"] for column in inspector.get_columns("users")} != {
            column.name for column in User.__table__.columns}:
        raise AccountFlowError("Account migration required. Run: python run.py migrate-accounts")


def _display_credentials(output, user, temporary):
    output.write("Temporary credentials — displayed once. Share privately offline.\n"
                 "Username: " + user.username + "\n"
                 "Temporary password: " + temporary + "\n"
                 "Sign in and choose your own password before using DALEEL.\n")
    output.flush()


def bootstrap_administrator(output):
    """Only an installation with zero users may receive an initial Administrator."""
    verify_account_schema()
    db.session.rollback()
    db.session.execute(text("BEGIN IMMEDIATE"))
    if User.query.count():
        db.session.rollback()
        return False
    require_terminal(output)
    temporary = generate_temporary_password()
    user = User(username="admin_" + secrets.token_hex(6),
                full_name="Initial Administrator", role="administrator",
                account_status="Active", password_initialized=True,
                must_change_password=True)
    user.set_password(temporary)
    db.session.add(user)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    # No plaintext receipt, log entry, environment variable or session storage.
    _display_credentials(output, user, temporary)
    return True


def recovery_available(username=None):
    verify_account_schema()
    query = User.query.filter_by(role="administrator")
    if username:
        query = query.filter_by(username=username)
    users = query.all()
    if not users:
        raise AccountFlowError("No matching Administrator exists. Recovery cannot create an account.")
    if len(users) != 1:
        raise AccountFlowError("More than one Administrator exists. Specify --username to recover one account.")
    return users[0]


def recover_administrator(output, username=None):
    """Operator-only recovery. Never selected during ordinary startup."""
    require_terminal(output)
    db.session.rollback()
    db.session.execute(text("BEGIN IMMEDIATE"))
    user = recovery_available(username)
    temporary = generate_temporary_password()
    user.set_password(temporary)
    user.account_status = "Active"
    user.password_initialized = True
    user.must_change_password = True
    user.session_version += 1
    db.session.commit()
    _display_credentials(output, user, temporary)
