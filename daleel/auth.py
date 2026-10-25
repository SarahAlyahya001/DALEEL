from functools import wraps
import time
import unicodedata
from sqlalchemy import text

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash

from .models import User, db
from .account_security import password_error

bp = Blueprint("auth", __name__)


def _attempt_keys(username):
    return (
        ("address", request.remote_addr or "unknown"),
        ("account", username.casefold()),
    )


def _failed_attempts(keys, add=False):
    limiter = current_app.extensions["daleel_login_limiter"]
    now = time.monotonic()
    cutoff = now - current_app.config["LOGIN_WINDOW_SECONDS"]
    with limiter["lock"]:
        counts = []
        for key in keys:
            recent = [
                value
                for value in limiter["attempts"].get(key, ())
                if value > cutoff
            ]
            if add:
                recent.append(now)
            if recent:
                limiter["attempts"][key] = recent
            else:
                limiter["attempts"].pop(key, None)
            counts.append(len(recent))
        return max(counts, default=0)


def _clear_attempts(keys):
    limiter = current_app.extensions["daleel_login_limiter"]
    with limiter["lock"]:
        for key in keys:
            limiter["attempts"].pop(key, None)


def authenticated_user() -> User | None:
    uid = session.get("user_id")
    if not isinstance(uid, int) or isinstance(uid, bool):
        return None
    user = db.session.get(User, uid)
    if (user is None or user.account_status != "Active"
            or not user.password_initialized or user.password_hash is None
            or session.get("session_version", 0) != user.session_version):
        return None
    return user


def current_user() -> User | None:
    user = authenticated_user()
    return user if user and not user.must_change_password else None


def login_destination(user):
    if user.must_change_password:
        return "auth.change_password"
    return "admin.accounts" if user.is_admin else "main.upload"


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:
            return redirect(url_for("auth.login"))
        if not user.is_admin:
            abort(403, description="Administrator access is required for that page.")
        return view(*args, **kwargs)
    return wrapped


def analyst_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:
            return redirect(url_for("auth.login"))
        if user.role != "analyst":
            abort(403, description="Analyst access is required for missions.")
        return view(*args, **kwargs)
    return wrapped


@bp.route("/login", methods=["GET", "POST"])
def login():
    if authenticated_user() is not None:
        return redirect(url_for(login_destination(authenticated_user())))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        keys = _attempt_keys(username)
        if _failed_attempts(keys) >= current_app.config["LOGIN_MAX_ATTEMPTS"]:
            flash("Too many login attempts. Try again later.", "error")
            return (
                render_template("login.html"),
                429,
                {"Retry-After": str(current_app.config["LOGIN_WINDOW_SECONDS"])},
            )
        user = User.query.filter_by(username=username).first() if len(username) <= 100 else None
        # A failed attempt must not disclose which field was wrong.
        if (
            user is None
            or len(password) > 256
            or user.account_status != "Active"
            or not user.password_initialized or user.password_hash is None
            or not check_password_hash(user.password_hash, unicodedata.normalize("NFC", password))
        ):
            _failed_attempts(keys, add=True)
            flash("Invalid username or password.", "error")
        else:
            # A successful account login clears that account's failures; the
            # address bucket ages out normally so one known account cannot be
            # used to reset brute-force attempts against other accounts.
            _clear_attempts(keys[1:])
            session.clear()
            session["user_id"] = user.user_id
            session["session_version"] = user.session_version
            return redirect(url_for(login_destination(user)))
    return render_template("login.html")


@bp.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))


@bp.route("/change-password", methods=["GET", "POST"])
def change_password():
    user = authenticated_user()
    if user is None:
        session.clear()
        return redirect(url_for("auth.login"))
    if not user.must_change_password:
        return redirect(url_for(login_destination(user)))
    if request.method == "POST":
        # Serialize with administrator resets and other password-change requests.
        db.session.rollback()
        db.session.execute(text("BEGIN IMMEDIATE"))
        user = authenticated_user()
        if user is None or not user.must_change_password:
            db.session.rollback()
            session.clear()
            return redirect(url_for("auth.login"))
        password = unicodedata.normalize("NFC", request.form.get("password", ""))
        confirm = unicodedata.normalize("NFC", request.form.get("confirm_password", ""))
        error = password_error(password, user)
        if password != confirm:
            error = "The passwords do not match."
        if not error and user.check_password(password):
            error = "Choose a password different from your temporary password."
        if error:
            db.session.rollback()
            flash(error, "error")
            return render_template("change_password.html"), 400
        user.set_password(password)
        user.must_change_password = False
        user.session_version += 1  # Ends every earlier session of this account.
        db.session.commit()
        # Continue in a fresh session bound to the new session version.
        session.clear()
        session["user_id"] = user.user_id
        session["session_version"] = user.session_version
        flash("Password saved.", "success")
        return redirect(url_for(login_destination(user)))
    return render_template("change_password.html")
