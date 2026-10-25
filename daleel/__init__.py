import hmac
import os
import secrets
import threading
from pathlib import Path

import click
from flask import Flask, abort, flash, g, redirect, request, session, url_for
from sqlalchemy import event, inspect
from sqlalchemy.engine import URL, make_url
from werkzeug.middleware.proxy_fix import ProxyFix

from .models import db
from .pipeline.detector import DETECTION_CONF


PROTECTED_DATABASE = Path(__file__).resolve().parent.parent / "instance" / "daleel.db"
WEIGHTS_DIR = PROTECTED_DATABASE.parent.parent / "weights"


def _database_url(app):
    """Resolve the database path and reject the protected location before any engine can connect."""
    url = make_url(app.config["SQLALCHEMY_DATABASE_URI"])
    if (url.drivername not in ("sqlite", "sqlite+pysqlite") or url.query
            or url.host or url.username or url.password or url.port
            or app.config.get("SQLALCHEMY_BINDS")
            or app.config.get("SQLALCHEMY_ENGINE_OPTIONS")):
        raise ValueError("Use a plain SQLite file URL without binds or engine overrides.")
    if url.database == ":memory:":
        if not app.testing:
            raise ValueError("In-memory databases are reserved for tests.")
        return url
    if not url.database or url.database.startswith("file:"):
        raise ValueError("An explicit SQLite file path is required.")
    target = Path(url.database)
    if not target.is_absolute():
        target = Path(app.instance_path) / target
    target = target.resolve()
    protected = PROTECTED_DATABASE.resolve()
    same_file = target == protected
    if not same_file and target.exists() and protected.exists():
        same_file = target.samefile(protected)  # metadata only, also catches hard links
    if same_file:
        raise ValueError("The protected legacy database must not be opened.")
    return URL.create("sqlite", database=str(target))


def _sqlite_foreign_keys(connection, connection_record):
    cursor = connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def initialize_database():
    """Explicitly initialize an empty database; never migrate existing tables."""
    if inspect(db.engine).get_table_names():
        raise RuntimeError("Database already contains tables; initialization refused.")
    db.create_all()


def _load_or_create_secret(instance_path: str) -> str:
    """Return an environment secret or a private, persistent local secret."""
    configured = os.environ.get("DALEEL_SECRET_KEY")
    if configured:
        return configured

    secret_path = os.path.join(instance_path, "secret_key")
    try:
        with open(secret_path, "r", encoding="ascii") as fh:
            value = fh.read().strip()
            if value:
                return value
    except FileNotFoundError:
        pass

    value = secrets.token_hex(32)
    try:
        fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        with open(secret_path, "r", encoding="ascii") as fh:
            return fh.read().strip()
    with os.fdopen(fd, "w", encoding="ascii") as fh:
        fh.write(value)
    return value


def create_app(instance_path: str | None = None, test_config: dict | None = None) -> Flask:
    if instance_path is None:
        from .account_runtime import resolve_instance_path
        instance_path = str(resolve_instance_path(PROTECTED_DATABASE.parent))
    app = Flask(__name__, instance_path=instance_path, instance_relative_config=True)

    app.config.update(
        SQLALCHEMY_DATABASE_URI=URL.create("sqlite", database=os.path.join(app.instance_path, "daleel.db")),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("DALEEL_COOKIE_SECURE") == "1",
        # Number of reverse proxies (for example Caddy or Nginx) in front of DALEEL.
        TRUSTED_PROXIES=os.environ.get("DALEEL_TRUSTED_PROXIES", "0"),
        # 500 MB video limit, plus a margin for the SRT part of the multipart body.
        MAX_CONTENT_LENGTH=505 * 1024 * 1024,
        VIDEO_MAX_BYTES=500 * 1024 * 1024,
        UPLOAD_DIR=os.path.join(app.instance_path, "uploads"),
        ARTIFACT_DIR=os.path.join(app.instance_path, "artifacts"),
        # Fixed source sampling and matching; one evaluated confidence threshold for both models.
        SAMPLE_EVERY_N=15,
        DETECTION_CONF=DETECTION_CONF,
        INFERENCE_SIZES=(640, 960, 1280),
        # 1280 px (longer side) is both the application and the detection-evaluation setting.
        INFERENCE_IMGSZ=int(os.environ.get("DALEEL_IMGSZ", "1280")),
        WEIGHTS_N=os.environ.get("DALEEL_WEIGHTS_N", str(WEIGHTS_DIR / "yolov8n_sar_v1.pt")),
        WEIGHTS_S=os.environ.get("DALEEL_WEIGHTS_S", str(WEIGHTS_DIR / "yolov8s_sar_v1.pt")),
        TRACK_IOU=0.30,
        TRACK_MAX_GAP=3,  # Up to three MISSED sampled frames.
        LOGIN_MAX_ATTEMPTS=5,
        LOGIN_WINDOW_SECONDS=300,
    )
    if test_config:
        app.config.update(test_config)
    app.config["SQLALCHEMY_DATABASE_URI"] = _database_url(app)
    # Hide sensitive parameter values in unexpected driver error diagnostics.
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"hide_parameters": True}
    os.makedirs(app.instance_path, exist_ok=True)
    database_path = app.config["SQLALCHEMY_DATABASE_URI"].database
    if database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = _load_or_create_secret(app.instance_path)
    if app.config["INFERENCE_IMGSZ"] not in app.config["INFERENCE_SIZES"]:
        raise RuntimeError("DALEEL_IMGSZ must be one of 640, 960, or 1280.")
    proxies = str(app.config["TRUSTED_PROXIES"]).strip()
    if not proxies.isdigit():
        raise RuntimeError("DALEEL_TRUSTED_PROXIES must be 0 or the number of reverse proxies.")
    if int(proxies):
        # Trust X-Forwarded-For/Proto/Host from exactly this many proxies, so that the
        # sign-in rate limit sees real client addresses and HTTPS is recognized.
        # Trusting more proxies than actually exist would let clients forge these headers.
        hops = int(proxies)
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=hops, x_proto=hops, x_host=hops)
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["ARTIFACT_DIR"], exist_ok=True)
    app.extensions["daleel_login_limiter"] = {
        "attempts": {},
        "lock": threading.Lock(),
    }

    db.init_app(app)
    with app.app_context():
        event.listen(db.engine, "connect", _sqlite_foreign_keys)

    @app.cli.command("init-db")
    def init_db_command():
        """Create the canonical tables in an empty development database."""
        try:
            initialize_database()
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo("Initialized the canonical DALEEL database.")

    from .auth import bp as auth_bp
    from .routes import bp as main_bp
    from .admin import bp as admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)

    def csrf_token() -> str:
        token = session.get("_csrf_token")
        if token is None:
            token = secrets.token_urlsafe(32)
            session["_csrf_token"] = token
        return token

    def csp_nonce() -> str:
        nonce = getattr(g, "csp_nonce", None)
        if nonce is None:
            nonce = secrets.token_urlsafe(18)
            g.csp_nonce = nonce
        return nonce

    @app.before_request
    def protect_post_requests():
        if request.method != "POST":
            return None
        expected = session.get("_csrf_token", "")
        supplied = request.form.get("_csrf_token", "")
        if not expected or not supplied or not hmac.compare_digest(expected, supplied):
            abort(400, description="Invalid or missing CSRF token.")
        return None

    @app.before_request
    def restrict_temporary_credentials():
        from .auth import authenticated_user
        user = authenticated_user()
        if user and user.must_change_password and request.endpoint not in (
                "auth.change_password", "auth.logout", "static"):
            return redirect(url_for("auth.change_password"))
        return None

    @app.context_processor
    def inject_context():
        from .auth import current_user
        from .mission_storage import original_video_name
        from .presentation import format_mission_clock, saudi_datetime, target_label
        return {
            "g_user": current_user(),
            "csrf_token": csrf_token,
            "csp_nonce": csp_nonce,
            "saudi_datetime": saudi_datetime,
            "format_mission_clock": format_mission_clock,
            "target_label": target_label,
            "original_video_name": lambda mission: original_video_name(app, mission),
        }

    @app.after_request
    def add_security_headers(response):
        nonce = csp_nonce()
        response.headers["Content-Security-Policy"] = "; ".join((
            "default-src 'self'",
            f"script-src 'self' 'nonce-{nonce}'",
            # Leaflet and the progress UI set element styles dynamically.
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: "
            "https://tile.openstreetmap.org https://*.tile.openstreetmap.org "
            "https://server.arcgisonline.com",
            "connect-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            "frame-ancestors 'none'",
            "form-action 'self'",
        ))
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        if request.endpoint != "static":
            response.headers.setdefault("Cache-Control", "no-store")
        if request.endpoint in ("auth.change_password", "admin.accounts", "admin.reset_password"):
            response.headers["Referrer-Policy"] = "no-referrer"
        if request.is_secure:
            # Browsers then refuse plain-HTTP connections to this host for one year.
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.errorhandler(413)
    def too_large(_e):
        return "Upload rejected: the complete request exceeds the configured size limit.", 413

    return app
