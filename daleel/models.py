"""SQLite data model for accounts, missions and reviewed observations."""

import math
import re
from datetime import datetime
from numbers import Real
from pathlib import PureWindowsPath

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.orm import validates
from werkzeug.security import check_password_hash, generate_password_hash


db = SQLAlchemy()


def _choice(key, value, choices):
    if value not in choices:
        raise ValueError(f"{key} must be one of {', '.join(choices)}.")
    return value


def _number(key, value, low=0, high=None, optional=False, integer=False):
    if value is None and optional:
        return None
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < low
            or (high is not None and value > high)
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{key} is outside its permitted numeric range or type.")
    return value


def _relative_path(key, value):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{key} must be a nonempty relative path.")
    normalized = value.replace("\\", "/")
    if (PureWindowsPath(value).drive or normalized.startswith("/")
            or any(part in ("", ".", "..") for part in normalized.split("/"))):
        raise ValueError(f"{key} must stay inside its media directory.")
    return normalized


class User(db.Model):
    __tablename__ = "users"
    __table_args__ = (
        db.CheckConstraint("length(trim(username)) > 0 AND length(username) <= 100",
                           name="ck_users_username"),
        db.CheckConstraint("password_hash IS NULL OR length(trim(password_hash)) > 0",
                           name="ck_users_password_hash"),
        db.CheckConstraint("account_status != 'Active' OR "
                           "(password_hash IS NOT NULL AND password_initialized = 1)",
                           name="ck_users_active_credentials"),
        db.CheckConstraint("role IN ('administrator', 'analyst')", name="ck_users_role"),
        db.CheckConstraint("account_status IN ('Active', 'Invited', 'Suspended')", name="ck_users_status"),
        db.CheckConstraint("session_version >= 0", name="ck_users_session_version"),
        db.CheckConstraint("must_change_password IN (0, 1)", name="ck_users_must_change_password"),
    )

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=True)
    role = db.Column(db.String(20), nullable=False)
    full_name = db.Column(db.String(150), nullable=False, default="", server_default="")
    email = db.Column(db.String(254), unique=True)
    account_status = db.Column(db.String(20), nullable=False, default="Active", server_default="Active")
    session_version = db.Column(db.Integer, nullable=False, default=0, server_default="0")
    password_initialized = db.Column(db.Boolean, nullable=False, default=True, server_default="1")

    must_change_password = db.Column(db.Boolean, nullable=False, default=False, server_default="0")

    @property
    def display_name(self):
        return self.full_name or self.username

    @validates("account_status")
    def validate_account_status(self, key, value):
        return _choice(key, value, ("Active", "Invited", "Suspended"))

    missions = db.relationship("Mission", back_populates="analyst")

    @validates("username")
    def validate_username(self, key, value):
        if not isinstance(value, str) or not value.strip() or len(value) > 100:
            raise ValueError("username must contain 1-100 characters and not be blank.")
        return value

    @validates("password_hash")
    def validate_password_hash(self, key, value):
        if value is None:
            return None  # Invited accounts have no password until atomic activation.
        # Accept salted hashes produced by Werkzeug, never plaintext passwords.
        if not isinstance(value, str) or not re.fullmatch(
            r"(?:scrypt:[1-9]\d*:[1-9]\d*:[1-9]\d*\$[A-Za-z0-9]+\$[0-9a-f]{128}"
            r"|pbkdf2:sha256:[1-9]\d*\$[A-Za-z0-9]+\$[0-9a-f]{64})", value
        ):
            raise ValueError("password_hash must be a supported salted one-way hash.")
        return value

    @validates("role")
    def validate_role(self, key, value):
        return _choice(key, value, ("administrator", "analyst"))

    def set_password(self, password):
        if not isinstance(password, str) or not password:
            raise ValueError("password is required.")
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return bool(self.password_hash) and check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == "administrator"


class Mission(db.Model):
    __tablename__ = "missions"
    __table_args__ = (
        db.CheckConstraint("status IN ('Processing', 'Complete', 'Failed')", name="ck_missions_status"),
        db.CheckConstraint("elapsed_seconds BETWEEN 0 AND 1.7976931348623157e308",
                           name="ck_missions_elapsed"),
        db.CheckConstraint("typeof(yolov8n_frames) = 'integer' AND yolov8n_frames >= 0",
                           name="ck_missions_yolov8n_frames"),
        db.CheckConstraint("typeof(yolov8s_frames) = 'integer' AND yolov8s_frames >= 0",
                           name="ck_missions_yolov8s_frames"),
        db.CheckConstraint("length(trim(video_filename)) > 0", name="ck_missions_video"),
        db.CheckConstraint("length(trim(srt_filename)) > 0", name="ck_missions_srt"),
    )

    mission_id = db.Column(db.Integer, primary_key=True)
    analyst_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False, index=True)
    video_filename = db.Column(db.Text, nullable=False)
    srt_filename = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    status = db.Column(db.String(20), nullable=False, default="Processing", server_default="Processing")
    elapsed_seconds = db.Column(db.Float)
    # Sampled frames processed by each model; both models process every sampled frame.
    yolov8n_frames = db.Column(db.Integer, nullable=False, default=0, server_default="0")
    yolov8s_frames = db.Column(db.Integer, nullable=False, default=0, server_default="0")

    analyst = db.relationship("User", back_populates="missions")
    observations = db.relationship("Observation", back_populates="mission")

    @validates("video_filename", "srt_filename")
    def validate_path(self, key, value):
        return _relative_path(key, value)

    @validates("submitted_at")
    def validate_submitted_at(self, key, value):
        if not isinstance(value, datetime):
            raise ValueError("submitted_at must be a datetime.")
        return value

    @validates("status")
    def validate_status(self, key, value):
        return _choice(key, value, ("Processing", "Complete", "Failed"))

    @validates("elapsed_seconds")
    def validate_duration(self, key, value):
        return _number(key, value, optional=True)

    @validates("yolov8n_frames", "yolov8s_frames")
    def validate_counter(self, key, value):
        return _number(key, value, integer=True)


class Observation(db.Model):
    __tablename__ = "observations"
    __table_args__ = (
        db.CheckConstraint("video_time_ms BETWEEN 0 AND 1.7976931348623157e308",
                           name="ck_observations_time"),
        db.CheckConstraint("typeof(first_frame_index) = 'integer' AND first_frame_index >= 0",
                           name="ck_observations_first"),
        db.CheckConstraint("typeof(last_frame_index) = 'integer' AND last_frame_index >= first_frame_index",
                           name="ck_observations_last"),
        db.CheckConstraint("typeof(frame_count) = 'integer' AND frame_count > 0",
                           name="ck_observations_count"),
        db.CheckConstraint("length(trim(frame_path)) > 0", name="ck_observations_path"),
        db.CheckConstraint("target_class IN ('person', 'car')", name="ck_observations_class"),
        db.CheckConstraint("agreement IN ('Both models', 'One model')", name="ck_observations_agreement"),
        db.CheckConstraint("yolov8n_score BETWEEN 0 AND 1", name="ck_observations_yolov8n_score"),
        db.CheckConstraint("yolov8s_score BETWEEN 0 AND 1", name="ck_observations_yolov8s_score"),
        # The label states which models reported the representative detection, and nothing else.
        db.CheckConstraint("(agreement = 'Both models' AND yolov8n_score IS NOT NULL AND yolov8s_score IS NOT NULL)"
                           " OR (agreement = 'One model' AND (yolov8n_score IS NULL) != (yolov8s_score IS NULL))",
                           name="ck_observations_agreement_scores"),
        db.CheckConstraint("latitude BETWEEN -90 AND 90", name="ck_observations_latitude"),
        db.CheckConstraint("longitude BETWEEN -180 AND 180", name="ck_observations_longitude"),
        db.CheckConstraint("review_status IN ('Unreviewed', 'Confirmed', 'Rejected')", name="ck_observations_review"),
    )

    observation_id = db.Column(db.Integer, primary_key=True)
    mission_id = db.Column(db.Integer, db.ForeignKey("missions.mission_id"), nullable=False, index=True)
    video_time_ms = db.Column(db.Float, nullable=False)
    first_frame_index = db.Column(db.Integer, nullable=False)
    last_frame_index = db.Column(db.Integer, nullable=False)
    frame_count = db.Column(db.Integer, nullable=False)
    frame_path = db.Column(db.Text, nullable=False)
    target_class = db.Column(db.String(20), nullable=False)
    agreement = db.Column(db.String(20), nullable=False)
    yolov8n_score = db.Column(db.Float)  # NULL when YOLOv8n did not report the representative detection
    yolov8s_score = db.Column(db.Float)  # NULL when YOLOv8s did not report the representative detection
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    review_status = db.Column(db.String(20), nullable=False, default="Unreviewed", server_default="Unreviewed")

    mission = db.relationship("Mission", back_populates="observations")

    @validates("video_time_ms")
    def validate_time(self, key, value):
        return _number(key, value)

    @validates("first_frame_index", "last_frame_index", "frame_count")
    def validate_frame(self, key, value):
        return _number(key, value, low=1 if key == "frame_count" else 0, integer=True)

    @validates("frame_path")
    def validate_path(self, key, value):
        return _relative_path(key, value)

    @validates("target_class")
    def validate_class(self, key, value):
        return _choice(key, value, ("person", "car"))

    @validates("agreement")
    def validate_agreement(self, key, value):
        return _choice(key, value, ("Both models", "One model"))

    @validates("yolov8n_score", "yolov8s_score")
    def validate_score(self, key, value):
        return _number(key, value, high=1, optional=True)

    @property
    def detection_score(self):
        """The higher score of the models that reported the representative detection."""
        return max(s for s in (self.yolov8n_score, self.yolov8s_score) if s is not None)

    @validates("latitude", "longitude")
    def validate_coordinate(self, key, value):
        limit = 90 if key == "latitude" else 180
        return _number(key, value, low=-limit, high=limit, optional=True)

    @validates("review_status")
    def validate_review(self, key, value):
        return _choice(key, value, ("Unreviewed", "Confirmed", "Rejected"))


@event.listens_for(db.session.session_factory.class_, "before_flush")
def _validate_frame_span(session, flush_context, instances):
    # Validate after both assignments, so callers can move an entire span.
    for item in session.new.union(session.dirty):
        if (isinstance(item, Observation) and item.first_frame_index is not None
                and item.last_frame_index is not None
                and item.last_frame_index < item.first_frame_index):
            raise ValueError("last_frame_index must be at least first_frame_index.")
        if isinstance(item, Observation) and item.agreement is not None:  # NULL is left to NOT NULL
            reported = (item.yolov8n_score is not None, item.yolov8s_score is not None)
            expected = {(True, True): "Both models", (True, False): "One model", (False, True): "One model"}
            if expected.get(reported) != item.agreement:
                raise ValueError("The agreement label must match the models that reported the detection.")
