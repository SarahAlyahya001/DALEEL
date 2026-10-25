from datetime import datetime

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash

from daleel import create_app, initialize_database
from daleel.models import Mission, Observation, User, db


@pytest.fixture
def database(tmp_path):
    app = create_app(str(tmp_path / "instance"), {"TESTING": True})
    with app.app_context():
        initialize_database()
        yield db.session
        db.session.rollback()
        db.session.remove()
        db.engine.dispose()


@pytest.fixture(scope="module")
def password_hash():
    return generate_password_hash("synthetic-test-password")


@pytest.fixture
def mission(database, password_hash):
    user = User(username="analyst", password_hash=password_hash, role="analyst")
    item = Mission(analyst=user, video_filename="1/mission.mp4", srt_filename="1/mission.srt")
    database.add(item)
    database.commit()
    return item


def observation(mission, **changes):
    values = dict(mission=mission, video_time_ms=0, first_frame_index=0,
                  last_frame_index=0, frame_count=1, frame_path="frames/evidence.jpg",
                  target_class="person", agreement="One model", yolov8n_score=0.3)
    values.update(changes)
    return Observation(**values)


def test_exact_schema_and_relationships(database):
    expected = {
        "users": {"user_id", "username", "password_hash", "role", "full_name", "email",
                  "account_status", "session_version", "password_initialized", "must_change_password"},
        "missions": {"mission_id", "analyst_id", "video_filename", "srt_filename",
                     "submitted_at", "status", "elapsed_seconds", "yolov8n_frames", "yolov8s_frames"},
        "observations": {"observation_id", "mission_id", "video_time_ms", "first_frame_index",
                         "last_frame_index", "frame_count", "frame_path", "target_class", "agreement",
                         "yolov8n_score", "yolov8s_score", "latitude", "longitude", "review_status"},
    }
    inspector = inspect(db.engine)
    assert set(inspector.get_table_names()) == set(expected)
    optional = {"missions": {"elapsed_seconds"},
                "observations": {"yolov8n_score", "yolov8s_score", "latitude", "longitude"}, "users": {"email", "password_hash"}}
    for table, columns in expected.items():
        reflected = inspector.get_columns(table)
        assert {column["name"] for column in reflected} == columns
        assert {column["name"] for column in reflected if column["nullable"]} == optional[table]
    assert inspector.get_pk_constraint("users")["constrained_columns"] == ["user_id"]
    assert inspector.get_pk_constraint("missions")["constrained_columns"] == ["mission_id"]
    assert inspector.get_pk_constraint("observations")["constrained_columns"] == ["observation_id"]
    assert set(inspect(User).relationships.keys()) == {"missions"}
    assert set(inspect(Mission).relationships.keys()) == {"analyst", "observations"}
    assert set(inspect(Observation).relationships.keys()) == {"mission"}
    assert inspector.get_foreign_keys("missions")[0]["referred_columns"] == ["user_id"]
    assert inspector.get_foreign_keys("observations")[0]["referred_columns"] == ["mission_id"]
    assert {m.class_ for m in db.Model.registry.mappers} == {User, Mission, Observation}


def test_defaults_singleton_and_relationship_round_trip(database, mission):
    owner = mission.analyst
    first = observation(mission)
    second = observation(mission, target_class="car", agreement="Both models", yolov8s_score=0.8)
    another = Mission(analyst=owner, video_filename="2/video.mp4", srt_filename="2/video.srt")
    database.add_all([first, second, another])
    database.commit()
    observation_id, mission_id = first.observation_id, mission.mission_id
    database.remove()
    loaded = db.session.get(Observation, observation_id)
    assert loaded.frame_count == 1
    assert loaded.first_frame_index == loaded.last_frame_index == 0
    assert loaded.review_status == "Unreviewed"
    assert loaded.latitude is None and loaded.longitude is None
    assert loaded.yolov8s_score is None
    assert loaded.mission.mission_id == mission_id
    assert len(loaded.mission.observations) == 2
    assert len(loaded.mission.analyst.missions) == 2
    assert loaded.mission.status == "Processing"
    assert loaded.mission.yolov8n_frames == loaded.mission.yolov8s_frames == 0
    assert loaded.mission.elapsed_seconds is None
    assert isinstance(loaded.mission.submitted_at, datetime)


def test_passwords_are_salted_and_verified(database):
    first = User(username="first", role="analyst")
    second = User(username="second", role="administrator")
    first.set_password("synthetic-password")
    second.set_password("synthetic-password")
    assert first.password_hash != second.password_hash
    assert first.password_hash != "synthetic-password"
    database.add_all([first, second])
    database.commit()
    assert first.check_password("synthetic-password")
    assert not first.check_password("incorrect-password")
    assert second.is_admin and not first.is_admin


@pytest.mark.parametrize("value", ["", "plaintext", "sha256$unsalted", "pbkdf2:sha256:1$$" + "a" * 64])
def test_reject_invalid_password_hash(value):
    with pytest.raises(ValueError, match="password_hash"):
        User(password_hash=value)


@pytest.mark.parametrize("value", [None, "", " ", "\t", "x" * 101])
def test_reject_invalid_username(value):
    with pytest.raises(ValueError, match="username"):
        User(username=value)


def test_username_boundary_and_uniqueness(database, password_hash):
    database.add(User(username="x" * 100, password_hash=password_hash, role="analyst"))
    database.commit()
    database.add(User(username="x" * 100, password_hash=password_hash, role="analyst"))
    with pytest.raises(IntegrityError):
        database.commit()


@pytest.mark.parametrize("model,field,value", [
    (User, "role", "admin"), (User, "role", "Analyst"), (User, "role", None),
    (Mission, "status", "pending"), (Mission, "status", "completed"), (Mission, "status", None),
    (Observation, "target_class", "vehicle"), (Observation, "target_class", "truck"),
    (Observation, "agreement", "Dual-Model"), (Observation, "agreement", "Screening-Only"),
    (Observation, "review_status", "confirmed"), (Observation, "review_status", None),
])
def test_reject_superseded_and_invalid_enums(model, field, value):
    with pytest.raises(ValueError, match=field):
        model(**{field: value})


@pytest.mark.parametrize("model,field", [
    (Mission, "yolov8n_frames"), (Mission, "yolov8s_frames"),
    (Observation, "first_frame_index"), (Observation, "last_frame_index"), (Observation, "frame_count"),
])
@pytest.mark.parametrize("value", [-1, 0.5, "1", True, None])
def test_integer_fields_reject_invalid_values(model, field, value):
    with pytest.raises(ValueError, match=field):
        model(**{field: value})


@pytest.mark.parametrize("model,field", [(Mission, "elapsed_seconds"), (Observation, "video_time_ms"),
    (Observation, "yolov8n_score"), (Observation, "yolov8s_score"),
    (Observation, "latitude"), (Observation, "longitude")])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "0.5"])
def test_numeric_fields_reject_nonfinite_and_nonnumeric_values(model, field, value):
    with pytest.raises(ValueError, match=field):
        model(**{field: value})


@pytest.mark.parametrize("field,value", [("yolov8n_score", -0.001), ("yolov8n_score", 1.001),
    ("yolov8s_score", -0.001), ("yolov8s_score", 1.001),
    ("latitude", -90.001), ("latitude", 90.001), ("longitude", -180.001), ("longitude", 180.001),
    ("frame_count", 0), ("video_time_ms", -0.001)])
def test_observation_numeric_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        Observation(**{field: value})


def test_duration_and_submission_validation():
    with pytest.raises(ValueError):
        Mission(elapsed_seconds=-0.001)
    with pytest.raises(ValueError):
        Mission(submitted_at="2026-01-01")
    with pytest.raises(ValueError):
        Mission(submitted_at=None)


@pytest.mark.parametrize("score,latitude,longitude", [(0, -90, -180), (1, 90, 180)])
def test_numeric_boundaries_persist(database, mission, score, latitude, longitude):
    item = observation(mission, agreement="Both models", yolov8n_score=score, yolov8s_score=score,
                       latitude=latitude, longitude=longitude, video_time_ms=0.125)
    mission.elapsed_seconds = 0
    database.add(item)
    database.commit()
    database.expire_all()
    assert item.latitude == latitude and item.longitude == longitude
    assert item.yolov8n_score == score and item.yolov8s_score == score
    assert item.video_time_ms == 0.125


@pytest.mark.parametrize("status", ["Processing", "Complete", "Failed"])
@pytest.mark.parametrize("review", ["Unreviewed", "Confirmed", "Rejected"])
def test_exact_states_round_trip(database, mission, status, review):
    item = observation(mission)
    database.add(item)
    database.commit()
    mission.status = status
    item.review_status = review
    database.commit()
    database.expire_all()
    assert mission.status == status and item.review_status == review


def test_frame_order_checked_on_insert_and_update(database, mission):
    item = observation(mission, first_frame_index=30, last_frame_index=15)
    database.add(item)
    with pytest.raises(ValueError, match="last_frame_index"):
        database.flush()
    database.rollback()
    item = observation(mission, first_frame_index=15, last_frame_index=30)
    database.add(item)
    database.commit()
    item.first_frame_index, item.last_frame_index = 45, 60
    database.commit()  # a valid span can be reassigned in either order
    item.last_frame_index = 30
    with pytest.raises(ValueError, match="last_frame_index"):
        database.flush()


@pytest.mark.parametrize("model,field", [(Mission, "video_filename"), (Mission, "srt_filename"),
                                         (Observation, "frame_path")])
@pytest.mark.parametrize("value", [None, "", " ", "/root/file", "../file", "a/../file",
    "C:\\file", "C:file", "\\\\server\\share\\file", "a\\..\\file", "a\x00b", "."])
def test_paths_reject_absolute_or_escaping_values(model, field, value):
    with pytest.raises(ValueError, match=field):
        model(**{field: value})


def test_relative_paths_normalize_without_opening_media(database, mission):
    item = observation(mission, frame_path="frames\\evidence.jpg")
    database.add(item)
    database.commit()
    assert item.frame_path == "frames/evidence.jpg"


@pytest.mark.parametrize("field", ["username", "password_hash", "role"])
def test_missing_required_user_fields_fail(database, password_hash, field):
    values = dict(username="missing", password_hash=password_hash, role="analyst")
    del values[field]
    database.add(User(**values))
    with pytest.raises(IntegrityError):
        database.flush()


@pytest.mark.parametrize("field", ["analyst", "video_filename", "srt_filename"])
def test_missing_required_mission_fields_fail(database, mission, field):
    values = dict(analyst=mission.analyst, video_filename="video.mp4", srt_filename="video.srt")
    del values[field]
    database.add(Mission(**values))
    with pytest.raises(IntegrityError):
        database.flush()


@pytest.mark.parametrize("field", ["mission", "video_time_ms", "first_frame_index", "last_frame_index",
    "frame_count", "frame_path", "target_class", "agreement"])
def test_missing_required_observation_fields_fail(database, mission, field):
    item = observation(None)
    # Construct without the field, rather than bypassing validators with None.
    values = {column.key: getattr(item, column.key) for column in Observation.__table__.columns
              if column.key not in {"observation_id", "mission_id", "review_status"}}
    values["mission"] = mission
    del values[field]
    database.add(Observation(**values))
    with pytest.raises(IntegrityError):
        database.flush()


@pytest.mark.parametrize("table,field,value", [
    (User, "username", "x" * 101), (User, "role", "admin"),
    (Mission, "status", "completed"), (Mission, "yolov8n_frames", -1),
    (Mission, "yolov8s_frames", 0.5), (Mission, "elapsed_seconds", -1),
    (Observation, "target_class", "vehicle"), (Observation, "agreement", "ALERT"),
    (Observation, "review_status", "confirmed"), (Observation, "frame_count", 0),
    (Observation, "first_frame_index", -1), (Observation, "last_frame_index", -1),
    (Observation, "yolov8n_score", 1.01), (Observation, "yolov8s_score", -0.01),
    (Observation, "latitude", 91), (Observation, "longitude", 181),
    (Observation, "video_time_ms", -1), (Observation, "frame_path", ""),
])
def test_database_checks_also_protect_direct_sql(database, mission, table, field, value):
    database.add(observation(mission))
    database.commit()
    with pytest.raises(IntegrityError):
        database.execute(table.__table__.update().values(**{field: value}))


def test_foreign_keys_reject_orphans_and_parent_deletion(database, mission):
    database.add(Mission(analyst_id=999999, video_filename="v.mp4", srt_filename="v.srt"))
    with pytest.raises(IntegrityError):
        database.flush()
    database.rollback()
    item = observation(None)
    item.mission_id = 999999
    database.add(item)
    with pytest.raises(IntegrityError):
        database.flush()
    database.rollback()
    database.add(observation(mission))
    database.commit()
    with pytest.raises(IntegrityError):
        database.execute(Mission.__table__.delete())
    database.rollback()
    with pytest.raises(IntegrityError):
        database.execute(User.__table__.delete())


@pytest.mark.parametrize("agreement,n_score,s_score", [
    ("Both models", 0.5, None), ("Both models", None, 0.5), ("One model", 0.5, 0.6), ("One model", None, None)])
def test_agreement_label_must_match_the_models_that_reported_the_detection(database, mission, agreement,
                                                                           n_score, s_score):
    database.add(observation(mission, agreement=agreement, yolov8n_score=n_score, yolov8s_score=s_score))
    with pytest.raises(ValueError, match="agreement label"):
        database.flush()


@pytest.mark.parametrize("values", [
    {"agreement": "Both models"}, {"yolov8n_score": None}, {"yolov8s_score": 0.7}])
def test_agreement_consistency_also_protects_direct_sql(database, mission, values):
    database.add(observation(mission))  # One model, YOLOv8n only
    database.commit()
    with pytest.raises(IntegrityError):
        database.execute(Observation.__table__.update().values(**values))


def test_agreement_and_review_status_are_independent(database, mission):
    agreed = observation(mission, agreement="Both models", yolov8n_score=0.6, yolov8s_score=0.9)
    database.add(agreed)
    database.commit()
    assert agreed.review_status == "Unreviewed"  # agreement never confirms an observation
    assert agreed.detection_score == 0.9
    agreed.review_status = "Rejected"
    database.commit()
    assert (agreed.agreement, agreed.review_status) == ("Both models", "Rejected")