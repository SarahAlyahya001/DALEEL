import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import URL

import daleel
from daleel import create_app, initialize_database
from daleel.models import Mission, User, db


@pytest.mark.parametrize("variant", ["absolute", "relative", "dot_segments", "default_instance"])
def test_protected_path_is_refused_before_engine_or_secret_access(tmp_path, monkeypatch, variant):
    init = Mock(side_effect=AssertionError("Engine registration must not run"))
    secret = Mock(side_effect=AssertionError("Secret access must not run"))
    monkeypatch.setattr(db, "init_app", init)
    monkeypatch.setattr(daleel, "_load_or_create_secret", secret)
    protected = daleel.PROTECTED_DATABASE
    instance = str(tmp_path / "instance")
    settings = {}
    if variant == "absolute":
        settings["SQLALCHEMY_DATABASE_URI"] = URL.create("sqlite", database=str(protected))
    elif variant == "dot_segments":
        settings["SQLALCHEMY_DATABASE_URI"] = URL.create("sqlite", database=str(protected.parent / "stage1" / ".." / "daleel.db"))
    elif variant == "relative":
        instance = str(protected.parent / "stage1")
        settings["SQLALCHEMY_DATABASE_URI"] = "sqlite:///../daleel.db"
    else:
        instance = str(protected.parent)
    with pytest.raises(ValueError, match="protected legacy database"):
        create_app(instance, settings)
    init.assert_not_called()
    secret.assert_not_called()


@pytest.mark.parametrize("settings", [
    {"SQLALCHEMY_DATABASE_URI": "sqlite:///file:legacy.db?uri=true"},
    {"SQLALCHEMY_DATABASE_URI": "sqlite:///file:legacy.db"},
    {"SQLALCHEMY_DATABASE_URI": "sqlite://"},
    {"SQLALCHEMY_BINDS": {"other": "sqlite:///other.db"}},
    {"SQLALCHEMY_ENGINE_OPTIONS": {"connect_args": {"uri": True}}},
    {"SQLALCHEMY_DATABASE_URI": "postgresql://localhost/test"},
])
def test_alternate_connection_paths_cannot_bypass_guard(tmp_path, monkeypatch, settings):
    init = Mock()
    monkeypatch.setattr(db, "init_app", init)
    with pytest.raises(ValueError):
        create_app(str(tmp_path / "instance"), settings)
    init.assert_not_called()


def test_hard_link_to_synthetic_protected_file_is_refused(tmp_path, monkeypatch):
    protected = tmp_path / "synthetic-legacy.db"
    protected.write_bytes(b"synthetic sentinel, not a real mission database")
    alias = tmp_path / "alias.db"
    alias.hardlink_to(protected)
    monkeypatch.setattr(daleel, "PROTECTED_DATABASE", protected)
    init = Mock()
    monkeypatch.setattr(db, "init_app", init)
    with pytest.raises(ValueError, match="protected legacy database"):
        create_app(str(tmp_path / "instance"), {"SQLALCHEMY_DATABASE_URI": URL.create("sqlite", database=str(alias))})
    init.assert_not_called()


def test_default_development_location_without_creating_real_database(tmp_path, monkeypatch):
    # Substitute the project location; never initialize the real instance here.
    monkeypatch.setattr(daleel, "PROTECTED_DATABASE", tmp_path / "instance" / "daleel.db")
    app = create_app()
    expected = tmp_path / "instance" / "v2" / "daleel.db"
    assert Path(app.instance_path) == expected.parent
    assert Path(app.config["SQLALCHEMY_DATABASE_URI"].database) == expected
    assert not expected.exists()  # factory registered an engine but never connected
    assert "daleel_processor" not in app.extensions


def test_explicit_initialization_creates_only_canonical_tables(tmp_path):
    app = create_app(str(tmp_path / "instance"), {"TESTING": True})
    assert not (tmp_path / "instance" / "daleel.db").exists()
    result = app.test_cli_runner().invoke(args=["init-db"])
    assert result.exit_code == 0, result.output
    with app.app_context():
        assert set(inspect(db.engine).get_table_names()) == {"users", "missions", "observations"}
        assert db.session.execute(text("PRAGMA foreign_keys")).scalar() == 1
        db.session.remove()
        db.engine.dispose()
    second = app.test_cli_runner().invoke(args=["init-db"])
    assert second.exit_code != 0
    assert "initialization refused" in second.output
    with app.app_context():
        db.engine.dispose()


def test_existing_synthetic_database_is_not_migrated(tmp_path):
    instance = tmp_path / "instance"
    instance.mkdir()
    database_path = instance / "daleel.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('unchanged')")
    before = database_path.read_bytes()
    app = create_app(str(instance), {"TESTING": True})
    assert database_path.read_bytes() == before
    result = app.test_cli_runner().invoke(args=["init-db"])
    assert result.exit_code != 0
    with app.app_context():
        assert inspect(db.engine).get_table_names() == ["sentinel"]
        db.engine.dispose()
    assert database_path.read_bytes() == before


def test_factory_restart_does_not_change_mission_state(tmp_path):
    instance = str(tmp_path / "instance")
    first = create_app(instance, {"TESTING": True})
    with first.app_context():
        initialize_database()
        user = User(username="restart", role="analyst")
        user.set_password("synthetic-password")
        mission = Mission(analyst=user, video_filename="m.mp4", srt_filename="m.srt")
        db.session.add(mission)
        db.session.commit()
        mission_id = mission.mission_id
        db.session.remove()
        db.engine.dispose()
    second = create_app(instance, {"TESTING": True})
    with second.app_context():
        assert db.session.get(Mission, mission_id).status == "Processing"
        assert len(inspect(db.engine).get_table_names()) == 3
        db.session.remove()
        db.engine.dispose()


def test_test_configuration_is_applied_before_registration(tmp_path):
    app = create_app(str(tmp_path / "instance"), {"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    with app.app_context():
        initialize_database()
        assert db.engine.url.database == ":memory:"
        assert db.session.execute(text("PRAGMA foreign_keys")).scalar() == 1
        db.session.remove()
        db.engine.dispose()
    assert not (tmp_path / "instance" / "daleel.db").exists()
