"""Mission history, deletion and interface refinements, using isolated media and databases only."""
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from werkzeug.datastructures import MultiDict

from daleel.mission_storage import display_basename, original_video_name, save_submission_names
from daleel.models import Mission, Observation, db
from daleel.presentation import format_mission_clock, saudi_datetime, target_label
from test_results import env  # Reuse the isolated owned-mission fixture.


def delete(env, **values):
    return env.client.post(f"/missions/{env.mission}/delete", data={
        "_csrf_token": "synthetic-token", "confirm": "delete", **values})


def test_delete_requires_confirmation_and_archives_owned_mission(env):
    response = env.client.get(f"/missions/{env.mission}/delete")
    assert response.status_code == 200 and env.folder.is_dir()
    assert b'name="confirm" value="delete"' in response.data
    assert response.headers["Cache-Control"] == "private, no-store"
    assert delete(env).status_code == 303
    archives = list((Path(env.app.instance_path) / "trash").iterdir())
    assert len(archives) == 1
    archive = archives[0]
    assert (archive / "media/video.mp4").read_bytes() == b"synthetic input; never decoded"
    assert (archive / "media/frames/evidence.jpg").is_file()
    backup = json.loads((archive / "mission.json").read_text(encoding="utf-8"))
    assert backup["mission"]["mission_id"] == env.mission
    assert {o["review_status"] for o in backup["observations"]} == {"Unreviewed", "Confirmed", "Rejected"}
    assert all("password" not in key for key in backup["mission"])
    assert not env.folder.exists()
    with env.app.app_context():
        assert db.session.get(Mission, env.mission) is None
        assert Observation.query.count() == 0
    for endpoint in (f"/results/{env.mission}", f"/frame/{env.ids[0]}",
                     f"/export/{env.mission}/pdf", f"/export/{env.mission}/gpx"):
        assert env.client.get(endpoint).status_code == 404
    history = env.client.get("/history").data
    assert b"Mission deleted." in history
    assert b"local trash" not in history and b"this computer" not in history
    assert b"local trash" not in response.data and b"this computer" not in response.data


@pytest.mark.parametrize("user,code", [(1, 404), (2, 403), (None, 302)])
@pytest.mark.parametrize("method", ["get", "post"])
def test_delete_ownership_role_and_login(env, user, code, method):
    with env.client.session_transaction() as session:
        if user is None:
            session.pop("user_id")
        else:
            session["user_id"] = env.users[user]
    response = getattr(env.client, method)(f"/missions/{env.mission}/delete", data={
        "_csrf_token": "synthetic-token", "confirm": "delete"})
    assert response.status_code == code
    assert env.folder.is_dir()


@pytest.mark.parametrize("values", [{"confirm": ""}, {"confirm": "yes"}, {"extra": "x"},
                                   {"_csrf_token": "wrong"}, {"_csrf_token": ""}])
def test_delete_invalid_post_cannot_remove_data(env, values):
    assert delete(env, **values).status_code == 400
    assert env.folder.is_dir()
    with env.app.app_context():
        assert db.session.get(Mission, env.mission) is not None
        assert Observation.query.count() == 3


def test_delete_rejects_duplicate_confirmation(env):
    response = env.client.post(f"/missions/{env.mission}/delete", data=MultiDict([
        ("_csrf_token", "synthetic-token"), ("confirm", "delete"), ("confirm", "delete")]))
    assert response.status_code == 400 and env.folder.is_dir()


def test_processing_mission_cannot_be_deleted(env):
    with env.app.app_context():
        db.session.get(Mission, env.mission).status = "Processing"
        db.session.commit()
    assert env.client.get(f"/missions/{env.mission}/delete").status_code == 409
    assert delete(env).status_code == 409
    assert env.folder.is_dir()


def test_failed_mission_can_be_deleted(env):
    with env.app.app_context():
        db.session.get(Mission, env.mission).status = "Failed"
        db.session.commit()
    assert delete(env).status_code == 303


@pytest.mark.parametrize("failure", ["move", "commit"])
def test_delete_failure_restores_media_and_database(env, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise OSError("synthetic failure")
    if failure == "move":
        monkeypatch.setattr(Path, "rename", fail)
    else:
        monkeypatch.setattr(db.session, "commit", fail)
    assert delete(env).status_code == 500
    assert (env.folder / "frames/evidence.jpg").is_file()
    assert (env.folder / "video.mp4").is_file()
    assert list((Path(env.app.instance_path) / "trash").iterdir()) == []
    with env.app.app_context():
        assert db.session.get(Mission, env.mission) is not None
        assert Observation.query.count() == 3


def test_shared_media_cannot_be_archived(env):
    with env.app.app_context():
        mission = db.session.get(Mission, env.mission)
        db.session.add(Mission(analyst_id=env.users[1], status="Failed",
                               video_filename=mission.video_filename, srt_filename=mission.srt_filename))
        db.session.commit()
    assert delete(env).status_code == 409 and env.folder.is_dir()


@pytest.mark.parametrize("kind", ["mission", "trash"])
def test_resolved_junction_escape_refuses_archive(env, monkeypatch, kind):
    original = Path.resolve
    target = env.folder if kind == "mission" else Path(env.app.instance_path) / "trash"
    outside = Path(env.app.instance_path).parent / "outside"
    def resolve(path, *args, **kwargs):
        return original(outside if path == target else path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    assert delete(env).status_code == 409
    assert env.folder.is_dir() and not outside.exists()


@pytest.mark.parametrize("contents", ["not json", "[]", '{"video_name": 42}', '{"video_name": ""}', "x" * 8193])
def test_missing_or_invalid_name_metadata_uses_legacy_basename(env, contents):
    with env.app.app_context():
        mission = db.session.get(Mission, env.mission)
        assert original_video_name(env.app, mission) == "video.mp4"
        (env.folder / "submission.json").write_text(contents, encoding="utf-8")
        assert original_video_name(env.app, mission) == "video.mp4"


def test_original_name_display_is_escaped_and_searchable(env):
    save_submission_names(env.folder, 'Saudi <img onerror="bad">.mp4', "matching.srt")
    response = env.client.get("/history")
    assert b"Saudi &lt;img" in response.data and b'<img onerror="bad">' not in response.data
    assert b"MP4 footage" not in response.data and b"SRT telemetry" not in response.data
    assert b"Submitted (AST)" in response.data and b"Saudi time" not in response.data


def test_display_names_never_contain_paths_or_control_characters():
    assert display_basename(r"C:\fakepath\flight.mp4") == "flight.mp4"
    assert display_basename("../../رحلة.mp4") == "رحلة.mp4"
    assert display_basename("a\x00\n.mp4") == "a.mp4"
    assert display_basename(None) == ""


@pytest.mark.parametrize("source,expected", [
    (datetime(2026, 9, 26, 22, 15), "2026-09-27 01:15 +0300"),
    (datetime(2026, 9, 26, 22, 15, tzinfo=timezone.utc), "2026-09-27 01:15 +0300"),
    (datetime(2026, 9, 27, 1, 15, tzinfo=timezone(timedelta(hours=3))), "2026-09-27 01:15 +0300"),
])
def test_saudi_display_handles_timezone_and_date_rollover(source, expected):
    assert saudi_datetime(source).strftime("%Y-%m-%d %H:%M %z") == expected
    assert saudi_datetime(source).tzname() == "AST"


def test_clean_rows_and_evidence_keep_details_separate(env):
    html = env.client.get(f"/results/{env.mission}").data.decode()
    rows = re.findall(r'<a class="detection".*?</a>', html, re.S)
    assert len(rows) == 3
    for row in rows:
        assert "detection__icon" in row and "observation-number" in row
        assert "detection__status" in row
        assert "review-pending-dot" not in row and "detection__support" not in row
        assert "frame_count" not in row and "Video time" not in row and "24.000000" not in row
        assert "detection__badge" not in row and "detection__meta" not in row
    details = re.findall(r'<div class="frame-info">(.*?)</article>', html, re.S)
    assert all("Drone-Position Approximation" not in detail and "Open image" not in detail for detail in details)
    assert "Markers show the UAV position" not in html and "map-help" not in html
    assert "AST" in html and "Arabia Standard Time (UTC+03:00)" in html
    assert "Saudi time" not in html
    with env.app.app_context():
        db.session.get(Mission, env.mission).submitted_at = datetime(2026, 9, 26, 22, 15)
        db.session.commit()
    assert b"27 Sep" in env.client.get("/history").data
    assert b"1:15 AM" in env.client.get("/history").data


def test_upload_brief_removed_and_sidebar_has_accessible_toggle(env):
    html = env.client.get("/upload").data.decode()
    assert "mission-brief" not in html and "Footage &amp; telemetry" not in html
    assert html.count('class="dropzone file-panel"') == 2
    assert 'data-sidebar-toggle aria-controls="primary-sidebar"' in html
    script = env.client.get("/static/interface.js").data.decode()
    assert "sidebar.inert = collapsed" not in script
    assert 'sidebarToggle.setAttribute("aria-expanded"' in script
    aside = re.search(r'<aside.*?</aside>', html, re.S).group(0)
    topbar = re.search(r'<header class="topbar">.*?</header>', html, re.S).group(0)
    assert "data-sidebar-toggle" in aside and "data-sidebar-toggle" not in topbar
    assert 'aria-label="New Mission" data-tooltip="New Mission"' in aside
    assert "sidebar__brand-mark" not in aside


def test_history_has_clean_heading_and_native_mission_links(env):
    html = env.client.get("/history").data.decode()
    assert "Past missions" not in html and "View results" not in html and "View status" not in html
    assert 'data-history-row data-mission-href="/results/' in html
    assert f'aria-label="Open Mission {env.mission}"' in html
    assert 'aria-label="Delete Mission' in html


def test_leaflet_assets_are_local_with_restrictive_script_policy(env):
    html = env.client.get(f"/results/{env.mission}")
    assert "unpkg.com" not in html.headers["Content-Security-Policy"]
    assert env.client.get("/static/vendor/leaflet/leaflet.js").status_code == 200
    assert env.client.get("/static/vendor/leaflet/leaflet.css").status_code == 200
    assert b"maxNativeZoom: 16, maxZoom: 19" in env.client.get("/static/map.js").data


@pytest.mark.parametrize("hour,expected", [(21, "12:05 AM"), (9, "12:05 PM"), (12, "3:05 PM"), (22, "1:05 AM")])
def test_mission_clock_explicit_am_pm_and_ast(hour, expected):
    assert format_mission_clock(datetime(2026, 9, 26, hour, 5)) == expected


def test_formal_vehicle_label_is_presentation_only(env):
    assert target_label("car") == "Vehicle" and target_label("person") == "Person"
    html = env.client.get(f"/results/{env.mission}").data.decode()
    assert '<h2 class="frame-info__class">Vehicle</h2>' in html
    assert 'title="Vehicle' in html and ": Vehicle, Both models" in html
    assert "Visibility" not in html and "Seen 3 times" not in html
    assert "00:00:00.200" in html and "Supporting frames" in html
    with env.app.app_context():
        assert db.session.get(Observation, env.ids[0]).target_class == "car"


def test_upload_heading_and_map_have_no_redundant_labels(env):
    assert b"Mission setup" not in env.client.get("/upload").data
    html = env.client.get(f"/results/{env.mission}").data
    assert b"Satellite detail varies" not in html and b"map-help" not in html
    assert b"UAV positions associated with mission observations" in html


def test_history_uses_bounded_columns_and_am_pm(env):
    html = env.client.get("/history").data.decode()
    assert '<colgroup>' in html and 'class="mission-table__status"' in html
    assert re.search(r'class="date-abs">\d{1,2}:\d{2} (?:AM|PM)</span>', html)
    css = env.client.get("/static/style.css").data.decode()
    assert '.mission-table__actions { width: 64px; }' in css
