"""Verified archive of data from the earlier version before the dual-model version starts."""
import io
import json
import sqlite3
from contextlib import closing
from unittest.mock import Mock

import pytest

import run
from daleel.legacy_archive import ArchiveError, archive_previous_installations


class Terminal(io.StringIO):
    def isatty(self):
        return True


def earlier_installation(root, location="app"):
    folder = root / location
    (folder / "uploads" / "mission-1" / "frames").mkdir(parents=True)
    with closing(sqlite3.connect(folder / "daleel.db")) as connection:
        connection.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, support_label TEXT)")
        connection.executemany("INSERT INTO observations VALUES (?, ?)", [(1, "Dual-Model"), (2, "Screening-Only")])
        connection.commit()
    (folder / "uploads" / "mission-1" / "video.mp4").write_bytes(b"synthetic video")
    (folder / "uploads" / "mission-1" / "frames" / "track-1.jpg").write_bytes(b"synthetic evidence")
    (folder / "secret_key").write_text("synthetic key", encoding="ascii")
    return folder


def contents(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("location", ["app", "review/current", "stage1"])
def test_earlier_data_is_copied_verified_and_left_in_place(tmp_path, location):
    root = tmp_path / "instance"
    folder = earlier_installation(root, location)
    before = contents(folder)
    output = io.StringIO()
    index = archive_previous_installations(root, output)
    archive = root / index[location]["archive"]
    manifest = json.loads((archive / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"daleel.db", "secret_key", "uploads/mission-1/video.mp4",
                                      "uploads/mission-1/frames/track-1.jpg"}
    assert contents(folder) == before  # the original folder is unchanged
    assert (archive / "uploads/mission-1/video.mp4").read_bytes() == b"synthetic video"
    with closing(sqlite3.connect(archive / "daleel.db")) as copy:
        assert copy.execute("SELECT * FROM observations ORDER BY id").fetchall() == [(1, "Dual-Model"), (2, "Screening-Only")]
    assert "release 2026-10-14" in (archive / "RESTORE.md").read_text(encoding="utf-8")
    assert "Verified archive of 4 files" in output.getvalue()
    assert not list(root.glob("archive/*.partial"))


def test_archive_runs_once(tmp_path):
    root = tmp_path / "instance"
    earlier_installation(root)
    archive_previous_installations(root, io.StringIO())
    again = io.StringIO()
    archive_previous_installations(root, again)
    assert again.getvalue() == ""
    assert len([p for p in (root / "archive").iterdir() if p.is_dir()]) == 1


def test_several_earlier_layouts_are_all_archived(tmp_path):
    root = tmp_path / "instance"
    for location in ("app", "stage1"):
        earlier_installation(root, location)
    index = archive_previous_installations(root, io.StringIO())
    assert set(index) == {"app", "stage1"}


def test_unreadable_database_stops_without_changes(tmp_path):
    root = tmp_path / "instance"
    folder = root / "app"
    folder.mkdir(parents=True)
    (folder / "daleel.db").write_bytes(b"not a database")
    with pytest.raises(ArchiveError, match="could not be read"):
        archive_previous_installations(root, io.StringIO())
    assert (folder / "daleel.db").read_bytes() == b"not a database"
    assert not list((root / "archive").glob("*"))


def test_fresh_installation_has_nothing_to_archive(tmp_path):
    root = tmp_path / "instance"
    assert archive_previous_installations(root, io.StringIO()) == {}
    assert not (root / "archive").exists()


def test_launcher_archives_before_opening_the_new_database(tmp_path, monkeypatch):
    root = tmp_path / "instance"
    earlier_installation(root)
    monkeypatch.setattr(run, "INSTANCE_ROOT", root)
    order = []
    original = run.archive_previous_installations
    monkeypatch.setattr(run, "archive_previous_installations",
                        lambda *args: order.append("archive") or original(*args))
    monkeypatch.setattr(run, "create_app", Mock(side_effect=lambda: order.append("open") or (_ for _ in ()).throw(
        RuntimeError("stop after the archive"))))
    assert run.main(["init"], Terminal()) == 1
    assert order == ["archive", "open"]
    assert (root / "archive" / "index.json").is_file()
