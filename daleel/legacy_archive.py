"""Verified, read-only archive of an installation made by the earlier single-model-screening version.

The dual-model version stores its data in a fresh database under instance/v2/. Observations of the
earlier version were produced by different weights, thresholds and labels, so they are not migrated.
Before the new database is first used, every earlier data folder is copied into instance/archive/
and each copy is verified; the original folders are only read and remain where they are.
"""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

LEGACY_LAYOUTS = ("app", "review/current", "stage1")
INDEX = "index.json"
RESTORE_TEXT = """# Restoring an archived DALEEL installation

This folder is a verified copy of {source}, made on {created} before the dual-model version
of DALEEL started with a fresh database. The original folder was not changed.

To use the earlier data again:

1. Stop DALEEL.
2. Install the earlier version (release 2026-10-14) in a separate folder.
3. Copy the contents of this folder into that installation's instance/app folder.
4. Optionally check the copy: every file and its SHA-256 hash is listed in MANIFEST.json.
5. Start the earlier version with python run.py.
"""


class ArchiveError(RuntimeError):
    """Safe operator guidance; the original files are never modified."""


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_digests(connection):
    tables = [name for (name,) in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {table: hashlib.sha256(repr(connection.execute(
        f'SELECT * FROM "{table}" ORDER BY 1').fetchall()).encode("utf-8")).hexdigest() for table in tables}


def _copy_database(source, target):
    """SQLite online backup from a read-only connection, then integrity and row-for-row checks."""
    try:
        # closing(): a connection used as a context manager only commits, and an open handle
        # would keep the copy locked on Windows.
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(target)) as copy:
                original.backup(copy)
                if copy.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise ArchiveError("The archived database failed its integrity check. Nothing was changed.")
                tables = _table_digests(original)
                if tables != _table_digests(copy):
                    raise ArchiveError("The archived database differs from the original. Nothing was changed.")
    except sqlite3.Error as exc:
        raise ArchiveError(f"The earlier database {source} could not be read for archiving. "
                           "Nothing was changed.") from exc
    return tables


def legacy_folders(instance_root):
    root = Path(instance_root).resolve()
    return [root / layout for layout in LEGACY_LAYOUTS if (root / layout / "daleel.db").is_file()]


def _archived(instance_root):
    index = Path(instance_root) / "archive" / INDEX
    return json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}


def archive_one(instance_root, source):
    """Copy one earlier data folder into instance/archive/ and verify every file."""
    root = Path(instance_root).resolve()
    created = datetime.now(timezone.utc)
    name = source.relative_to(root).as_posix().replace("/", "-")
    target = root / "archive" / f"{name}-{created.strftime('%Y%m%dT%H%M%SZ')}"
    partial = target.with_name(target.name + ".partial")
    partial.mkdir(parents=True)
    try:
        files = {}
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ArchiveError(f"A link was found in {source}; archive it manually. Nothing was changed.")
            if not path.is_file() or path.name in ("daleel.db-journal", "daleel.db-wal", "daleel.db-shm"):
                continue
            relative = path.relative_to(source).as_posix()
            destination = partial / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative == "daleel.db":
                tables = _copy_database(path, destination)
                files[relative] = {"sha256": _sha256(destination), "tables": tables}
            else:
                shutil.copy2(path, destination)
                digest = _sha256(path)
                if _sha256(destination) != digest:
                    raise ArchiveError(f"The copy of {relative} differs from the original. Nothing was changed.")
                files[relative] = {"sha256": digest, "bytes": path.stat().st_size}
        manifest = {"source": str(source), "created_utc": created.isoformat(timespec="seconds"), "files": files}
        (partial / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        (partial / "RESTORE.md").write_text(RESTORE_TEXT.format(source=source, created=manifest["created_utc"]),
                                            encoding="utf-8")
        os.replace(partial, target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target, manifest


def archive_previous_installations(instance_root, output):
    """Archive each earlier data folder once; the new version never opens an earlier database."""
    root = Path(instance_root).resolve()
    index = _archived(root)
    for source in legacy_folders(root):
        key = source.relative_to(root).as_posix()
        if key in index:
            continue
        output.write(f"Archiving the earlier DALEEL data in {source} (the original is not changed)...\n")
        output.flush()
        target, manifest = archive_one(root, source)
        index[key] = {"archive": target.relative_to(root).as_posix(), "files": len(manifest["files"]),
                      "created_utc": manifest["created_utc"],
                      "manifest_sha256": _sha256(target / "MANIFEST.json")}
        (root / "archive" / INDEX).write_text(json.dumps(index, indent=1), encoding="utf-8")
        output.write(f"Verified archive of {len(manifest['files'])} files: {target}\n"
                     f"Restore instructions: {target / 'RESTORE.md'}\n")
        output.flush()
    return index
