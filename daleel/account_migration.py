"""Explicit, verified-backup single-column forward migration."""
from datetime import datetime
import hashlib
import os
from pathlib import Path
import secrets
import sqlite3

from sqlalchemy import inspect

from .models import Mission, Observation, User, db


def _snapshot(connection, table, columns=None):
    selected = "*" if columns is None else ", ".join('"' + name + '"' for name in columns)
    rows = connection.execute(f'SELECT {selected} FROM "{table}" ORDER BY 1').fetchall()
    return hashlib.sha256(repr(rows).encode("utf-8")).digest()


def _verified_backup(database):
    directory = database.parent / "backups"
    directory.mkdir(exist_ok=True)
    backup = directory / ("accounts-" + datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                          + "-" + secrets.token_hex(4) + ".sqlite3")
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(backup) as destination:
            source.backup(destination)
            if destination.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError("Recovery backup integrity verification failed; migration refused.")
            if destination.execute("PRAGMA foreign_key_check").fetchall():
                raise RuntimeError("Recovery backup foreign-key verification failed; migration refused.")
            tables = source.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                    "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
            for (table,) in tables:
                if _snapshot(source, table) != _snapshot(destination, table):
                    raise RuntimeError("Recovery backup verification failed; migration refused.")
    return backup



def upgrade_accounts():
    inspector = inspect(db.engine)
    tables = set(inspector.get_table_names())
    if not {"users", "missions", "observations"} <= tables or tables - {
            "users", "missions", "observations", "account_tokens", "account_deliveries"}:
        raise RuntimeError("Unsupported database schema. No changes were made.")
    columns = {column["name"] for column in inspector.get_columns("users")}
    canonical = {column.name for column in User.__table__.columns}
    if columns not in (canonical, canonical - {"must_change_password"}):
        raise RuntimeError("Unsupported account schema. No changes were made.")
    for model in (Mission, Observation):
        if {column["name"] for column in inspector.get_columns(model.__tablename__)} != {
                column.name for column in model.__table__.columns}:
            raise RuntimeError("Unsupported mission schema. No changes were made.")
    if columns == canonical:
        return None
    database = Path(db.engine.url.database).resolve()
    db.session.remove()
    db.engine.dispose()
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        try:
            backup = _verified_backup(database)
            prior_columns = [row[1] for row in connection.execute("PRAGMA table_info(users)")]
            connection.execute("ALTER TABLE users ADD COLUMN must_change_password "
                               "BOOLEAN NOT NULL DEFAULT 0 CHECK (must_change_password IN (0, 1))")
            with sqlite3.connect(backup) as recovery:
                for table in tables:
                    selected = prior_columns if table == "users" else None
                    if _snapshot(connection, table, selected) != _snapshot(recovery, table, selected):
                        raise RuntimeError("Existing data differs; migration rolled back.")
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError("Post-migration integrity check failed; migration rolled back.")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise RuntimeError("Post-migration foreign-key check failed; migration rolled back.")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    db.engine.dispose()
    return backup
