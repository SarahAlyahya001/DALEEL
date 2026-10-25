"""Original display names and recoverable removal of a single owned mission."""
from contextlib import contextmanager
import json
from pathlib import Path
import unicodedata
import uuid

from .mission_media import EvidenceUnavailable, _relative
from .models import Mission, db


SUBMISSION_NAMES = "submission.json"


def display_basename(value):
    if not isinstance(value, str):
        return ""
    name = unicodedata.normalize("NFC", value.replace("\\", "/").rsplit("/", 1)[-1])
    return "".join(char for char in name if not unicodedata.category(char).startswith("C"))[:512].strip()


def mission_folder(app, mission):
    instance = Path(app.instance_path).resolve()
    root = Path(app.config["UPLOAD_DIR"]).resolve()
    if not root.is_relative_to(instance):
        raise ValueError("Mission storage is outside the application instance.")
    try:
        video_path = instance / _relative(mission.video_filename)
        srt_path = instance / _relative(mission.srt_filename)
    except EvidenceUnavailable:
        raise ValueError("Invalid mission storage.") from None
    folder = video_path.parent
    # Only a direct mission child; reject symlinks and Windows junction escapes.
    if (folder.is_symlink() or folder.resolve() != folder.absolute() or folder.resolve().parent != root
            or video_path.resolve().parent != folder.resolve()
            or srt_path.resolve().parent != folder.resolve()):
        raise ValueError("Mission storage must be one contained folder.")
    return folder.resolve()


def original_names(app, mission):
    """Uploaded file names for display; falls back to the stored names."""
    names = {"video_name": display_basename(mission.video_filename),
             "srt_name": display_basename(mission.srt_filename)}
    try:
        folder = mission_folder(app, mission)
        path = folder / SUBMISSION_NAMES
        if path.is_symlink() or path.resolve().parent != folder or path.stat().st_size > 8192:
            return names
        with path.open("r", encoding="utf-8") as source:
            saved = json.load(source)
        if isinstance(saved, dict):
            for key in names:
                names[key] = display_basename(saved.get(key)) or names[key]
        return names
    except (OSError, ValueError, RuntimeError):
        return names


def original_video_name(app, mission):
    return original_names(app, mission)["video_name"]


def save_submission_names(folder, video_name, srt_name):
    with (Path(folder) / SUBMISSION_NAMES).open("x", encoding="utf-8") as target:
        json.dump({"video_name": display_basename(video_name),
                   "srt_name": display_basename(srt_name)}, target, ensure_ascii=False)


def _archive_fields(record):
    values = {}
    for column in record.__table__.columns:
        value = getattr(record, column.name)
        values[column.name] = value.isoformat() if hasattr(value, "isoformat") else value
    return values


@contextmanager
def archive_mission(app, mission):
    """Move only this mission to local trash; restore its files on rollback.

    The caller deletes the mission/observations and commits inside this context.
    No recursive filesystem deletion and no database/schema reset are involved.
    """
    folder = mission_folder(app, mission)
    instance = Path(app.instance_path).resolve()
    # Never move media also referenced by another mission, regardless of owner.
    for other in db.session.query(Mission).filter(Mission.mission_id != mission.mission_id):
        for value in (other.video_filename, other.srt_filename):
            try:
                path = (instance / _relative(value)).resolve()
            except (EvidenceUnavailable, OSError, ValueError):
                continue
            if path.is_relative_to(folder):
                raise ValueError("Mission media is shared with another mission.")
    trash = instance / "trash"
    if (trash.is_symlink() or trash.resolve() != trash.absolute()
            or trash.resolve().parent != instance):
        raise ValueError("Unsafe trash directory.")
    trash.mkdir(exist_ok=True)
    archive = trash / (f"mission-{mission.mission_id}-" + uuid.uuid4().hex)
    archive.mkdir()
    backup = archive / "mission.json"
    moved = False
    try:
        with backup.open("x", encoding="utf-8") as target:
            json.dump({"mission": _archive_fields(mission),
                       "observations": [_archive_fields(item) for item in mission.observations]},
                      target, ensure_ascii=False, allow_nan=False)
        if folder.exists():
            folder.rename(archive / "media")
            moved = True
        yield archive
    except Exception:
        if moved:
            (archive / "media").rename(folder)
        if backup.exists():
            backup.unlink()
        archive.rmdir()
        raise
