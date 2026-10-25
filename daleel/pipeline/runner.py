"""Single-process, one-slot mission processing with an explicit runtime lifecycle."""

from concurrent.futures import ThreadPoolExecutor
import math
import re
from pathlib import Path
import shutil
import threading
import time
import uuid

import cv2
from sqlalchemy import select

from ..models import Mission, Observation, db
from .detector import DualDetector, local_weights
from .evidence import write_representative
from .srt_parser import coord_at, parse_srt
from .tracker import Tracker
from .weights import WEIGHT_SPECS


class ProcessingUnavailable(RuntimeError):
    """No job was admitted; no mission should be committed."""


class TimingError(ValueError):
    pass


def _finite(value):
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


class VideoClock:
    """Decoder time first; validated FPS fallback without changing the schema."""
    def __init__(self, fps):
        self.fps = fps
        self.previous_decoder = None
        self.sources = set()

    def timestamp(self, source_index, decoder_ms):
        usable = (_finite(decoder_ms) and decoder_ms >= 0
                  and (source_index == 0 or decoder_ms > 0)
                  and (self.previous_decoder is None or decoder_ms >= self.previous_decoder))
        if usable:
            self.previous_decoder = decoder_ms
            self.sources.add("decoder")
            return decoder_ms
        if _finite(self.fps) and self.fps > 0:
            self.sources.add("FPS-derived")
            return source_index / self.fps * 1000
        raise TimingError("Video timing failed: neither decoder timestamps nor positive finite FPS is usable.")


def _mission_folder(app, mission):
    instance = Path(app.instance_path).resolve()
    video = (instance / mission.video_filename).resolve()
    srt = (instance / mission.srt_filename).resolve()
    uploads = Path(app.config["UPLOAD_DIR"]).resolve()
    if (not video.is_relative_to(uploads) or not srt.is_relative_to(uploads)
            or video.parent != srt.parent or video.parent == uploads):
        raise ValueError("Mission input paths must share a contained mission folder.")
    return video.parent, video, srt


def _remove_owned_folder(path, parent):
    """Only remove a concrete, caller-owned direct child; never a storage root."""
    path, parent = Path(path), Path(parent).resolve()
    if path.is_symlink() or path.resolve().parent != parent:
        raise ValueError("Unsafe cleanup path refused.")
    if path.exists():
        shutil.rmtree(path)


def _observation(track, points):
    member, item = track.representative, track.representative.candidate
    lat, lon = coord_at(points, member.video_time_ms / 1000)
    return dict(
        video_time_ms=member.video_time_ms,
        first_frame_index=track.first_frame_index,
        last_frame_index=track.last_frame_index,
        frame_count=track.frame_count,
        frame_path=f"frames/track-{track.track_id}.jpg",
        target_class=track.target_class, agreement=item.agreement,
        yolov8n_score=item.yolov8n_score, yolov8s_score=item.yolov8s_score,
        latitude=lat, longitude=lon, review_status="Unreviewed",
    )


def _process(app, mission, stats):
    folder, video, srt = _mission_folder(app, mission)
    work = folder / ".processing"
    work.mkdir()  # Refuse to overwrite evidence from another attempt.
    points = parse_srt(str(srt))
    detector = DualDetector(app.config["WEIGHTS_N"], app.config["WEIGHTS_S"],
                            app.config["DETECTION_CONF"], app.config["INFERENCE_IMGSZ"], WEIGHT_SPECS)
    tracker = Tracker(app.config["TRACK_IOU"], app.config["TRACK_MAX_GAP"])
    observations = []  # Compact scalar/path dictionaries, never image arrays.
    capture = cv2.VideoCapture()
    clock = None
    def counted(model):
        stats[f"{model}_frames"] += 1
    try:
        if not capture.open(str(video)) or not capture.isOpened():
            raise ValueError("The mission video could not be opened.")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
        except cv2.error:
            fps = None
        clock = VideoClock(fps)
        reported = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        source_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise ValueError("The mission video contains an unreadable frame.")
            try:
                decoder_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            except cv2.error:
                decoder_ms = None
            timestamp = clock.timestamp(source_index, decoder_ms)
            if source_index % 15 == 0:
                result = detector.detect(frame, on_call=counted)
                expired, changed = tracker.update(source_index, timestamp, result.candidates)
                for track in changed:
                    write_representative(frame, track, work)
                observations.extend(_observation(track, points) for track in expired)
            source_index += 1
        if source_index == 0 or (math.isfinite(reported) and reported > 0 and source_index < int(reported)):
            raise ValueError("The mission video ended before all expected frames decoded.")
        observations.extend(_observation(track, points) for track in tracker.finalize())
        return observations, work
    finally:
        capture.release()
        if clock is not None:
            app.logger.info("Stage 3 timestamp sources used: %s", ", ".join(sorted(clock.sources)) or "none")


def _run(app, mission_id, stats):
    """Worker-only entry: one terminal transaction, or rollback and Failed."""
    with app.app_context():
        work = final = None
        try:
            mission = db.session.get(Mission, mission_id)
            if mission is None or mission.status != "Processing":
                raise ValueError("An accepted Processing mission is required.")
            folder, _, _ = _mission_folder(app, mission)
            work, final = folder / ".processing", folder / "frames"
            # New submission folders must not already contain generated evidence.
            if work.exists() or final.exists():
                work = final = None  # Never delete pre-existing files.
                raise ValueError("Mission evidence already exists.")
            rows, _ = _process(app, mission, stats)
            work.rename(final)
            for row in rows:
                db.session.add(Observation(mission_id=mission_id, **row))
            mission.yolov8n_frames = stats["yolov8n_frames"]
            mission.yolov8s_frames = stats["yolov8s_frames"]
            mission.elapsed_seconds = time.perf_counter() - stats["started"]
            mission.status = "Complete"
            db.session.commit()  # Observations/counters/Complete become visible together.
        except Exception as exc:
            db.session.rollback()
            for generated in (work, final):
                if generated is not None:
                    try:
                        _remove_owned_folder(generated, generated.parent)
                    except Exception:
                        app.logger.error("Stage 3 generated-evidence cleanup failed.")
            try:
                mission = db.session.get(Mission, mission_id)
                if mission is not None and mission.status == "Processing":
                    mission.status = "Failed"
                    mission.yolov8n_frames = stats["yolov8n_frames"]
                    mission.yolov8s_frames = stats["yolov8s_frames"]
                    mission.elapsed_seconds = time.perf_counter() - stats["started"]
                    db.session.commit()
                app.logger.error("Stage 3 analysis failed (%s); original mission inputs retained.", type(exc).__name__)
            except Exception:
                db.session.rollback()
                app.extensions["daleel_processor"].accepting = False
                app.logger.error("Stage 3 failure status could not be persisted; processing disabled until restart.")
        finally:
            db.session.remove()


class Processor:
    """One local process, one admitted job, no queue and no OS locking."""
    def __init__(self, app):
        self.app = app
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="daleel")
        self.lifecycle_lock = threading.Lock()
        self.slot = threading.BoundedSemaphore(1)
        self.accepting = True
        self.future = None
        self.mission_id = None
        self.stats = None

    def _job(self, gate, state):
        try:
            gate.wait()  # Job accepted before commit, analysis only after commit.
            if state["committed"]:
                self.mission_id = state["mission_id"]
                self.stats = dict(yolov8n_frames=0, yolov8s_frames=0, started=time.perf_counter())
                _run(self.app, self.mission_id, self.stats)
        finally:
            self.mission_id = self.stats = None
            self.slot.release()

    def submit(self, analyst_id, staging):
        if not self.accepting or not self.slot.acquire(blocking=False):
            raise ProcessingUnavailable("Processing is unavailable or busy. No mission was created.")
        folder, future = None, None
        gate, state = threading.Event(), {"committed": False}
        try:
            try:
                local_weights(self.app.config["WEIGHTS_N"], self.app.config["WEIGHTS_S"])
            except ValueError as exc:
                raise ProcessingUnavailable(str(exc) + " No mission was created.") from exc
            uploads = Path(self.app.config["UPLOAD_DIR"]).resolve()
            if not uploads.is_relative_to(Path(self.app.instance_path).resolve()):
                raise ValueError("Mission storage must remain inside the application instance.")
            target = uploads / ("mission-" + uuid.uuid4().hex)
            target.mkdir()  # Exclusive ownership before any input transfer.
            folder = target
            for name in ("video.mp4", "telemetry.srt"):
                shutil.copyfile(Path(staging) / name, folder / name)
            names = Path(staging) / "submission.json"
            if names.is_file():
                shutil.copyfile(names, folder / names.name)
            mission = Mission(
                analyst_id=analyst_id,
                video_filename=(folder / "video.mp4").relative_to(self.app.instance_path).as_posix(),
                srt_filename=(folder / "telemetry.srt").relative_to(self.app.instance_path).as_posix(),
                status="Processing",
            )
            db.session.add(mission)
            db.session.flush()
            mission_id = mission.mission_id
            state["mission_id"] = mission_id
            with self.lifecycle_lock:
                if not self.accepting:
                    raise ProcessingUnavailable("Processing is shutting down. No mission was created.")
                future = self.executor.submit(self._job, gate, state)
            self.future = future
            db.session.commit()
            state["committed"] = True
            return mission_id
        except Exception:
            db.session.rollback()
            if folder is not None:
                _remove_owned_folder(folder, folder.parent)
            raise
        finally:
            gate.set()
            if future is None:
                self.slot.release()

    def snapshot(self, mission_id):
        stats = self.stats
        if self.mission_id == mission_id and stats is not None:
            return dict(yolov8n_frames=stats["yolov8n_frames"], yolov8s_frames=stats["yolov8s_frames"],
                        elapsed_seconds=time.perf_counter() - stats["started"])
        return {}

    def shutdown(self):
        with self.lifecycle_lock:
            self.accepting = False
        self.executor.shutdown(wait=True)


def initialize_processor(app):
    """Explicit single-process startup; factory construction never calls this."""
    if "daleel_processor" in app.extensions:
        return app.extensions["daleel_processor"]
    if (app.config["SAMPLE_EVERY_N"] != 15 or app.config["TRACK_IOU"] != 0.30
            or app.config["TRACK_MAX_GAP"] != 3):
        raise ValueError("Stage 3 requires sampling=15, track IoU=0.30 and three missed sampled steps.")
    with app.app_context():
        # Called before the server accepts requests, not by polling or app creation.
        for mission in db.session.scalars(select(Mission).where(Mission.status == "Processing")):
            # Clean only generated files in this workflow's accepted folder layout.
            # Never touch originals or evidence in an older/unrecognized layout.
            try:
                folder, _, _ = _mission_folder(app, mission)
            except ValueError:
                folder = None  # Older records still fail, without touching their files.
            if folder is not None and re.fullmatch(r"mission-[0-9a-f]{32}", folder.name) and not mission.observations:
                for name in (".processing", "frames"):
                    _remove_owned_folder(folder / name, folder)
            mission.status = "Failed"
        db.session.commit()
    processor = Processor(app)
    app.extensions["daleel_processor"] = processor
    return processor


def start_processing(app, analyst_id, staging):
    processor = app.extensions.get("daleel_processor")
    if processor is None:
        raise ProcessingUnavailable("Processing is not initialized. No mission was created.")
    return processor.submit(analyst_id, staging)
