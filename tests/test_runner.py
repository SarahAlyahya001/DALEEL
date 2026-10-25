import io
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from sqlalchemy import select

from daleel import create_app, initialize_database
from daleel.models import Mission, Observation, User, db
from daleel.pipeline import runner
from daleel.pipeline.detector import Candidate, FrameDetections


SRT = b"1\n00:00:00,000 --> 00:00:09,000\nGPS(46,24)\n"


class Capture:
    def __init__(self, count=61, fail_at=None, fps=30, timestamp=None):
        self.count, self.index, self.fail_at = count, -1, fail_at
        self.fps, self.timestamp_source, self.released = fps, timestamp, False
    def open(self, path):
        return True
    def isOpened(self):
        return True
    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.count
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.index * 1000 / 30 if self.timestamp_source is None else self.timestamp_source
        return 0
    def read(self):
        self.index += 1
        if self.index == self.fail_at:
            raise cv2.error("synthetic decode failure")
        if self.index >= self.count:
            return False, None
        return True, np.full((24, 24, 3), self.index % 255, dtype=np.uint8)
    def release(self):
        self.released = True


@pytest.fixture
def setup(tmp_path, monkeypatch):
    app = create_app(str(tmp_path / "instance"), {"TESTING": True})
    for key in ("WEIGHTS_N", "WEIGHTS_S"):
        weight = tmp_path / (key + ".pt")
        weight.write_bytes(b"mock weight")
        app.config[key] = str(weight)
    with app.app_context():
        initialize_database()
        user = User(username="analyst", role="analyst")
        user.set_password("test-password")
        db.session.add(user)
        db.session.commit()
        user_id = user.user_id
    capture = Capture()
    monkeypatch.setattr(runner.cv2, "VideoCapture", lambda: capture)
    seen = []
    class Detector:
        def __init__(self, *args):
            pass
        def detect(self, frame, on_call):
            on_call("yolov8n")
            seen.append(int(frame[0, 0, 0]))
            on_call("yolov8s")
            return FrameDetections([Candidate("person", (1, 2, 12, 15), .6, .8)])
    monkeypatch.setattr(runner, "DualDetector", Detector)
    processor = runner.initialize_processor(app)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "video.mp4").write_bytes(b"original video")
    (staging / "telemetry.srt").write_bytes(SRT)
    yield SimpleNamespace(app=app, processor=processor, user_id=user_id, staging=staging,
                          capture=capture, seen=seen, tmp_path=tmp_path)
    processor.shutdown()
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def submit(env):
    with env.app.app_context():
        return runner.start_processing(env.app, env.user_id, env.staging)


def wait(env):
    env.processor.future.result(timeout=10)


def assert_originals(env, mission):
    root = Path(env.app.instance_path)
    assert (root / mission.video_filename).read_bytes() == b"original video"
    assert (root / mission.srt_filename).read_bytes() == SRT


def test_admission_retains_original_filename_metadata(setup):
    from daleel.mission_storage import original_video_name, save_submission_names
    save_submission_names(setup.staging, "Saudi desert rescue.mp4", "flight coordinates.srt")
    mission_id = submit(setup)
    wait(setup)
    with setup.app.app_context():
        mission = db.session.get(Mission, mission_id)
        assert mission.status == "Complete"
        assert mission.video_filename.endswith("/video.mp4")
        assert original_video_name(setup.app, mission) == "Saudi desert rescue.mp4"
        assert_originals(setup, mission)


def test_connected_streaming_pipeline_persists_one_consolidated_observation(setup, caplog):
    env = setup
    caplog.set_level("INFO")
    mission_id = submit(env)
    wait(env)
    assert env.seen == [0, 15, 30, 45, 60]
    assert env.capture.released
    with env.app.app_context():
        mission = db.session.get(Mission, mission_id)
        assert mission.status == "Complete"
        assert mission.yolov8n_frames == mission.yolov8s_frames == 5
        assert mission.elapsed_seconds >= 0
        observation = mission.observations[0]
        assert len(mission.observations) == 1
        assert (observation.first_frame_index, observation.last_frame_index, observation.frame_count) == (0, 60, 5)
        assert observation.video_time_ms == 0
        assert observation.target_class == "person"
        assert observation.agreement == "Both models"
        assert observation.yolov8n_score == .6 and observation.yolov8s_score == .8
        assert (observation.latitude, observation.longitude, observation.review_status) == (24, 46, "Unreviewed")
        folder = Path(env.app.instance_path) / Path(mission.video_filename).parent
        assert (folder / observation.frame_path).is_file()
        assert not (folder / ".processing").exists()
        assert_originals(env, mission)
    assert "timestamp sources used: decoder" in caplog.text


def test_no_candidates_completes_with_zero_observations(setup, monkeypatch):
    class Empty:
        def __init__(self, *args):
            pass
        def detect(self, frame, on_call):
            on_call("yolov8n")
            on_call("yolov8s")
            return FrameDetections([])
    monkeypatch.setattr(runner, "DualDetector", Empty)
    mission_id = submit(setup)
    wait(setup)
    with setup.app.app_context():
        mission = db.session.get(Mission, mission_id)
        assert (mission.status, mission.yolov8n_frames, mission.yolov8s_frames) == ("Complete", 5, 5)
        assert mission.observations == []


@pytest.mark.parametrize("failure", ["decode", "model", "evidence", "timing", "commit"])
def test_analysis_failure_rolls_back_observations_preserves_inputs_and_counts(setup, monkeypatch, failure):
    env = setup
    if failure == "decode":
        env.capture.fail_at = 16
    elif failure == "model":
        class Broken:
            def __init__(self, *args):
                pass
            def detect(self, frame, on_call):
                on_call("yolov8n")
                raise RuntimeError("synthetic inference failure")
        monkeypatch.setattr(runner, "DualDetector", Broken)
    elif failure == "evidence":
        monkeypatch.setattr(runner, "write_representative", lambda *args: (_ for _ in ()).throw(OSError("synthetic write failure")))
    elif failure == "timing":
        env.capture.fps, env.capture.timestamp_source = 0, float("nan")
    else:
        original_commit = db.session.commit
        def commit():
            if any(isinstance(item, Observation) for item in db.session.new):
                db.session.flush()  # Exercise rollback even after inserts reached SQLite.
                raise RuntimeError("synthetic terminal commit failure")
            return original_commit()
        monkeypatch.setattr(db.session, "commit", commit)
    mission_id = submit(env)
    wait(env)
    with env.app.app_context():
        mission = db.session.get(Mission, mission_id)
        assert mission.status == "Failed"
        assert mission.observations == []
        assert mission.elapsed_seconds >= 0
        expected = {"decode": (2, 2), "model": (1, 0), "evidence": (1, 1),
                    "timing": (0, 0), "commit": (5, 5)}[failure]
        assert (mission.yolov8n_frames, mission.yolov8s_frames) == expected
        assert_originals(env, mission)
        folder = Path(env.app.instance_path) / Path(mission.video_filename).parent
        assert sorted(p.name for p in folder.iterdir()) == ["telemetry.srt", "video.mp4"]
    assert env.capture.released


@pytest.mark.parametrize("failure", ["weights", "transfer", "flush", "dispatch", "commit"])
def test_precommit_failures_leave_no_mission_or_permanent_folder(setup, monkeypatch, failure):
    env = setup
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic admission failure")
    if failure == "weights":
        env.app.config["WEIGHTS_S"] = str(env.tmp_path / "missing.pt")
    elif failure == "transfer":
        monkeypatch.setattr(runner.shutil, "copyfile", fail)
    elif failure == "flush":
        monkeypatch.setattr(db.session, "flush", fail)
    elif failure == "dispatch":
        monkeypatch.setattr(env.processor.executor, "submit", fail)
    else:
        monkeypatch.setattr(db.session, "commit", fail)
    with pytest.raises((RuntimeError, runner.ProcessingUnavailable)):
        submit(env)
    if env.processor.future is not None:
        wait(env)
    with env.app.app_context():
        assert db.session.scalar(select(db.func.count()).select_from(Mission)) == 0
    assert list(Path(env.app.config["UPLOAD_DIR"]).iterdir()) == []
    assert (env.staging / "video.mp4").read_bytes() == b"original video"
    assert env.processor.slot.acquire(blocking=False)
    env.processor.slot.release()


def test_busy_worker_rejects_second_mission_without_queue_or_folder(setup, monkeypatch):
    env = setup
    entered, release = threading.Event(), threading.Event()
    original = runner._process
    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)
    monkeypatch.setattr(runner, "_process", blocked)
    mission_id = submit(env)
    try:
        assert entered.wait(5)
        with pytest.raises(runner.ProcessingUnavailable, match="busy"):
            submit(env)
        with env.app.app_context():
            assert Mission.query.count() == 1
            assert db.session.get(Mission, mission_id).status == "Processing"
            assert Observation.query.count() == 0
        assert len(list(Path(env.app.config["UPLOAD_DIR"]).iterdir())) == 1
    finally:
        release.set()
        wait(env)


def test_explicit_recovery_not_factory_marks_interrupted_missions_failed(setup):
    env = setup
    env.processor.shutdown()
    with env.app.app_context():
        mission = Mission(analyst_id=env.user_id, video_filename="uploads/old/video.mp4",
                          srt_filename="uploads/old/t.srt", status="Processing", yolov8n_frames=7)
        complete = Mission(analyst_id=env.user_id, video_filename="uploads/done/video.mp4",
                           srt_filename="uploads/done/t.srt", status="Complete")
        db.session.add_all([mission, complete])
        db.session.commit()
        ids = mission.mission_id, complete.mission_id
    second = create_app(env.app.instance_path, {"TESTING": True})
    with second.app_context():
        assert db.session.get(Mission, ids[0]).status == "Processing"
    processor = runner.initialize_processor(second)
    try:
        assert runner.initialize_processor(second) is processor
        with second.app_context():
            mission = db.session.get(Mission, ids[0])
            assert mission.status == "Failed" and mission.yolov8n_frames == 7
            assert db.session.get(Mission, ids[1]).status == "Complete"
    finally:
        processor.shutdown()
        with second.app_context():
            db.engine.dispose()


def test_shutdown_rejects_admission_without_records(setup):
    setup.processor.shutdown()
    with pytest.raises(runner.ProcessingUnavailable):
        submit(setup)
    with setup.app.app_context():
        assert Mission.query.count() == 0


def test_recovery_removes_only_incomplete_generated_evidence(setup):
    env = setup
    env.processor.shutdown()
    folder = Path(env.app.config["UPLOAD_DIR"]) / ("mission-" + "a" * 32)
    folder.mkdir()
    (folder / "video.mp4").write_bytes(b"preserve video")
    (folder / "telemetry.srt").write_bytes(SRT)
    (folder / ".processing").mkdir()
    (folder / ".processing" / "partial.jpg").write_bytes(b"partial generated image")
    with env.app.app_context():
        mission = Mission(analyst_id=env.user_id,
            video_filename=(folder / "video.mp4").relative_to(env.app.instance_path).as_posix(),
            srt_filename=(folder / "telemetry.srt").relative_to(env.app.instance_path).as_posix())
        db.session.add(mission)
        db.session.commit()
    second = create_app(env.app.instance_path, {"TESTING": True})
    processor = runner.initialize_processor(second)
    try:
        assert (folder / "video.mp4").read_bytes() == b"preserve video"
        assert (folder / "telemetry.srt").read_bytes() == SRT
        assert not (folder / ".processing").exists()
        with second.app_context():
            assert Mission.query.one().status == "Failed"
    finally:
        processor.shutdown()
        with second.app_context():
            db.engine.dispose()


def test_cleanup_refuses_parent_root_and_preserves_unrelated_files(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    sentinel = child / "keep.txt"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="Unsafe cleanup"):
        runner._remove_owned_folder(tmp_path, tmp_path)
    assert sentinel.read_bytes() == b"preserve"


def test_fps_source_logged(setup, caplog):
    caplog.set_level("INFO")
    setup.capture.timestamp_source = 0
    submit(setup)
    wait(setup)
    assert "FPS-derived" in caplog.text


def test_failure_status_storage_outage_disables_new_work(setup, monkeypatch, caplog):
    original_commit = db.session.commit
    def commit():
        if any(isinstance(item, Mission) and item.status in {"Complete", "Failed"}
               for item in db.session.dirty):
            raise RuntimeError("synthetic database outage")
        return original_commit()
    monkeypatch.setattr(db.session, "commit", commit)
    mission_id = submit(setup)
    wait(setup)
    assert not setup.processor.accepting
    assert "could not be persisted" in caplog.text
    with setup.app.app_context():
        mission = db.session.get(Mission, mission_id)
        assert mission.status != "Complete"
        assert mission.observations == []
        assert_originals(setup, mission)
    with pytest.raises(runner.ProcessingUnavailable):
        submit(setup)


def test_flask_busy_and_missing_weights_reject_without_retained_uploads(setup, monkeypatch):
    env = setup
    monkeypatch.setattr("daleel.submission.validate_mp4", lambda path: None)
    client = env.app.test_client()
    with client.session_transaction() as session:
        session["user_id"], session["_csrf_token"] = env.user_id, "token"
    def post():
        return client.post("/upload", data={"_csrf_token": "token",
            "video": (io.BytesIO(b"original video"), "v.mp4"),
            "srt": (io.BytesIO(SRT), "t.srt")})
    assert env.processor.slot.acquire(blocking=False)
    try:
        response = post()
        assert response.status_code == 503 and b"busy" in response.data
    finally:
        env.processor.slot.release()
    env.app.config["WEIGHTS_N"] = None
    response = post()
    assert response.status_code == 503 and b"weight files are required" in response.data
    with env.app.app_context():
        assert Mission.query.count() == Observation.query.count() == 0
    assert list(Path(env.app.config["UPLOAD_DIR"]).iterdir()) == []


def test_worker_cannot_enter_analysis_before_admission_commit(setup, monkeypatch):
    env = setup
    entered = threading.Event()
    original_process, original_commit = runner._process, db.session.commit
    committed = threading.Event()
    def process(*args):
        assert committed.is_set()
        entered.set()
        return original_process(*args)
    def commit():
        is_admission = any(isinstance(item, Mission) and item.status == "Processing"
                           for item in db.session.identity_map.values())
        if is_admission:
            assert not entered.is_set()
        result = original_commit()
        if is_admission:
            committed.set()
        return result
    monkeypatch.setattr(runner, "_process", process)
    monkeypatch.setattr(db.session, "commit", commit)
    submit(env)
    wait(env)
    assert entered.is_set()


def test_run_entry_uses_one_process_without_reloader():
    # A reloader would start a second process with a second processor.
    source = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8")
    assert "from waitress import serve" in source and "app.run(" not in source
    assert "processor.shutdown()" in source


def test_flask_upload_status_and_stage4_results(setup, monkeypatch):
    env = setup
    monkeypatch.setattr("daleel.submission.validate_mp4", lambda path: None)
    client = env.app.test_client()
    with client.session_transaction() as session:
        session["user_id"], session["_csrf_token"] = env.user_id, "synthetic-token"
    response = client.post("/upload", data={
        "_csrf_token": "synthetic-token",
        "video": (io.BytesIO(b"original video"), "flight.mp4"),
        "srt": (io.BytesIO(SRT), "unrelated.srt"),
    })
    assert response.status_code == 303
    wait(env)
    mission_id = int(response.location.rsplit("/", 1)[1])
    page = client.get(response.location)
    assert page.status_code == 200 and b"Complete" in page.data
    assert b"View results" in page.data
    state = client.get(f"/api/status/{mission_id}")
    assert state.json["status"] == "Complete"
    assert state.json["yolov8n_frames"] == 5
    assert "progress_pct" not in state.json
    for suffix in ("results",):
        assert client.get(f"/{suffix}/{mission_id}").status_code == 200
    assert client.get(f"/export/{mission_id}/pdf").status_code == 200
    assert client.get(f"/export/{mission_id}/gpx").status_code == 200
    with env.app.app_context():
        observation = Observation.query.one()
        assert client.get(f"/frame/{observation.observation_id}").status_code == 200
    folders = list(Path(env.app.config["UPLOAD_DIR"]).iterdir())
    assert len(folders) == 1 and folders[0].name.startswith("mission-")
