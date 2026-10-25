import sys
from types import SimpleNamespace

import numpy as np
import pytest

from daleel.pipeline.detector import (
    BOTH_MODELS, DETECTION_CONF, ONE_MODEL, Candidate, DualDetector, fuse, iou,
)
from daleel.pipeline.evidence import write_representative
from daleel.pipeline.runner import TimingError, VideoClock
from daleel.pipeline.srt_parser import TelemetryPoint, coord_at, parse_srt
from daleel.pipeline.tracker import Tracker
from daleel.pipeline.weights import WeightSpec, sha256_file


def candidate(score=0.6, second=None, box=(0, 0, 10, 10), cls="person"):
    """A YOLOv8n detection, also reported by YOLOv8s with score `second` when given."""
    return Candidate(cls, box, score, second)


def test_iou_bounds():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_fusion_keeps_every_detection_and_labels_agreement():
    n = [("person", (0, 0, 10, 10), .7), ("car", (20, 20, 30, 30), .65)]
    s = [("person", (1, 1, 11, 11), .8), ("person", (50, 50, 60, 60), .9)]
    result = fuse(n, s)
    assert [(c.target_class, c.agreement) for c in result] == [
        ("person", BOTH_MODELS), ("person", ONE_MODEL), ("car", ONE_MODEL)]
    pair, s_only, n_only = result
    assert pair.box == (1, 1, 11, 11) and pair.score == .8  # box and score of the higher-scoring model
    assert (pair.yolov8n_score, pair.yolov8s_score) == (.7, .8)
    assert (s_only.yolov8n_score, s_only.yolov8s_score, s_only.box) == (None, .9, (50, 50, 60, 60))
    assert (n_only.yolov8n_score, n_only.yolov8s_score) == (.65, None)


def test_fusion_exact_tie_keeps_the_yolov8n_box():
    pair = fuse([("person", (0, 0, 10, 10), .6)], [("person", (1, 0, 11, 10), .6)])[0]
    assert pair.box == (0, 0, 10, 10) and pair.agreement == BOTH_MODELS


def test_fusion_one_to_one_same_class_and_deterministic():
    n = [("person", (0, 0, 10, 10), .7)] * 2
    s = [("person", (0, 0, 10, 10), .8), ("car", (0, 0, 10, 10), .9)]
    result = fuse(n, s)
    assert sorted(c.agreement for c in result) == [BOTH_MODELS, ONE_MODEL, ONE_MODEL]
    assert {c.target_class for c in result if c.agreement == BOTH_MODELS} == {"person"}
    assert fuse(n, s) == fuse(n, s)
    assert [c.agreement for c in fuse([], s)] == [ONE_MODEL, ONE_MODEL]


def test_fusion_pairs_by_highest_overlap_first():
    n = [("person", (0, 0, 10, 10), .6)]
    s = [("person", (2, 0, 12, 10), .5), ("person", (0, 0, 10, 10), .4)]
    result = fuse(n, s)
    agreed = [c for c in result if c.agreement == BOTH_MODELS]
    assert len(agreed) == 1 and agreed[0].yolov8s_score == .4  # identical box (IoU 1.0) wins
    assert sum(c.agreement == ONE_MODEL for c in result) == 1


@pytest.mark.parametrize("width,both", [(5, True), (4.99, False)])
def test_fusion_inclusive_half_iou(width, both):
    result = fuse([("person", (0, 0, 10, 10), .6)], [("person", (0, 0, width, 10), .9)])
    assert (len(result) == 1 and result[0].agreement == BOTH_MODELS) is both


def test_candidate_requires_a_model_score_and_derives_score_and_label():
    with pytest.raises(ValueError, match="at least one model"):
        Candidate("person", (0, 0, 1, 1))
    assert Candidate("car", (0, 0, 1, 1), yolov8s_score=.5).agreement == ONE_MODEL
    assert Candidate("car", (0, 0, 1, 1), .45, .7).score == .7


def model_box(class_id, score=.6, box=(0, 0, 10, 10)):
    return SimpleNamespace(cls=np.array(class_id), conf=np.array(score), xyxy=np.array([box]))


def mock_models(tmp_path, monkeypatch, n_classes, s_classes):
    paths = [tmp_path / "yolov8n_sar_v1.pt", tmp_path / "yolov8s_sar_v1.pt"]
    for path in paths:
        path.write_bytes(b"mock weights")
    calls = []
    names = {i: str(i) for i in range(80)}
    names.update({0: "person", 2: "car"})

    class Model:
        def __init__(self, path):
            self.model = "n" if path.endswith("yolov8n_sar_v1.pt") else "s"
            self.names = names

        def predict(self, frame, **kwargs):
            calls.append((self.model, kwargs))
            classes = n_classes if self.model == "n" else s_classes
            return [SimpleNamespace(boxes=[model_box(c) for c in classes])]
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    return paths, calls


@pytest.mark.parametrize("n_classes,s_classes", [([], []), ([3, 5, 7, 1], [0]), ([0], [0]), ([2, 0, 7], [5]), ([], [2])])
def test_both_models_run_on_every_sampled_frame(tmp_path, monkeypatch, n_classes, s_classes):
    paths, calls = mock_models(tmp_path, monkeypatch, n_classes, s_classes)
    detector = DualDetector(*paths)
    counted = []
    result = detector.detect(np.zeros((20, 20, 3)), on_call=counted.append)
    assert [model for model, _ in calls] == ["n", "s"]  # YOLOv8s runs even when YOLOv8n finds nothing
    assert counted == ["yolov8n", "yolov8s"]
    assert all(c.target_class in {"person", "car"} for c in result.candidates)
    assert all(kwargs["classes"] == [0, 2] and kwargs["device"] == "cpu" and kwargs["conf"] == DETECTION_CONF
               and kwargs["imgsz"] == 1280 for _, kwargs in calls)


def test_detection_found_only_by_yolov8s_is_kept(tmp_path, monkeypatch):
    paths, _ = mock_models(tmp_path, monkeypatch, [], [0])
    [only] = DualDetector(*paths).detect(np.zeros((20, 20, 3))).candidates
    assert (only.target_class, only.agreement, only.yolov8n_score, only.yolov8s_score) == ("person", ONE_MODEL, None, .6)


def test_detector_refuses_files_that_are_not_the_pinned_models(tmp_path, monkeypatch):
    paths, calls = mock_models(tmp_path, monkeypatch, [0], [0])
    good = [WeightSpec(p.name, "", p.stat().st_size, sha256_file(p)) for p in paths]
    DualDetector(*paths, specs=good)
    wrong = [good[0], WeightSpec(good[1].filename, "", good[1].size, "0" * 64)]
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: pytest.fail("No model construction")))
    with pytest.raises(ValueError, match="setup_weights"):
        DualDetector(*paths, specs=wrong)


def test_missing_second_weight_does_not_construct_either_model(tmp_path, monkeypatch):
    existing = tmp_path / "n.pt"
    existing.write_bytes(b"mock")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: pytest.fail("No model construction")))
    with pytest.raises(ValueError, match="Both local"):
        DualDetector(existing, tmp_path / "missing.pt")


@pytest.mark.parametrize("gap,counts", [(3, [2]), (4, [1, 1])])
def test_gap_counts_missed_sampled_frames(gap, counts):
    tracker = Tracker()
    tracker.update(0, 0, [candidate()])
    ended = []
    for step in range(1, gap + 1):
        expired, _ = tracker.update(step * 15, step * 500, [])
        ended.extend(expired)
    expired, _ = tracker.update((gap + 1) * 15, (gap + 1) * 500, [candidate()])
    ended.extend(expired)
    ended.extend(tracker.finalize())
    assert [t.frame_count for t in ended] == counts
    if gap == 3:
        assert ended[0].first_frame_index == 0
        assert ended[0].last_frame_index == 60  # sampled step 4 reconnects with step 0


def test_tracker_singletons_classes_and_compact_metadata():
    tracker = Tracker()
    tracker.update(0, 0, [candidate(), candidate(cls="car")])
    tracks = tracker.finalize()
    assert len(tracks) == 2
    assert all(t.frame_count == 1 and not hasattr(t, "members") for t in tracks)
    assert tracker.active == []
    assert not hasattr(tracker, "finished")


@pytest.mark.parametrize("width,expected", [(3, 1), (2.99, 2)])
def test_tracking_inclusive_point_three_iou(width, expected):
    tracker = Tracker()
    tracker.update(0, 0, [candidate()])
    tracker.update(15, 500, [candidate(box=(0, 0, width, 10))])
    assert len(tracker.finalize()) == expected


def test_representative_prefers_both_models_then_higher_score_then_earliest():
    tracker = Tracker()
    sequence = [candidate(.99), candidate(.5, .7), candidate(.6, .8), candidate(.8, .6), candidate(.98)]
    for step, item in enumerate(sequence):
        tracker.update(step * 15, step * 500, [item])
    track = tracker.finalize()[0]
    assert track.representative.frame_index == 30
    assert track.representative.candidate is sequence[2]  # label and scores of that one detection
    assert track.frame_count == 5
    assert track.first_frame_index == 0 and track.last_frame_index == 60


def test_representative_label_is_never_combined_across_frames():
    """YOLOv8n in one frame and YOLOv8s in the next do not make a Both models observation."""
    tracker = Tracker()
    tracker.update(0, 0, [Candidate("person", (0, 0, 10, 10), .7)])
    tracker.update(15, 500, [Candidate("person", (0, 0, 10, 10), yolov8s_score=.9)])
    [track] = tracker.finalize()
    assert track.frame_count == 2
    assert track.representative.candidate.agreement == ONE_MODEL
    assert (track.representative.candidate.yolov8n_score, track.representative.candidate.yolov8s_score) == (None, .9)


def test_tracker_never_associates_two_same_frame_candidates_to_one_track():
    tracker = Tracker()
    tracker.update(0, 0, [candidate()])
    tracker.update(15, 500, [candidate(), candidate()])
    assert sorted(t.frame_count for t in tracker.finalize()) == [1, 2]


def test_tracking_fallback_recovers_small_displaced_box_below_iou_threshold():
    tracker = Tracker()
    tracker.update(0, 0, [candidate(box=(0, 0, 10, 20))])
    tracker.update(15, 500, [candidate(box=(6, 0, 16, 20))])
    assert [track.frame_count for track in tracker.finalize()] == [2]


def test_tracking_fallback_predicts_constant_velocity_in_sampled_steps():
    tracker = Tracker()
    tracker.update(0, 0, [candidate(box=(0, 0, 10, 20))])
    tracker.update(15, 500, [candidate(box=(6, 0, 16, 20))])
    tracker.update(30, 1000, [])
    tracker.update(45, 1500, [candidate(box=(18, 0, 28, 20))])
    assert [track.frame_count for track in tracker.finalize()] == [3]


@pytest.mark.parametrize("second", [
    candidate(box=(6, 0, 26, 40)),
    candidate(box=(6, 0, 26, 10)),
    candidate(box=(30, 0, 40, 20)),
    candidate(box=(6, 0, 16, 20), cls="car"),
])
def test_tracking_fallback_rejects_unsafe_size_aspect_distance_and_class(second):
    tracker = Tracker()
    tracker.update(0, 0, [candidate(box=(0, 0, 10, 20))])
    tracker.update(15, 500, [second])
    assert sorted(track.frame_count for track in tracker.finalize()) == [1, 1]


def test_global_assignment_preserves_two_nearby_people_when_candidates_cross_order():
    tracker = Tracker()
    tracker.update(0, 0, [candidate(box=(0, 0, 10, 20)), candidate(box=(14, 0, 24, 20))])
    first_ids = tracker.last_assignments
    tracker.update(15, 500, [candidate(box=(11, 0, 21, 20)), candidate(box=(3, 0, 13, 20))])
    assert tracker.last_assignments == (first_ids[1], first_ids[0])
    assert sorted(track.frame_count for track in tracker.finalize()) == [2, 2]


def test_global_assignment_is_deterministic_for_exact_ties():
    def result():
        tracker = Tracker()
        tracker.update(0, 0, [candidate(), candidate()])
        tracker.update(15, 500, [candidate(), candidate()])
        assignments = tracker.last_assignments
        tracks = [(track.track_id, track.frame_count) for track in tracker.finalize()]
        return assignments, tracks
    assert result() == result()


def _class_flip_result(person_box=(0, 0, 20, 10), car_boxes=None):
    car_boxes = car_boxes or [(2 + 2 * step, 0, 22 + 2 * step, 10) for step in range(5)]
    tracker = Tracker()
    tracker.update(0, 0, [candidate(.40, box=person_box)])
    expired = []
    for step, boxes in enumerate(car_boxes, start=1):
        expired.extend(tracker.update(
            step * 15, step * 500,
            [candidate(.60, box=box, cls="car") for box in (boxes if isinstance(boxes, list) else [boxes])],
        )[0])
    return tracker, expired + tracker.finalize()


def test_short_weak_vehicle_shaped_person_flip_reconciles_into_stable_car():
    tracker, tracks = _class_flip_result()
    assert [(track.target_class, track.frame_count) for track in tracks] == [("car", 6)]
    assert tracker.reconciled_pairs == [(1, 2)]


def test_real_person_beside_car_remains_separate():
    tracker, tracks = _class_flip_result(person_box=(0, 0, 8, 20))
    assert sorted((track.target_class, track.frame_count) for track in tracks) == [
        ("car", 5), ("person", 1),
    ]
    assert tracker.reconciled_pairs == []


def test_ambiguous_cross_class_candidates_are_not_reconciled():
    car_boxes = [[(9 + step, 0, 29 + step, 10), (11 + step, 0, 31 + step, 10)]
                 for step in range(5)]
    tracker, tracks = _class_flip_result(person_box=(10, 0, 30, 10), car_boxes=car_boxes)
    assert sorted(track.frame_count for track in tracks) == [1, 5, 5]
    assert tracker.reconciled_pairs == []


def test_class_flip_reconciliation_is_deterministic():
    def result():
        tracker, tracks = _class_flip_result()
        return tracker.reconciled_pairs, [(item.track_id, item.target_class, item.frame_count) for item in tracks]
    assert result() == result()


def test_tracking_preserves_candidate_metadata_and_representative_selection_with_fallback():
    first = candidate(.7, box=(0, 0, 10, 20))
    supported = candidate(.6, .8, box=(6, 0, 16, 20))
    tracker = Tracker()
    tracker.update(0, 0, [first])
    tracker.update(15, 500, [supported])
    track = tracker.finalize()[0]
    assert track.representative.candidate is supported
    assert track.representative.candidate.box == supported.box
    assert track.representative.candidate.agreement == BOTH_MODELS


@pytest.mark.parametrize("timestamp,latitude", [(0, 10), (.5, 10), (.51, 20), (1, 20), (100, 20)])
def test_nearest_coordinates_ties_duplicates_endpoints(timestamp, latitude):
    points = [TelemetryPoint(0, 10, 40, None), TelemetryPoint(1, 20, 50, None),
              TelemetryPoint(1, 30, 60, None)]
    assert coord_at(points, timestamp)[0] == latitude


def test_nearest_uses_cue_start_and_preserves_file_order(tmp_path):
    path = tmp_path / "t.srt"
    path.write_text("1\n00:00:02,000 --> 00:00:09,000\nGPS(46,24)\n\n"
                    "2\n00:00:02,000 --> 00:00:03,000\nGPS(47,25)\n", encoding="utf-8")
    assert coord_at(parse_srt(path), 8) == (24, 46)


@pytest.mark.parametrize("fps,values,expected,sources", [
    (30, [0, 40, 80], [0, 40, 80], {"decoder"}),
    (25, [0, 0, 0], [0, 40, 80], {"decoder", "FPS-derived"}),
    (25, [0, 100, 50], [0, 100, 80], {"decoder", "FPS-derived"}),
    (25, [float("nan"), float("inf"), -1], [0, 40, 80], {"FPS-derived"}),
    (float("nan"), [0, 40, 80], [0, 40, 80], {"decoder"}),
    (0, [0, 40, 40], [0, 40, 40], {"decoder"}),
    (25, [None, None, None], [0, 40, 80], {"FPS-derived"}),
])
def test_timestamp_policy(fps, values, expected, sources):
    clock = VideoClock(fps)
    assert [clock.timestamp(i, value) for i, value in enumerate(values)] == expected
    assert clock.sources == sources


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_timing_fails_only_when_both_methods_fail(fps):
    clock = VideoClock(fps)
    assert clock.timestamp(0, 0) == 0
    with pytest.raises(TimingError, match="neither"):
        clock.timestamp(1, 0)


def test_evidence_replaces_only_best_image_and_uses_candidate_box(tmp_path, monkeypatch):
    from daleel.pipeline import evidence
    tracker = Tracker()
    tracker.update(0, 0, [candidate(box=(2, 3, 12, 13))])
    track = tracker.active[0]
    boxes = []
    monkeypatch.setattr(evidence.cv2, "rectangle", lambda frame, a, b, *args: boxes.append((a, b)))
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    root = tmp_path / "evidence"
    assert write_representative(frame, track, root) == "frames/track-1.jpg"
    write_representative(frame, track, root)
    assert boxes == [((2, 3), (12, 13))] * 2
    assert len(list(root.iterdir())) == 1
    assert not frame.any()  # Annotation worked on a copy.


def test_evidence_write_failure_is_not_success(tmp_path, monkeypatch):
    from daleel.pipeline import evidence
    tracker = Tracker()
    tracker.update(0, 0, [candidate()])
    monkeypatch.setattr(evidence.cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="could not be written"):
        write_representative(np.zeros((20, 20, 3), dtype=np.uint8), tracker.active[0], tmp_path)
