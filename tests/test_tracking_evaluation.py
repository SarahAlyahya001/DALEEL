from copy import deepcopy

import pytest

from daleel.pipeline.tracker import Tracker
from eval.tracking_evaluation import (LegacyIoUTracker, candidate_fingerprint, compare,
                                      identity_metrics, load_labels, run_tracker, validate_cache)


def item(box, cls="person", support="Screening-Only"):
    return {
        "target_class": cls,
        "box": list(box),
        "screening_score": 0.7,
        "verification_score": 0.8 if support == "Dual-Model" else None,
        "support_label": support,
    }


def cache():
    return {"version": 1, "frames": [
        {"sampled_step": 0, "frame_index": 0, "video_time_ms": 0,
         "candidates": [item((0, 0, 10, 20)), item((50, 0, 60, 20), "car", "Dual-Model")]},
        {"sampled_step": 1, "frame_index": 15, "video_time_ms": 500,
         "candidates": [item((6, 0, 16, 20))]},
    ]}


def test_candidate_fingerprint_is_stable_and_covers_classes_support_and_boxes():
    original = cache()
    assert candidate_fingerprint(original) == candidate_fingerprint(deepcopy(original))
    changed = deepcopy(original)
    changed["frames"][0]["candidates"][0]["box"][0] = 1
    assert candidate_fingerprint(original) != candidate_fingerprint(changed)


def test_run_tracker_reports_assignments_lengths_and_preserves_inputs():
    source = cache()
    untouched = deepcopy(source)
    assignments, lengths = run_tracker(source)
    assert source == untouched
    assert assignments[(0, 0)] == assignments[(15, 0)]
    assert sorted(lengths.values()) == [1, 2]


def test_run_tracker_resolves_reconciled_track_aliases_without_losing_candidates():
    frames = []
    frames.append({"sampled_step": 0, "frame_index": 0, "video_time_ms": 0,
                   "candidates": [{**item((0, 0, 20, 10)), "screening_score": .4}]})
    for step in range(1, 6):
        frames.append({"sampled_step": step, "frame_index": step * 15,
                       "video_time_ms": step * 500,
                       "candidates": [{**item((2 * step, 0, 20 + 2 * step, 10), "car"),
                                        "screening_score": .6}]})
    assignments, lengths = run_tracker({"version": 1, "frames": frames})
    assert len(set(assignments.values())) == 1
    assert list(lengths.values()) == [6]
    assert len(assignments) == 6


def test_identity_metrics_counts_fragmentation_merges_switches_and_singletons():
    labels = {"version": 1, "labels": [
        {"frame_index": 0, "candidate_index": 0, "identity": "P01"},
        {"frame_index": 15, "candidate_index": 0, "identity": "P01"},
        {"frame_index": 30, "candidate_index": 0, "identity": "P02"},
        {"frame_index": 0, "candidate_index": 1, "identity": "C01", "legitimate_singleton": True},
        {"frame_index": 30, "candidate_index": 1, "identity": None, "ambiguous": True},
    ]}
    assignments = {(0, 0): 1, (15, 0): 2, (30, 0): 2, (0, 1): 3, (30, 1): 4}
    metrics = identity_metrics(assignments, {1: 1, 2: 2, 3: 1, 4: 1}, labels)
    assert metrics["confirmed_fragmentation"] == 1
    assert metrics["incorrect_merge_count"] == 1
    assert metrics["identity_switches"] == 1
    assert metrics["ambiguous_candidates"] == 1
    assert metrics["retained_legitimate_singletons"] == 1


def test_compare_uses_identical_candidates_and_shows_fallback_improvement():
    labels = {"version": 1, "labels": [
        {"frame_index": 0, "candidate_index": 0, "identity": "P01"},
        {"frame_index": 15, "candidate_index": 0, "identity": "P01"},
        {"frame_index": 0, "candidate_index": 1, "identity": "C01", "legitimate_singleton": True},
    ]}
    result = compare(cache(), labels, LegacyIoUTracker(), Tracker())
    assert result["input_detection_count"] == 3
    assert result["input_classes"] == {"car": 1, "person": 2}
    assert result["input_support_labels"] == {"Dual-Model": 1, "Screening-Only": 2}
    assert result["before"]["confirmed_fragmentation"] == 1
    assert result["after"]["confirmed_fragmentation"] == 0
    assert result["after"]["incorrect_merge_count"] == 0
    assert result["after"]["retained_legitimate_singletons"] == 1


def test_validation_rejects_bad_cache_duplicate_labels_and_missing_candidates():
    bad = cache()
    bad["frames"][1]["frame_index"] = 0
    with pytest.raises(ValueError, match="increasing"):
        validate_cache(bad)
    duplicate = {"version": 1, "labels": [
        {"frame_index": 0, "candidate_index": 0, "identity": "P01"},
        {"frame_index": 0, "candidate_index": 0, "identity": "P02"},
    ]}
    with pytest.raises(ValueError, match="two manual labels"):
        load_labels(duplicate)
    with pytest.raises(ValueError, match="absent"):
        identity_metrics({}, {}, {"version": 1, "labels": [
            {"frame_index": 0, "candidate_index": 0, "identity": "P01"},
        ]})
