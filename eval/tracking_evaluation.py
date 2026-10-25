"""Generic, deterministic identity-based evaluation for DALEEL trackers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from daleel.pipeline.detector import Candidate
from daleel.pipeline.detector import iou
from daleel.pipeline.tracker import Member, Track, Tracker


CACHE_VERSION = 1
ANNOTATION_VERSION = 1


@dataclass(frozen=True)
class Label:
    frame_index: int
    candidate_index: int
    identity: str | None
    window: str = "all"
    ambiguous: bool = False
    legitimate_singleton: bool = False


class LegacyIoUTracker:
    """Greedy same-class IoU tracker, used as a comparison baseline in tracking evaluation."""
    def __init__(self, iou_thr=0.30, max_gap_steps=3):
        self.iou_thr, self.max_gap_steps = iou_thr, max_gap_steps
        self.active = []
        self._step = -1
        self._next_id = 1
        self.last_assignments = ()

    def update(self, frame_index, video_time_ms, candidates):
        self._step += 1
        expired, active = [], []
        for track in self.active:
            missed = self._step - track.last_seen_sampled_step - 1
            (expired if missed > self.max_gap_steps else active).append(track)
        self.active = active
        pairs = sorted(
            (-overlap, track.track_id, ci, ti)
            for ti, track in enumerate(active)
            for ci, candidate in enumerate(candidates)
            if track.target_class == candidate.target_class
            and (overlap := iou(track.last_box, candidate.box)) >= self.iou_thr
        )
        matched, claimed = {}, set()
        for _, _, ci, ti in pairs:
            if ci not in matched and ti not in claimed:
                matched[ci] = active[ti]
                claimed.add(ti)
        changed, assignments = [], []
        for ci, candidate in enumerate(candidates):
            member = Member(frame_index, video_time_ms, candidate)
            track = matched.get(ci)
            if track is None:
                track = Track(self._next_id, candidate.target_class, candidate.box,
                              self._step, frame_index, frame_index, 1, member)
                self._next_id += 1
                self.active.append(track)
                changed.append(track)
            else:
                track.last_box = candidate.box
                track.last_seen_sampled_step = self._step
                track.last_frame_index = frame_index
                track.frame_count += 1
                if member.rank > track.representative.rank:
                    track.representative = member
                    changed.append(track)
            assignments.append(track.track_id)
        self.last_assignments = tuple(assignments)
        return expired, changed

    def finalize(self):
        tracks, self.active = self.active, []
        self.last_assignments = ()
        return tracks


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def candidate_fingerprint(cache):
    """Hash immutable tracker inputs, excluding optional descriptive metadata."""
    content = []
    for frame in cache["frames"]:
        content.append({
            "sampled_step": frame["sampled_step"],
            "frame_index": frame["frame_index"],
            "video_time_ms": frame["video_time_ms"],
            "candidates": frame["candidates"],
        })
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_cache(cache):
    if cache.get("version") != CACHE_VERSION or not isinstance(cache.get("frames"), list):
        raise ValueError("Unsupported tracking candidate cache.")
    previous_step = previous_frame = -1
    for frame in cache["frames"]:
        step, frame_index = frame["sampled_step"], frame["frame_index"]
        if step <= previous_step or frame_index <= previous_frame:
            raise ValueError("Cached frames must be in strictly increasing order.")
        previous_step, previous_frame = step, frame_index
        for item in frame["candidates"]:
            if item["target_class"] not in {"person", "car"}:
                raise ValueError("Cached candidate has an unsupported class.")
            if item["support_label"] not in {"Dual-Model", "Screening-Only"}:
                raise ValueError("Cached candidate has an unsupported support label.")
            box = item["box"]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Cached candidate has an invalid box.")
    return cache


def cached_candidate(item):
    """Candidate from a cache of the original pipeline: a Dual-Model candidate was reported by both
    models (YOLOv8n box), a Screening-Only candidate by YOLOv8n alone."""
    dual = item["support_label"] == "Dual-Model"
    return Candidate(item["target_class"], tuple(item["box"]), item["screening_score"],
                     item.get("verification_score") if dual else None)


def run_tracker(cache, tracker=None):
    """Return candidate-to-track assignments and descriptive track lengths."""
    validate_cache(cache)
    tracker = tracker or Tracker()
    assignments = {}
    lengths = Counter()
    completed = []
    for frame in cache["frames"]:
        candidates = [cached_candidate(item) for item in frame["candidates"]]
        expired, _ = tracker.update(frame["frame_index"], frame["video_time_ms"], candidates)
        completed.extend(expired)
        for candidate_index, track_id in enumerate(tracker.last_assignments):
            assignments[(frame["frame_index"], candidate_index)] = track_id
    completed.extend(tracker.finalize())
    aliases = getattr(tracker, "reconciled_track_ids", {})

    def resolved(track_id):
        while track_id in aliases:
            track_id = aliases[track_id]
        return track_id

    assignments = {key: resolved(track_id) for key, track_id in assignments.items()}
    for track in completed:
        lengths[resolved(track.track_id)] = track.frame_count
    return assignments, lengths


def load_labels(annotation):
    if annotation.get("version") != ANNOTATION_VERSION:
        raise ValueError("Unsupported tracking annotation file.")
    labels = []
    seen = set()
    for item in annotation.get("labels", []):
        label = Label(
            item["frame_index"], item["candidate_index"], item.get("identity"),
            str(item.get("window", "all")),
            bool(item.get("ambiguous", False)), bool(item.get("legitimate_singleton", False)),
        )
        key = (label.window, label.frame_index, label.candidate_index)
        if key in seen:
            raise ValueError("A candidate cannot have two manual labels.")
        if not label.ambiguous and not label.identity:
            raise ValueError("A non-ambiguous label requires an identity.")
        seen.add(key)
        labels.append(label)
    return labels


def identity_metrics(assignments, lengths, annotation):
    """Score only certain manual identities; report ambiguity separately."""
    labels = load_labels(annotation)
    certain = [label for label in labels if not label.ambiguous]
    missing = [label for label in labels if (label.frame_index, label.candidate_index) not in assignments]
    if missing:
        raise ValueError("Manual annotations refer to candidates absent from tracker output.")

    tracks_by_identity = defaultdict(set)
    identities_by_track = defaultdict(list)
    for label in certain:
        track_id = assignments[label.frame_index, label.candidate_index]
        identity_key = f"{label.window}:{label.identity}"
        tracks_by_identity[identity_key].add(track_id)
        identities_by_track[label.window, track_id].append((label.frame_index, label.identity))

    incorrect_merge_tracks = {
        f"{window}:{track_id}": sorted({identity for _, identity in members})
        for (window, track_id), members in identities_by_track.items()
        if len({identity for _, identity in members}) > 1
    }
    identity_switches = 0
    for members in identities_by_track.values():
        ordered = [identity for _, identity in sorted(members)]
        identity_switches += sum(left != right for left, right in zip(ordered, ordered[1:]))

    tracks_per_identity = {key: len(value) for key, value in sorted(tracks_by_identity.items())}
    fragmentation = sum(max(0, count - 1) for count in tracks_per_identity.values())
    singleton_labels = [label for label in certain if label.legitimate_singleton]
    window_frames = {str(key): set(value) for key, value in annotation.get("windows", {}).items()}
    occurrences = Counter()
    for (frame_index, _), track_id in assignments.items():
        for window in {label.window for label in singleton_labels}:
            frames = window_frames.get(window)
            if frames is None or frame_index in frames:
                occurrences[window, track_id] += 1
    retained_singletons = sum(
        occurrences[label.window, assignments[label.frame_index, label.candidate_index]] == 1
        for label in singleton_labels
    )
    total_tracks = len(lengths)
    singleton_tracks = sum(length == 1 for length in lengths.values())
    return {
        "labelled_identities": len(tracks_per_identity),
        "labelled_candidates": len(certain),
        "ambiguous_candidates": len(labels) - len(certain),
        "tracks_per_identity": tracks_per_identity,
        "confirmed_fragmentation": fragmentation,
        "duplicate_observations": fragmentation,
        "incorrect_merge_count": len(incorrect_merge_tracks),
        "incorrect_merge_tracks": incorrect_merge_tracks,
        "identity_switches": identity_switches,
        "legitimate_singletons": len(singleton_labels),
        "retained_legitimate_singletons": retained_singletons,
        "total_tracks": total_tracks,
        "singleton_tracks": singleton_tracks,
        "singleton_rate": singleton_tracks / total_tracks if total_tracks else 0,
    }


def compare(cache, annotation, before_tracker, after_tracker):
    """Evaluate two tracker configurations over exactly the same candidates."""
    fingerprint = candidate_fingerprint(validate_cache(cache))
    before_assignments, before_lengths = run_tracker(cache, before_tracker)
    after_assignments, after_lengths = run_tracker(cache, after_tracker)
    return {
        "candidate_fingerprint": fingerprint,
        "input_detection_count": sum(len(frame["candidates"]) for frame in cache["frames"]),
        "input_classes": dict(sorted(Counter(
            item["target_class"] for frame in cache["frames"] for item in frame["candidates"]
        ).items())),
        "input_support_labels": dict(sorted(Counter(
            item["support_label"] for frame in cache["frames"] for item in frame["candidates"]
        ).items())),
        "before": identity_metrics(before_assignments, before_lengths, annotation),
        "after": identity_metrics(after_assignments, after_lengths, annotation),
    }
