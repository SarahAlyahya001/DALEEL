"""Compact deterministic tracks with IoU-first, motion-gated association."""

from dataclasses import dataclass
import math

from .detector import BOTH_MODELS, ONE_MODEL, Candidate, iou


DEFAULT_CENTER_DISTANCE = 1.25
DEFAULT_MAX_AREA_RATIO = 2.50
DEFAULT_MAX_ASPECT_RATIO = 1.80
DEFAULT_ELAPSED_PENALTY = 0.20
CLASS_FLIP_CENTER_DISTANCE = 0.25
CLASS_FLIP_MAX_AREA_RATIO = 1.35
CLASS_FLIP_MAX_ASPECT_RATIO = 1.25
CLASS_FLIP_MIN_IOU = 0.30
CLASS_FLIP_SCORE_MARGIN = 0.05


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _aspect(box):
    height = box[3] - box[1]
    return (box[2] - box[0]) / height if height > 0 else math.inf


def _ratio(a, b):
    low, high = sorted((a, b))
    return high / low if low > 0 and math.isfinite(high) else math.inf


@dataclass(frozen=True)
class Member:
    frame_index: int
    video_time_ms: float
    candidate: Candidate

    @property
    def rank(self):
        # A detection reported by both models is preferred, then the higher score, then the earlier frame.
        # The representative keeps its own frame, box, scores and label; nothing is combined across frames.
        item = self.candidate
        return (item.agreement == BOTH_MODELS, item.score, -self.frame_index)


@dataclass
class Track:
    track_id: int
    target_class: str
    last_box: tuple
    last_seen_sampled_step: int
    first_frame_index: int
    last_frame_index: int
    frame_count: int
    representative: Member
    velocity: tuple[float, float] = (0.0, 0.0)
    first_box: tuple | None = None
    first_seen_sampled_step: int | None = None
    initial_velocity: tuple[float, float] | None = None
    score_sum: float | None = None
    score_count: int | None = None

    def __post_init__(self):
        if self.first_box is None:
            self.first_box = self.last_box
        if self.first_seen_sampled_step is None:
            self.first_seen_sampled_step = self.last_seen_sampled_step
        if self.score_sum is None:
            self.score_sum = self.representative.candidate.score
        if self.score_count is None:
            self.score_count = 1


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int
    track_index: int | None = None
    candidate_index: int | None = None


def _minimum_cost_maximum_matching(track_count, candidate_count, pair_costs):
    """Return a deterministic maximum-cardinality, minimum-cost assignment."""
    source = 0
    track_start = 1
    candidate_start = track_start + track_count
    sink = candidate_start + candidate_count
    graph = [[] for _ in range(sink + 1)]

    def add_edge(start, end, capacity, cost, track_index=None, candidate_index=None):
        forward = _FlowEdge(end, len(graph[end]), capacity, cost, track_index, candidate_index)
        reverse = _FlowEdge(start, len(graph[start]), 0, -cost)
        graph[start].append(forward)
        graph[end].append(reverse)

    for ti in range(track_count):
        add_edge(source, track_start + ti, 1, 0)
    for ci in range(candidate_count):
        add_edge(candidate_start + ci, sink, 1, 0)
    for (ti, ci), cost in sorted(pair_costs.items()):
        add_edge(track_start + ti, candidate_start + ci, 1, cost, ti, ci)

    while True:
        distance = [math.inf] * len(graph)
        previous = [None] * len(graph)
        distance[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distance[node] == math.inf:
                    continue
                for edge_index, edge in enumerate(edges):
                    proposal = distance[node] + edge.cost
                    if edge.capacity and proposal < distance[edge.to]:
                        distance[edge.to] = proposal
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            start, edge_index = previous[node]
            edge = graph[start][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = start

    matched = {}
    for ti in range(track_count):
        for edge in graph[track_start + ti]:
            if edge.candidate_index is not None and edge.capacity == 0:
                matched[edge.candidate_index] = ti
    return matched


class Tracker:
    def __init__(self, iou_thr=0.30, max_gap_steps=3,
                 center_distance=DEFAULT_CENTER_DISTANCE,
                 max_area_ratio=DEFAULT_MAX_AREA_RATIO,
                 max_aspect_ratio=DEFAULT_MAX_ASPECT_RATIO,
                 elapsed_penalty=DEFAULT_ELAPSED_PENALTY):
        self.iou_thr, self.max_gap_steps = iou_thr, max_gap_steps
        self.center_distance = center_distance
        self.max_area_ratio = max_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.elapsed_penalty = elapsed_penalty
        self.active = []
        self._step = -1
        self._next_id = 1
        self.last_assignments = ()
        self.reconciled_track_ids = {}
        self.reconciled_pairs = []

    def _pair_cost(self, track, candidate):
        if track.target_class != candidate.target_class:
            return None
        overlap = iou(track.last_box, candidate.box)
        if overlap >= self.iou_thr:
            return int(round((1 - overlap) * 1_000_000))

        elapsed = self._step - track.last_seen_sampled_step
        previous_center = _center(track.last_box)
        predicted = (previous_center[0] + track.velocity[0] * elapsed,
                     previous_center[1] + track.velocity[1] * elapsed)
        candidate_center = _center(candidate.box)
        scale = max(1.0, math.sqrt(_area(track.last_box)), math.sqrt(_area(candidate.box)))
        center_distance = math.dist(predicted, candidate_center) / scale
        area_ratio = _ratio(_area(track.last_box), _area(candidate.box))
        aspect_ratio = _ratio(_aspect(track.last_box), _aspect(candidate.box))
        if (center_distance > self.center_distance
                or area_ratio > self.max_area_ratio
                or aspect_ratio > self.max_aspect_ratio):
            return None

        fallback = (center_distance
                    + 0.15 * math.log(area_ratio)
                    + 0.15 * math.log(aspect_ratio)
                    + self.elapsed_penalty * (elapsed - 1))
        return 2_000_000 + int(round(fallback * 1_000_000))

    @staticmethod
    def _average_score(track):
        return track.score_sum / track.score_count

    def _class_flip_cost(self, short, stable):
        """Return a strict cost for a one-frame false class at a track boundary."""
        if ({short.target_class, stable.target_class} != {"person", "car"}
                or short.frame_count != 1
                or stable.frame_count < 3
                or short.representative.candidate.agreement != ONE_MODEL
                or short.last_seen_sampled_step + 1 != stable.first_seen_sampled_step
                or stable.initial_velocity is None
                or self._average_score(stable) - self._average_score(short)
                < CLASS_FLIP_SCORE_MARGIN):
            return None

        short_box = short.last_box
        stable_box = stable.first_box
        stable_center = _center(stable_box)
        predicted_center = (
            stable_center[0] - stable.initial_velocity[0],
            stable_center[1] - stable.initial_velocity[1],
        )
        scale = max(1.0, math.sqrt(_area(short_box)), math.sqrt(_area(stable_box)))
        predicted_distance = math.dist(_center(short_box), predicted_center) / scale
        area_ratio = _ratio(_area(short_box), _area(stable_box))
        aspect_ratio = _ratio(_aspect(short_box), _aspect(stable_box))
        overlap = iou(short_box, stable_box)
        if (predicted_distance > CLASS_FLIP_CENTER_DISTANCE
                or area_ratio > CLASS_FLIP_MAX_AREA_RATIO
                or aspect_ratio > CLASS_FLIP_MAX_ASPECT_RATIO
                or overlap < CLASS_FLIP_MIN_IOU):
            return None
        return (predicted_distance, -overlap, area_ratio, aspect_ratio,
                short.track_id, stable.track_id)

    def _reconcile_expired_class_flips(self, expired):
        """Absorb only unique, strongly supported cross-class boundary flips."""
        eligible = {}
        reverse = {}
        for short in expired:
            for stable in self.active:
                cost = self._class_flip_cost(short, stable)
                if cost is not None:
                    eligible.setdefault(short.track_id, []).append((cost, stable))
                    reverse.setdefault(stable.track_id, []).append(short)

        removed = set()
        for short in sorted(expired, key=lambda item: item.track_id):
            options = eligible.get(short.track_id, [])
            if len(options) != 1:
                continue
            _, stable = options[0]
            if len(reverse.get(stable.track_id, [])) != 1:
                continue
            stable.first_frame_index = min(stable.first_frame_index, short.first_frame_index)
            stable.last_frame_index = max(stable.last_frame_index, short.last_frame_index)
            stable.frame_count += short.frame_count
            self.reconciled_track_ids[short.track_id] = stable.track_id
            self.reconciled_pairs.append((short.track_id, stable.track_id))
            removed.add(short.track_id)
        return [track for track in expired if track.track_id not in removed]

    def update(self, frame_index, video_time_ms, candidates):
        """Return expired tracks and tracks whose representative just changed."""
        self._step += 1
        expired, active = [], []
        for track in self.active:
            missed_sampled_steps = self._step - track.last_seen_sampled_step - 1
            (expired if missed_sampled_steps > self.max_gap_steps else active).append(track)
        self.active = active

        pair_costs = {}
        for ti, track in enumerate(active):
            for ci, candidate in enumerate(candidates):
                cost = self._pair_cost(track, candidate)
                if cost is not None:
                    pair_costs[ti, ci] = cost * 1_000_000 + track.track_id * 1_000 + ci
        matched_indices = _minimum_cost_maximum_matching(len(active), len(candidates), pair_costs)
        matched = {ci: active[ti] for ci, ti in matched_indices.items()}

        changed = []
        assignments = []
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
                old_center = _center(track.last_box)
                new_center = _center(candidate.box)
                elapsed = self._step - track.last_seen_sampled_step
                observed_velocity = ((new_center[0] - old_center[0]) / elapsed,
                                     (new_center[1] - old_center[1]) / elapsed)
                if track.frame_count == 1:
                    track.initial_velocity = observed_velocity
                track.velocity = observed_velocity if track.frame_count == 1 else (
                    (track.velocity[0] + observed_velocity[0]) / 2,
                    (track.velocity[1] + observed_velocity[1]) / 2,
                )
                track.last_box = candidate.box
                track.last_seen_sampled_step = self._step
                track.last_frame_index = frame_index
                track.frame_count += 1
                track.score_sum += candidate.score
                track.score_count += 1
                if member.rank > track.representative.rank:
                    track.representative = member
                    changed.append(track)
            assignments.append(track.track_id)
        self.last_assignments = tuple(assignments)
        return self._reconcile_expired_class_flips(expired), changed

    def finalize(self):
        tracks = self._reconcile_expired_class_flips(self.active)
        self.active = []
        self.last_assignments = ()
        return tracks
