"""The original pretrained configuration (2026-10-14), kept unchanged to reproduce eval/results/.

That configuration screened each sampled frame with YOLOv8n (confidence > 0.30), ran YOLOv8s
(> 0.50) only on frames with a YOLOv8n candidate, and labeled each YOLOv8n candidate Dual-Model or
Screening-Only. The application now uses daleel.pipeline.detector instead.
"""

from dataclasses import dataclass
import math

from daleel.pipeline.detector import COCO_KEEP, iou

DUAL_MODEL = "Dual-Model"
SCREENING_ONLY = "Screening-Only"


@dataclass(frozen=True)
class LegacyCandidate:
    target_class: str
    box: tuple[float, float, float, float]  # Always the YOLOv8n box.
    screening_score: float
    verification_score: float | None = None
    support_label: str = SCREENING_ONLY


def extract(result):
    """Strict check of a model result, in the original stable order (class, box, -score)."""
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        if class_id not in COCO_KEEP:
            continue
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
        score = float(box.conf.item())
        if (len(xyxy) != 4 or not all(math.isfinite(v) for v in xyxy)
                or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("The model returned invalid candidate data.")
        detections.append((COCO_KEEP[class_id], xyxy, score))
    return sorted(detections, key=lambda d: (d[0], d[1], -d[2]))


def consensus(n_dets, s_dets) -> list[LegacyCandidate]:
    """Descending-IoU greedy matching; input indices break exact ties."""
    n_dets = [det for det in n_dets if det[0] in COCO_KEEP.values()]
    s_dets = [det for det in s_dets if det[0] in COCO_KEEP.values()]
    pairs = sorted(
        (-overlap, ni, si)
        for ni, (nc, nb, _) in enumerate(n_dets)
        for si, (sc, sb, _) in enumerate(s_dets)
        if nc == sc and (overlap := iou(nb, sb)) >= 0.50
    )
    matches, used_s = {}, set()
    for _, ni, si in pairs:
        if ni not in matches and si not in used_s:
            matches[ni] = si
            used_s.add(si)
    return [
        LegacyCandidate(cls, box, score, s_dets[matches[i]][2], DUAL_MODEL)
        if i in matches else LegacyCandidate(cls, box, score)
        for i, (cls, box, score) in enumerate(n_dets)
    ]
