"""Raw two-class protocol. No YOLO-label conversion, tracking, or mission data."""
from pathlib import Path
import numpy as np

CLASSES = ("person", "car")
GT_MAPPING = {1: "person", 2: "person", 4: "car"}


def clip(box, width, height):
    x1, y1, x2, y2 = box
    result = (max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2))
    if not np.isfinite(result).all() or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Invalid clipped box")
    return result


def coverage(box, regions):
    """Area of rectangle UNION intersected with box / box area (no double count)."""
    rectangles = []
    for r in regions:
        clipped = (max(box[0], r[0]), max(box[1], r[1]), min(box[2], r[2]), min(box[3], r[3]))
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            rectangles.append(clipped)
    edges = sorted({x for r in rectangles for x in (r[0], r[2])})
    area = 0.
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((r[1], r[3]) for r in rectangles if r[0] <= left and r[2] >= right)
        length, end = 0., -float("inf")
        for low, high in intervals:
            length += max(0., high - max(low, end))
            end = max(end, high)
        area += (right - left) * length
    return area / ((box[2] - box[0]) * (box[3] - box[1]))


def parse_annotations(path, width, height):
    rows, regions = [], []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines()):
        values = [float(v) for v in line.strip().rstrip(",").split(",")]
        if len(values) != 8 or not np.isfinite(values).all() or any(v != int(v) for v in values):
            raise ValueError("Malformed raw annotation")
        x, y, w, h, score, category, truncation, occlusion = map(int, values)
        if w <= 0 or h <= 0 or score not in (0, 1) or category not in range(12) or truncation not in (0, 1, 2) or occlusion not in (0, 1, 2):
            raise ValueError("Invalid raw annotation")
        box = clip((x, y, x + w, y + h), width, height)
        if category == 0:
            regions.append(box)
        elif category in GT_MAPPING:
            rows.append({"id": index, "class": GT_MAPPING[category], "box": box, "ignore": score == 0})
    excluded = [row for row in rows if coverage(row["box"], regions) >= .50]
    return {"targets": [row for row in rows if row not in excluded], "regions": regions,
            "region_excluded_gt": len(excluded), "width": width, "height": height}
