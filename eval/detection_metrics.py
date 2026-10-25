"""Independent deterministic NumPy metric implementation; 101-point interpolated AP."""
import numpy as np
from eval.visdrone import CLASSES, clip, coverage


def divide(n, d):
    return n / d if d else None


def counts(tp, fp, gt):
    return {"ground_truth": gt, "prediction_count": tp + fp, "tp": tp, "fp": fp, "fn": gt - tp,
            "precision": divide(tp, tp + fp), "recall": divide(tp, gt),
            "f1": divide(2 * tp, gt + tp + fp)}


def match_image(predictions, annotation, threshold=.50):
    """Valid GT first, confidence order; ignores absorb unmatched predictions only."""
    result, used = [], set()
    gt = annotation["targets"]
    for prediction in sorted(predictions, key=lambda p: (-p["score"], p["id"])):
        p = dict(prediction)
        if not np.isfinite(p["box"]).all():
            raise ValueError("Nonfinite prediction geometry")
        if p.get("geometry_status") == "degenerate_after_clipping":
            # Ranking-only policy: retain as FP, bypass ALL geometry/ignore math.
            p["status"] = "FP"
            result.append(p)
            continue
        box = clip(p["box"], annotation["width"], annotation["height"])
        p["status"] = "FP"
        if coverage(box, annotation["regions"]) >= .50:
            p["status"] = "ignored_region"
            result.append(p)
            continue
        candidates = [g for g in gt if g["class"] == p["class"]]
        if candidates:
            boxes = np.array([g["box"] for g in candidates])
            intersection = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2])).prod(axis=1)
            pred_area = (box[2] - box[0]) * (box[3] - box[1])
            overlaps = intersection / (np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) + pred_area - intersection)
            eligible = [(float(overlap), g["id"]) for g, overlap in zip(candidates, overlaps)
                        if not g["ignore"] and g["id"] not in used and overlap >= threshold]
            if eligible:
                _, identifier = sorted(eligible, key=lambda pair: (-pair[0], pair[1]))[0]
                used.add(identifier)
                p["status"] = "TP"
                p["matched_gt"] = identifier
            elif any(g["ignore"] and intersection[i] / pred_area >= threshold for i, g in enumerate(candidates)):
                p["status"] = "ignored_score_zero"
        result.append(p)
    return result


def average_precision(assignments, gt_count):
    eligible = sorted((p for p in assignments if p["status"] in ("TP", "FP")),
                      key=lambda p: (-p["score"], p.get("image_id", ""), p["id"]))
    grid = np.linspace(0, 1, 101)
    if not gt_count:
        return None, []
    tp = np.cumsum([p["status"] == "TP" for p in eligible])
    recall = tp / gt_count
    precision = tp / np.arange(1, len(eligible) + 1) if eligible else np.array([])
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    sampled = [float(envelope[np.searchsorted(recall, r, side="left")])
               if len(recall) and np.searchsorted(recall, r, side="left") < len(recall) else 0. for r in grid]
    return float(np.mean(sampled)), [{"recall": float(r), "precision": p} for r, p in zip(grid, sampled)]


def summarize(records, annotations, ranking=False):
    """records: image_id -> predictions. Returns metrics, AP50 curves, fixed assignments."""
    ground_truth = {cls: sum(g["class"] == cls and not g["ignore"] for a in annotations.values() for g in a["targets"]) for cls in CLASSES}
    thresholds = [round(.50 + .05 * i, 2) for i in range(10)] if ranking else [.50]
    aps = {cls: [] for cls in CLASSES}
    fixed, curves = [], {}
    for threshold in thresholds:
        assignments = [dict(p, image_id=image_id) for image_id in sorted(records)
                       for p in match_image(records[image_id], annotations[image_id], threshold)]
        if threshold == .50:
            fixed = assignments
        for cls in CLASSES:
            ap, curve = average_precision([p for p in assignments if p["class"] == cls], ground_truth[cls])
            aps[cls].append(ap)
            if threshold == .50:
                curves[cls] = curve
    per_class = {}
    for cls in CLASSES:
        items = [p for p in fixed if p["class"] == cls]
        metric = counts(sum(p["status"] == "TP" for p in items), sum(p["status"] == "FP" for p in items), ground_truth[cls])
        metric.update(raw_predictions=len(items), ignored_predictions=sum(p["status"].startswith("ignored") for p in items))
        if ranking:
            metric.update(ap50=aps[cls][0], map50_95=float(np.mean(aps[cls])) if ground_truth[cls] else None,
                          ap_by_iou=dict(zip(map(str, thresholds), aps[cls])))
        per_class[cls] = metric
    combined = counts(sum(v["tp"] for v in per_class.values()), sum(v["fp"] for v in per_class.values()), sum(ground_truth.values()))
    macro = {}
    for key in ("precision", "recall", "f1") + (("ap50", "map50_95") if ranking else ()):
        values = [v[key] for v in per_class.values() if v[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    return {"per_class": per_class, "combined": combined, "macro": macro}, curves, fixed


def support_partition(assignments):
    """Partition A's fixed assignments; there are no separate C/D FN populations."""
    output = {}
    for cls in (*CLASSES, "combined"):
        pool = [p for p in assignments if cls == "combined" or p["class"] == cls]
        groups = {}
        for label in ("Dual-Model", "Screening-Only"):
            items = [p for p in pool if p["support_label"] == label]
            tp, fp = (sum(p["status"] == status for p in items) for status in ("TP", "FP"))
            groups[label] = {"raw_predictions": len(items), "prediction_count": tp + fp, "tp": tp, "fp": fp,
                             "ignored_predictions": len(items) - tp - fp, "precision": divide(tp, tp + fp),
                             "proportion_of_A_tp": divide(tp, sum(p["status"] == "TP" for p in pool)),
                             "proportion_of_A_fp": divide(fp, sum(p["status"] == "FP" for p in pool))}
        dual, screen = (groups[label]["precision"] for label in ("Dual-Model", "Screening-Only"))
        groups["precision_difference"] = dual - screen if dual is not None and screen is not None else None
        output[cls] = groups
    return output
