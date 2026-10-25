"""Final evaluation of the integrated configuration with frozen settings (run once per dataset).

Usage (DALEEL_REPO = integrated repository): system_eval.py <images_root> <id_regex> <scene_width>
       <reference.json> <heridal|visdrone> <out.json>
The application's DualDetector (pinned weights, imgsz 1280) runs on every image prepared as in
val_run.py. Operating point: confidence > 0.42 (the application setting). AP@0.5: the same detector
with confidence > 0.001, fused at that threshold and ranked by fused score (addendum 2, step 5).
Targets, ignored regions and image sizes come from <reference.json> (a val_run.py output for the
same images). HERIDAL false positives are persons only; VisDrone scores persons and cars.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(os.environ['DALEEL_REPO'])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from daleel.pipeline.detector import COCO_KEEP, DETECTION_CONF, DualDetector  # noqa: E402
from daleel.pipeline.weights import WEIGHT_SPECS  # noqa: E402
from eval.detection_metrics import average_precision  # noqa: E402
import ens_eval as ens  # noqa: E402

root, pattern, scene_width = Path(sys.argv[1]), re.compile(sys.argv[2]), int(sys.argv[3])
reference, dataset, out = ens.load(sys.argv[4]), sys.argv[5], Path(sys.argv[6])
classes = ('person',) if dataset == 'heridal' else ('person', 'car')
images = sorted(p for p in root.rglob('*.jpg') if pattern.search(p.stem) and p.stem in reference)
assert {p.stem for p in images} == set(reference), 'image set differs from the reference'

class RankingDetector(DualDetector):
    """AP pass only: zero-area boxes are kept and scored as false positives (repository protocol)
    instead of being refused by the application's strict output check."""
    @staticmethod
    def _extract(result):
        detections = [(COCO_KEEP[int(b.cls.item())], tuple(float(v) for v in b.xyxy[0].tolist()), float(b.conf.item()))
                      for b in result.boxes if int(b.cls.item()) in COCO_KEEP]
        return sorted(detections, key=lambda d: -d[2])


paths = [REPO / 'weights' / spec.filename for spec in WEIGHT_SPECS]
operating = DualDetector(*paths, DETECTION_CONF, 1280, WEIGHT_SPECS)
ranking = RankingDetector(*paths, 0.001, 1280, WEIGHT_SPECS)


def label(c):
    return 'both models' if c.agreement == 'Both models' else 'one model'


records = {'operating': {}, 'ranking': {}}
started = time.perf_counter()
for path in images:
    original = cv2.imread(str(path))
    scale = scene_width / original.shape[1] if scene_width else 1.0
    frame = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for name, detector in (('operating', operating), ('ranking', ranking)):
        preds = [{'class': c.target_class, 'box': [v / scale for v in c.box], 'score': c.score,
                  'agreement': label(c), 'yolov8n_score': c.yolov8n_score, 'yolov8s_score': c.yolov8s_score}
                 for c in detector.detect(frame).candidates]
        records[name][path.stem] = dict(reference[path.stem], predictions=preds)
    print(path.stem, len(records['operating'][path.stem]['predictions']), flush=True)

report = {'images': len(images), 'dataset': dataset, 'scene_width': scene_width, 'confidence': DETECTION_CONF,
          'weights': {spec.filename: spec.sha256 for spec in WEIGHT_SPECS},
          'seconds': round(time.perf_counter() - started, 1)}
for cls in classes:
    tp = fp = gt = 0
    groups = {'both models': [0, 0], 'one model': [0, 0]}
    for image_id, record in records['operating'].items():
        for p in ens.scored(record, record['predictions'], cls):
            if p['status'] in ('TP', 'FP'):
                hit = p['status'] == 'TP'
                tp, fp = tp + hit, fp + (not hit)
                groups[p['agreement']][0 if hit else 1] += 1
        gt += sum(t['class'] == cls and not t['ignore'] for t in record['targets'])
    assigned, gt_rank = [], 0
    for image_id, record in records['ranking'].items():
        assigned += [dict(p, image_id=image_id) for p in ens.scored(record, record['predictions'], cls)]
        gt_rank += sum(t['class'] == cls and not t['ignore'] for t in record['targets'])
    report[cls] = {
        'gt': gt, 'tp': tp, 'fp': fp, 'fn': gt - tp, 'recall': tp / gt if gt else None,
        'precision': tp / (tp + fp) if tp + fp else None, 'fp_per_image': fp / len(images),
        'ap50': average_precision(assigned, gt_rank)[0],
        'by_agreement': {g: {'correct': c, 'false': f, 'precision_wilson95': ens.wilson(c, c + f)}
                         for g, (c, f) in groups.items()}}
if dataset == 'heridal':  # vehicles are not labelled in HERIDAL; car outputs are counted, not scored
    report['car_predictions_operating'] = sum(p['class'] == 'car' for r in records['operating'].values()
                                              for p in r['predictions'])
out.write_text(json.dumps(report, indent=1), encoding='utf-8')
out.with_suffix('.predictions.json').write_text(json.dumps({k: list(v.values()) for k, v in records.items()}),
                                                encoding='utf-8')
print(json.dumps(report, indent=1))
