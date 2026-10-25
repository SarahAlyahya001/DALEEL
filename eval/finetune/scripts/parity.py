"""V2 parity: the integrated application detector vs the evaluated configuration C, image level.

Usage (DALEEL_REPO = the integrated repository): parity.py <images_root> <id_regex|@stems> <scene_width>
        <heridal|visdrone> <n_eval.json> <s_eval.json> <out.json>

Each image is prepared exactly as in val_run.py (INTER_AREA resize to the scene width, 0 = native) and
passed to daleel.pipeline.detector.DualDetector (pinned weights, confidence > 0.42, imgsz 1280). Its
candidates are compared one by one with ens_eval.predictions('C', ...) at 0.42 built from the saved
evaluation predictions, and both are scored with the same matching code.
"""
import json
import os
import re
import sys
from pathlib import Path

import cv2

REPO = Path(os.environ['DALEEL_REPO'])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from daleel.pipeline.detector import DETECTION_CONF, DualDetector  # noqa: E402
from daleel.pipeline.weights import WEIGHT_SPECS  # noqa: E402
import ens_eval as ens  # noqa: E402

root, selector, scene_width, dataset = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
N, S, out = ens.load(sys.argv[5]), ens.load(sys.argv[6]), Path(sys.argv[7])
heridal = dataset == 'heridal'
if selector.startswith('@'):
    wanted = set(Path(selector[1:]).read_text(encoding='utf-8').split())
    images = sorted(p for p in root.rglob('*.jpg') if p.stem in wanted)
else:
    images = sorted(p for p in root.rglob('*.jpg') if re.search(selector, p.stem))
assert {p.stem for p in images} == set(N), 'image set differs from the evaluation'

detector = DualDetector(REPO / 'weights' / WEIGHT_SPECS[0].filename, REPO / 'weights' / WEIGHT_SPECS[1].filename,
                        DETECTION_CONF, 1280, WEIGHT_SPECS)
app_records, mismatches, compared = {}, [], 0
for path in images:
    original = cv2.imread(str(path))
    scale = scene_width / original.shape[1] if scene_width else 1.0
    frame = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    app = [{'class': c.target_class, 'box': [v / scale for v in c.box], 'score': c.score,
            'agreement': 'both models' if c.agreement == 'Both models' else
            ('YOLOv8n only' if c.yolov8n_score is not None else 'YOLOv8s only'),
            'yolov8n_score': c.yolov8n_score, 'yolov8s_score': c.yolov8s_score}
           for c in detector.detect(frame).candidates]
    ref = ens.predictions('C', N[path.stem], S[path.stem], DETECTION_CONF)
    compared += 1
    if len(app) != len(ref):
        mismatches.append({'image': path.stem, 'app': len(app), 'evaluation': len(ref)})
    else:
        for a, r in zip(app, ref):
            box_gap = max(abs(x - y) for x, y in zip(a['box'], r['box']))
            if a['class'] != r['class'] or a['agreement'] != r['agreement'] or abs(a['score'] - r['score']) > 1e-5 \
                    or box_gap > 0.01:
                mismatches.append({'image': path.stem, 'app': a, 'evaluation': {k: r[k] for k in ('class', 'box', 'score', 'agreement')},
                                   'box_gap_px': box_gap})
    app_records[path.stem] = dict(N[path.stem], predictions=app)
    print(path.stem, len(app), 'candidates', flush=True)

# Score both with the same code: the app's candidates are scored as a one-model candidate set.
empty = {k: dict(v, predictions=[]) for k, v in app_records.items()}
app_eval = ens.evaluate(app_records, empty, 'A', 0.0, heridal=heridal)
ref_eval = ens.evaluate(N, S, 'C', DETECTION_CONF, heridal=heridal)
keys = ('tp', 'person_fp', 'car_fp', 'gt', 'recall', 'precision', 'fp_per_image')
report = {'images': compared, 'candidates_app': sum(len(r['predictions']) for r in app_records.values()),
          'candidate_mismatches': mismatches[:50], 'mismatch_count': len(mismatches),
          'app': {k: app_eval[k] for k in keys}, 'evaluation_C': {k: ref_eval[k] for k in keys},
          'agreement_counts_app': {g: sum(p['agreement'] == g for r in app_records.values() for p in r['predictions'])
                                   for g in ('both models', 'YOLOv8n only', 'YOLOv8s only')}}
report['identical_scores'] = all(report['app'][k] == report['evaluation_C'][k] for k in ('tp', 'person_fp', 'car_fp'))
out.write_text(json.dumps(report, indent=1), encoding='utf-8')
(out.with_suffix('.predictions.json')).write_text(json.dumps({'records': list(app_records.values())}), encoding='utf-8')
print(json.dumps({k: report[k] for k in ('images', 'candidates_app', 'mismatch_count', 'app', 'evaluation_C',
                                         'agreement_counts_app', 'identical_scores')}, indent=1))
