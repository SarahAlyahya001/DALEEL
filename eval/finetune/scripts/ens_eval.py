"""Two-model comparison on validation predictions (ACCEPTANCE_CRITERIA.md, addendum 2).

Usage (DALEEL_REPO = repository root): ens_eval.py <out.json> <heridal_n.json> <heridal_s.json>
                   [<visdrone_n.json> <visdrone_s.json> <visdrone_pretrained_n.json>]
Inputs are val_run.py outputs (full frame, imgsz 1280, predictions down to confidence 0.001).
Candidates: A = YOLOv8n alone, B = YOLOv8s alone, C = both models on every frame, fused at the
threshold being evaluated (class-aware pairing at IoU >= 0.50, higher-scoring box, max score).
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ['DALEEL_REPO'])
from eval.detection_metrics import match_image, average_precision  # noqa: E402

_review = Path(__file__).resolve().parent.parent / 'splits' / 'car_review.json'
VAL_CAR_IMAGES = {x for x in json.loads(_review.read_text())['exclude'] if '_BLA_' in x or '_RAK_' in x}
THRESHOLDS = np.round(np.arange(.01, .91, .01), 2)
FP_LEVELS = (.25, .50, 1.00)
BANDS = ((None, .50), (.50, .75), (.75, 1.01))  # lower edge None = the operating threshold
SEED, RESAMPLES = 20261023, 2000
LIMITS = {'recall': .50, 'precision': .50, 'fp_per_image': 1.0}


def load(path):
    return {r['image_id']: r for r in json.loads(Path(path).read_text(encoding='utf-8'))['records']}


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def fuse(pn, ps):
    """App consensus pairing (greedy descending IoU >= 0.50, same class); every detection is kept."""
    fused = []
    for cls in ('person', 'car'):
        n = [p for p in pn if p['class'] == cls]
        s = [p for p in ps if p['class'] == cls]
        pairs = sorted((-o, i, j) for i, a in enumerate(n) for j, b in enumerate(s)
                       if (o := iou(a['box'], b['box'])) >= .50)
        used_n, used_s = set(), set()
        for _, i, j in pairs:
            if i not in used_n and j not in used_s:
                used_n.add(i)
                used_s.add(j)
                top = n[i] if n[i]['score'] >= s[j]['score'] else s[j]
                fused.append(dict(top, score=max(n[i]['score'], s[j]['score']), agreement='both models',
                                  score_n=n[i]['score'], score_s=s[j]['score']))
        fused += [dict(p, agreement='YOLOv8n only') for i, p in enumerate(n) if i not in used_n]
        fused += [dict(p, agreement='YOLOv8s only') for j, p in enumerate(s) if j not in used_s]
    return fused


def predictions(cand, rn, rs, t):
    pn = [p for p in rn['predictions'] if p['score'] > t]
    ps = [p for p in rs['predictions'] if p['score'] > t]
    if cand == 'A':
        return [dict(p, agreement='YOLOv8n only') for p in pn]
    if cand == 'B':
        return [dict(p, agreement='YOLOv8s only') for p in ps]
    return fuse(pn, ps)


def geometry(box, width, height):
    """Repository rule: a box with no area left after clipping is scored as a false positive."""
    x1, y1, x2, y2 = box
    degenerate = x2 <= x1 or y2 <= y1 or min(width, x2) <= max(0, x1) or min(height, y2) <= max(0, y1)
    return 'degenerate_after_clipping' if degenerate else 'valid'


def annotation(r):
    return {'width': r['width'], 'height': r['height'], 'regions': [tuple(x) for x in r.get('regions', [])],
            'targets': [dict(t, box=tuple(t['box'])) for t in r['targets']]}


def scored(r, preds, cls):
    items = [dict(p, id=i, geometry_status=geometry(p['box'], r['width'], r['height']), box=np.array(p['box'], float))
             for i, p in enumerate(preds) if p['class'] == cls]
    return match_image(items, annotation(r))


def evaluate(N, S, cand, t, heridal=True):
    """Per-image person TP/GT, false positives (AC3 definition on HERIDAL) and matched detections."""
    per_image, person_fp, car_fp, detections = [], 0, 0, []
    for image_id, rn in N.items():
        preds = predictions(cand, rn, S[image_id], t)
        m = scored(rn, preds, 'person')
        tp = sum(p['status'] == 'TP' for p in m)
        person_fp += sum(p['status'] == 'FP' for p in m)
        if heridal:  # persons only are labelled; car predictions are false alarms except in reviewed images
            car_fp += 0 if image_id in VAL_CAR_IMAGES else sum(p['class'] == 'car' for p in preds)
        else:
            car_fp += sum(p['status'] == 'FP' for p in scored(rn, preds, 'car'))
        per_image.append((tp, sum(x['class'] == 'person' and not x['ignore'] for x in rn['targets'])))
        detections += [dict(p, image_id=image_id) for p in m if p['status'] in ('TP', 'FP')]
    tp, gt = (sum(x[k] for x in per_image) for k in (0, 1))
    return {'tp': tp, 'person_fp': person_fp, 'car_fp': car_fp, 'gt': gt, 'recall': tp / gt,
            'precision': tp / (tp + person_fp) if tp + person_fp else None,
            'fp_per_image': (person_fp + car_fp) / len(N), 'per_image': per_image, 'detections': detections}


def ap50(N, S, cand, cls):
    assigned, gt = [], 0
    for image_id, rn in N.items():
        assigned += [dict(p, image_id=image_id) for p in scored(rn, predictions(cand, rn, S[image_id], .001), cls)]
        gt += sum(x['class'] == cls and not x['ignore'] for x in rn['targets'])
    return average_precision(assigned, gt)[0]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def paired(sweeps, matched, x, y):
    """Recall of x minus recall of y at 0.50 FP/img, with the paired percentile bootstrap over images."""
    px = np.array(sweeps[x][matched[(x, .50)]]['per_image'])
    py = np.array(sweeps[y][matched[(y, .50)]]['per_image'])
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(RESAMPLES):
        k = rng.integers(0, len(px), len(px))
        g = px[k, 1].sum()
        if g:
            diffs.append(px[k, 0].sum() / g - py[k, 0].sum() / g)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rx, ry = px[:, 0].sum() / px[:, 1].sum(), py[:, 0].sum() / py[:, 1].sum()
    return {'compared': f'{x} - {y}', f'recall_{x}': rx, f'recall_{y}': ry, 'difference': rx - ry,
            'bootstrap95': [float(lo), float(hi)], 'resamples_used': len(diffs), 'interval_above_zero': bool(lo > 0)}


def slim(e):
    return {k: v for k, v in e.items() if k not in ('per_image', 'detections')}


def main(argv):
    H_N, H_S = load(argv[1]), load(argv[2])
    assert H_N.keys() == H_S.keys(), 'the two prediction files cover different images'
    report = {'images': len(H_N), 'persons': sum(x['class'] == 'person' for r in H_N.values() for x in r['targets'])}
    sweeps = {c: {float(t): evaluate(H_N, H_S, c, t) for t in THRESHOLDS} for c in 'ABC'}
    operating, matched = {}, {}
    for c in 'ABC':
        ok = [(t, e) for t, e in sweeps[c].items() if e['precision'] is not None and e['precision'] >= .50 and e['tp']]
        t, e = max(ok, key=lambda x: (x[1]['recall'], x[0])) if ok else (None, None)
        operating[c] = t
        report[c] = {'operating_threshold': t, 'at_operating': slim(e) if e else None,
                     'passes': {k: bool(e) and (e[k] >= lim if k != 'fp_per_image' else e[k] <= lim) for k, lim in LIMITS.items()},
                     'person_ap50': ap50(H_N, H_S, c, 'person'), 'matched_fp': {}}
        for level in FP_LEVELS:
            within = [t for t, e in sweeps[c].items() if e['fp_per_image'] <= level]
            if within:
                t = min(within)
                report[c]['matched_fp'][str(level)] = {'threshold': t, **slim(sweeps[c][t])}
                matched[(c, level)] = t

    # M3 unique correct detections (A vs B at their operating thresholds; C's recoveries by YOLOv8s)
    found = {c: {d['matched_gt'] for d in sweeps[c][operating[c]]['detections'] if d['status'] == 'TP'}
             for c in 'AB' if operating[c] is not None}
    if len(found) == 2:
        report['unique_correct'] = {'YOLOv8n only': len(found['A'] - found['B']), 'YOLOv8s only': len(found['B'] - found['A']),
                                    'both': len(found['A'] & found['B'])}
    if operating['C'] is not None:
        det = sweeps['C'][operating['C']]['detections']
        report['C_true_positives_by_source'] = {g: sum(d['status'] == 'TP' and d['agreement'] == g for d in det)
                                                for g in ('both models', 'YOLOv8n only', 'YOLOv8s only')}
        # M4 agreement as evidence
        groups = {'both models': lambda d: d['agreement'] == 'both models',
                  'single model': lambda d: d['agreement'] != 'both models'}
        m4 = {}
        for name, keep in groups.items():
            sel = [d for d in det if keep(d)]
            m4[name] = {'detections': len(sel), 'precision_wilson95': wilson(sum(d['status'] == 'TP' for d in sel), len(sel))}
            for lo, hi in BANDS:  # [lo, hi); every detection already scores above the operating threshold
                lo_v = operating['C'] if lo is None else lo
                band = [d for d in sel if lo_v <= d['score'] < hi]
                m4[name][f'band_{lo_v:.2f}_{min(hi, 1):.2f}'] = {
                    'detections': len(band), 'precision_wilson95': wilson(sum(d['status'] == 'TP' for d in band), len(band))}
        both, single = m4['both models']['precision_wilson95'], m4['single model']['precision_wilson95']
        bands_ok = all(
            m4['both models'][k]['precision_wilson95'][0] >= m4['single model'][k]['precision_wilson95'][0]
            for k in m4['both models'] if k.startswith('band_')
            and m4['both models'][k]['detections'] >= 10 and m4['single model'][k]['detections'] >= 10)
        m4['agreement_is_useful_evidence'] = bool(both and single and both[1] > single[2] and bands_ok)
        report['agreement'] = m4

    # Primary decision: recall at 0.50 FP/img, C vs the better single model, paired bootstrap over images;
    # the design rule applies the same test to B vs A
    if all((c, .50) in matched for c in 'ABC'):
        rec = {c: sweeps[c][matched[(c, .50)]]['recall'] for c in 'AB'}
        best = 'A' if rec['A'] >= rec['B'] else 'B'
        report['primary'] = {'better_single': best, **paired(sweeps, matched, 'C', best)}
        report['B_vs_A'] = paired(sweeps, matched, 'B', 'A')

    # Secondary: held-out VisDrone images (persons and cars), M6 car check
    if len(argv) >= 6:
        V_N, V_S, V_B = load(argv[3]), load(argv[4]), load(argv[5])
        car_baseline = ap50(V_B, V_B, 'A', 'car')
        report['visdrone_heldout'] = {'car_ap50_pretrained_n': car_baseline}
        for c in 'ABC':
            e = evaluate(V_N, V_S, c, operating[c], heridal=False) if operating[c] is not None else None
            car = ap50(V_N, V_S, c, 'car')
            report['visdrone_heldout'][c] = {'person_ap50': ap50(V_N, V_S, c, 'person'), 'car_ap50': car,
                                             'car_check': car >= car_baseline - .02,
                                             'at_heridal_operating_threshold': slim(e) if e else None}
            report[c]['car_check'] = car >= car_baseline - .02

    Path(argv[0]).write_text(json.dumps(report, indent=1), encoding='utf-8')
    for c in 'ABC':
        r = report[c]
        a = r['at_operating'] or {}
        print(c, 'thr', r['operating_threshold'], 'recall', round(a.get('recall', 0), 3), 'prec', round(a.get('precision') or 0, 3),
              'FP/img', round(a.get('fp_per_image', 0), 2), 'AP50', round(r['person_ap50'], 3),
              'recall@FP', {k: round(v['recall'], 3) for k, v in r['matched_fp'].items()})
    for k in ('unique_correct', 'C_true_positives_by_source', 'primary', 'B_vs_A'):
        print(k, json.dumps(report.get(k)))
    print('agreement useful:', report.get('agreement', {}).get('agreement_is_useful_evidence'))


if __name__ == '__main__':
    main(sys.argv[1:])
