"""Agreement-based review ordering (ACCEPTANCE_CRITERIA.md, addendum 3; exploratory).

Usage (DALEEL_REPO set): review_order.py <out.json> <heridal_n> <heridal_s> <heridal_jpeg_dir>
                                         <visdrone_n> <visdrone_s>
Queues: O1 C by confidence, O2 C agreement-first, O3 A by confidence, O4 B by confidence.
Labels are fixed once per candidate set; only the order differs between O1 and O2.
"""
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ens_eval as ens  # noqa: E402

T_A, T_B, T_C = .27, .46, .42
QUEUES = ('O1', 'O2', 'O3', 'O4')
BUDGETS = (.10, .25, .50, .75, 1.00)
FRACTIONS = (.50, .75, .90)
SEED, RESAMPLES = 20261024, 2000


def labelled(cand, rn, rs, t, heridal):
    """The candidate set of one image with fixed correctness labels (1 correct, 0 incorrect)."""
    preds = ens.predictions(cand, rn, rs, t)
    items = []
    for cls in ('person', 'car'):
        if heridal and cls == 'car':
            if rn['image_id'] not in ens.VAL_CAR_IMAGES:  # unlabelled class: every car candidate is a false alarm
                items += [dict(p, label=0) for p in preds if p['class'] == 'car']
            continue
        items += [dict(p, label=int(p['status'] == 'TP')) for p in ens.scored(rn, preds, cls)
                  if p['status'] in ('TP', 'FP')]
    # within-image tie order: class, then box
    items.sort(key=lambda p: (p['class'], [float(v) for v in p['box']]))
    return [{'score': float(p['score']), 'agreed': p['agreement'] == 'both models', 'label': p['label']}
            for p in items]


def targets(r, heridal):
    classes = ('person',) if heridal else ('person', 'car')
    return sum(x['class'] in classes and not x['ignore'] for x in r['targets'])


def prepare(N, S, heridal):
    """{image_id: {'T': targets, queue: (score, agreed, label) arrays}} in image-id order."""
    data = {}
    for image_id in sorted(N):
        rn, rs = N[image_id], S[image_id]
        c = labelled('C', rn, rs, T_C, heridal)
        sets = {'O1': c, 'O2': c, 'O3': labelled('A', rn, rs, T_A, heridal), 'O4': labelled('B', rn, rs, T_B, heridal)}
        data[image_id] = {'T': targets(rn, heridal),
                          **{q: tuple(np.array([p[k] for p in v], dtype=float) for k in ('score', 'agreed', 'label'))
                             for q, v in sets.items()}}
    return data


def order(q, score, agreed, tie):
    """Index order of one queue. lexsort uses the last key first."""
    if q == 'O2':
        return np.lexsort((tie, -score, ~agreed.astype(bool)))
    return np.lexsort((tie, -score))


def curves(data, draw):
    """found-vs-reviewed curve of every queue for a list of drawn image ids (copies allowed)."""
    out = {}
    for q in QUEUES:
        parts = [data[i][q] for i in draw]
        score = np.concatenate([p[0] for p in parts])
        agreed = np.concatenate([p[1] for p in parts])
        label = np.concatenate([p[2] for p in parts])
        # tie-break: drawn image position (image ids are drawn in id order within the sample), then within-image order
        tie = np.concatenate([pos * 100000 + np.arange(len(p[0])) for pos, p in enumerate(parts)])
        out[q] = np.cumsum(label[order(q, score, agreed, tie)])
    return out


def measures(data, draw):
    cur = curves(data, draw)
    T = sum(data[i]['T'] for i in draw)
    L = len(cur['O1'])
    M = min(int(c[-1]) if len(c) else 0 for c in cur.values())

    def found(c, k):
        return int(c[min(k, len(c)) - 1]) if len(c) else 0

    res = {'T': T, 'L': L, 'M': M}
    for q, c in cur.items():
        padded = np.concatenate([c[:L], np.full(max(0, L - len(c)), c[-1] if len(c) else 0)])
        res[q] = {'length': len(c), 'final_correct': int(c[-1]) if len(c) else 0,
                  'auc': float(padded.sum() / (L * T)) if L and T else None,
                  'found_at_budget': {f'{f:.2f}': found(c, math.ceil(f * L)) for f in BUDGETS},
                  'reviewed_to_find': {f'{g:.2f}': (int(np.argmax(c >= math.ceil(g * M))) + 1 if M else None)
                                       for g in FRACTIONS}}
    res['m'] = {f'{g:.2f}': math.ceil(g * M) for g in FRACTIONS}
    res['budget_k'] = {f'{f:.2f}': math.ceil(f * L) for f in BUDGETS}
    return res


def differences(m):
    out = {}
    for other in ('O1', 'O3', 'O4'):
        d = {'auc': m['O2']['auc'] - m[other]['auc']}
        d.update({f'found_at_{k}': m['O2']['found_at_budget'][k] - m[other]['found_at_budget'][k] for k in m['O2']['found_at_budget']})
        d.update({f'reviewed_to_{k}': m['O2']['reviewed_to_find'][k] - m[other]['reviewed_to_find'][k] for k in m['O2']['reviewed_to_find']})
        out[f'O2-{other}'] = d
    return out


def time_blocks(times, seconds):
    """Capture-time clusters: per location group, a new block when an image is > seconds after the block start."""
    clusters, start, key = {}, {}, {}
    for image_id in sorted(times, key=lambda i: (i.split('_')[1], times[i])):
        group = image_id.split('_')[1]
        if group not in start or (times[image_id] - start[group]).total_seconds() > seconds:
            start[group] = times[image_id]
            key[group] = key.get(group, -1) + 1
        clusters[image_id] = f'{group}-{key[group]}'
    return clusters


def exif_times(jpeg_dir, ids):
    from PIL import Image
    out = {}
    for image_id in ids:
        exif = Image.open(Path(jpeg_dir) / f'{image_id}.jpg').getexif()
        out[image_id] = datetime.strptime(exif.get_ifd(0x8769)[36867], '%Y:%m:%d %H:%M:%S')
    return out


def bootstrap(data, clusters):
    groups = {}
    for image_id in sorted(data):
        groups.setdefault(clusters[image_id], []).append(image_id)
    names = sorted(groups)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(RESAMPLES):
        drawn = rng.integers(0, len(names), len(names))
        draw = sorted(i for g in drawn for i in groups[names[g]])  # image-id order; copies adjacent
        samples.append(differences(measures(data, draw)))
    intervals = {}
    for pair in samples[0]:
        intervals[pair] = {}
        for k in samples[0][pair]:
            vals = [s[pair][k] for s in samples if s[pair][k] is not None]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            intervals[pair][k] = [float(lo), float(hi)]
    return {'clusters': len(names), 'resamples': RESAMPLES, 'seed': SEED, 'intervals95': intervals}


def analyse(data, cluster_sets):
    point = measures(data, sorted(data))
    out = {'images': len(data), 'point': point, 'differences': differences(point)}
    for name, clusters in cluster_sets.items():
        out[f'bootstrap_{name}'] = bootstrap(data, clusters)
    return out


def main(argv):
    out_path, hn, hs, jpeg, vn, vs = argv
    H_N, H_S, V_N, V_S = ens.load(hn), ens.load(hs), ens.load(vn), ens.load(vs)
    assert H_N.keys() == H_S.keys() and V_N.keys() == V_S.keys()
    times = exif_times(jpeg, H_N)
    report = {
        'heridal': analyse(prepare(H_N, H_S, True), {'30s': time_blocks(times, 30), '60s': time_blocks(times, 60)}),
        'visdrone': analyse(prepare(V_N, V_S, False), {'sequence': {i: i.split('_')[0] for i in V_N}}),
    }
    primary = {'heridal': report['heridal']['bootstrap_30s'], 'visdrone': report['visdrone']['bootstrap_sequence']}
    report['decision'] = {
        'auc_intervals_above_zero': {ds: {pair: b['intervals95'][pair]['auc'][0] > 0 for pair in b['intervals95']}
                                     for ds, b in primary.items()}}
    report['decision']['useful_advantage'] = all(v for d in report['decision']['auc_intervals_above_zero'].values()
                                                 for v in d.values())
    Path(out_path).write_text(json.dumps(report, indent=1), encoding='utf-8')
    for ds in ('heridal', 'visdrone'):
        p = report[ds]['point']
        print(ds, 'T', p['T'], 'L', p['L'], 'M', p['M'], 'k', p['budget_k'], 'm', p['m'])
        for q in QUEUES:
            print(' ', q, {k: p[q][k] for k in ('length', 'final_correct')}, 'AUC', round(p[q]['auc'], 4),
                  'found', p[q]['found_at_budget'], 'reviewed', p[q]['reviewed_to_find'])
        for name in [k for k in report[ds] if k.startswith('bootstrap_')]:
            b = report[ds][name]
            print(' ', name, 'clusters', b['clusters'])
            for pair, d in report[ds]['differences'].items():
                print('   ', pair, 'AUC diff', round(d['auc'], 4), [round(x, 4) for x in b['intervals95'][pair]['auc']])
    print('decision', json.dumps(report['decision']))


if __name__ == '__main__':
    main(sys.argv[1:])
