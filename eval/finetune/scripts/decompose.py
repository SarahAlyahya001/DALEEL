"""Explain the union of A and B correct persons (at their own thresholds) vs C's correct persons.

Usage (DALEEL_REPO set): decompose.py <heridal_n.json> <heridal_s.json> <tA> <tB> <tC>
Steps: union of A@tA and B@tB -> union of the two models scored separately at tC (threshold effect)
-> C@tC (fusion effect), and each lost target is classified as merging or matching.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ens_eval as ens  # noqa: E402

N, S = ens.load(sys.argv[1]), ens.load(sys.argv[2])
tA, tB, tC = (float(x) for x in sys.argv[3:6])


def found(cand, t):
    """{target id: matched detection} for persons, scored as the candidate is scored."""
    out = {}
    for image_id, rn in N.items():
        for d in ens.scored(rn, ens.predictions(cand, rn, S[image_id], t), 'person'):
            if d['status'] == 'TP':
                out[d['matched_gt']] = dict(d, image_id=image_id)
    return out


a, b = found('A', tA), found('B', tB)
n_c, s_c, c = found('A', tC), found('B', tC), found('C', tC)
union_own, union_tc = set(a) | set(b), set(n_c) | set(s_c)
report = {'A_at_own': len(a), 'B_at_own': len(b), 'union_at_own_thresholds': len(union_own),
          'only_A': len(set(a) - set(b)), 'only_B': len(set(b) - set(a)), 'both': len(set(a) & set(b)),
          'n_at_tC': len(n_c), 's_at_tC': len(s_c), 'union_at_tC': len(union_tc), 'C_at_tC': len(c),
          'threshold_effect': {
              'lost_by_raising_n_threshold': sorted(set(a) - set(n_c)),
              'gained_by_lowering_s_threshold': sorted(set(s_c) - set(b)),
              'in_union_own_not_union_tC': sorted(union_own - union_tc),
              'in_union_tC_not_union_own': sorted(union_tc - union_own)},
          'fusion_effect': []}
for gt in sorted(union_tc - set(c)):
    image_id = gt.split(':')[0]
    rn = N[image_id]
    fused = ens.predictions('C', rn, S[image_id], tC)
    target = next(t for t in rn['targets'] if t['id'] == gt)
    near = [dict(f, iou=round(ens.iou(f['box'], target['box']), 3)) for f in fused
            if f['class'] == 'person' and ens.iou(f['box'], target['box']) > 0]
    srcs = [x for x in (n_c.get(gt), s_c.get(gt)) if x]
    # merging: the box that matched this person on its own was paired and replaced by its partner's box
    merged = any(not any(f['box'] == x['box'].tolist() for f in fused) for x in srcs)
    report['fusion_effect'].append({'target': gt, 'cause': 'merging' if merged else 'matching',
                                    'matched_alone_by': [('n' if x is n_c.get(gt) else 's') for x in srcs],
                                    'overlapping_fused_detections': near})
report['in_C_not_in_union_tC'] = sorted(set(c) - union_tc)
print(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
