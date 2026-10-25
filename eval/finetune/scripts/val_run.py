"""Run one detection configuration over a set of HERIDAL images and save all predictions.

Usage: val_run.py <data_root> <id_regex> <weights.pt> <scene_width> <mode> <size> <overlap> <out.json>
  mode 'full': the whole (resized) frame at imgsz=<size>
  mode 'tile': overlapping <size> x <size> tiles at native pixels, merged with class-wise NMS (IoU 0.5)
Predictions are kept down to confidence 0.001 so that any operating threshold can be applied later;
boxes are stored in original-image coordinates. Per-tile best scores and timings are kept so that
conditional execution of a second model can be costed afterwards.
"""
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import psutil
import torch
import torchvision
from ultralytics import YOLO

root, weights = Path(sys.argv[1]), sys.argv[3]
pattern = None if sys.argv[2].startswith('@') else re.compile(sys.argv[2])
scene_width, mode, size, overlap, out = int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), float(sys.argv[7]), Path(sys.argv[8])
infer = int(sys.argv[9]) if len(sys.argv) > 9 else None  # tile inference size (upscaling) if given
NAMES = {0: 'person', 2: 'car'}
model = YOLO(weights)


def starts(length):
    if length <= size:
        return [0]
    step = int(size * (1 - overlap))
    return list(range(0, length - size, step)) + [length - size]


def windows(h, w):
    if mode == 'full':
        return [(0, 0, w, h)]
    return [(x, y, min(size, w - x), min(size, h - y)) for y in starts(h) for x in starts(w)]


if sys.argv[2].startswith('@'):  # file listing image stems
    wanted = set(Path(sys.argv[2][1:]).read_text(encoding='utf-8').split())
    images = sorted(p for p in root.rglob('*.jpg') if p.stem in wanted)
else:
    images = sorted(p for p in root.rglob('*.jpg') if pattern.search(p.stem))
sys.path.insert(0, os.environ.get('DALEEL_REPO', '.'))
from eval.visdrone import parse_annotations  # noqa: E402  (repository VisDrone protocol)
# warm-up so that the first timed call is not inflated
model.predict(cv2.imread(str(images[0]))[:640, :640], conf=.001, classes=[0, 2], verbose=False, device='cpu',
              imgsz=640)
records, cap_hits = [], 0
for path in images:
    original = cv2.imread(str(path))
    scale = scene_width / original.shape[1] if scene_width else 1.0  # 0 keeps native resolution
    frame = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    boxes, scores, classes, tiles = [], [], [], []
    for x, y, tw, th in windows(h, w):
        crop = frame[y:y + th, x:x + tw]
        t0 = time.perf_counter()
        r = model.predict(crop, conf=.001, classes=[0, 2], verbose=False, device='cpu',
                          imgsz=size if mode == 'full' else (infer or max(tw, th)), iou=.7, max_det=300)[0]
        seconds = time.perf_counter() - t0
        cap_hits += len(r.boxes) >= 300
        best = 0.
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
            scores.append(float(b.conf.item()))
            classes.append(int(b.cls.item()))
            best = max(best, scores[-1])
        tiles.append({'seconds': seconds, 'best_score': best})
    preds = []
    if boxes:
        if mode == 'tile':  # merge duplicates across overlapping tiles; a full frame keeps the model's own NMS
            keep = torchvision.ops.batched_nms(torch.tensor(boxes), torch.tensor(scores), torch.tensor(classes), .5).tolist()
        else:
            keep = list(range(len(boxes)))
        for k in sorted(keep, key=lambda k: -scores[k])[:1000]:
            preds.append({'class': NAMES[classes[k]], 'box': [v / scale for v in boxes[k]], 'score': scores[k]})
    xml = next(root.rglob(path.stem + '.xml'), None)
    targets, regions = [], []
    if xml is not None:  # HERIDAL (persons only)
        for i, o in enumerate(ET.parse(xml).getroot().iter('object')):
            b = o.find('bndbox')
            targets.append({'id': f'{path.stem}:{i}', 'class': 'person', 'ignore': False,
                            'box': [float(b.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]})
    else:  # VisDrone (persons and cars, ignored regions)
        ann = parse_annotations(next(root.rglob(path.stem + '.txt')), original.shape[1], original.shape[0])
        targets = [dict(t, id=f'{path.stem}:{t["id"]}', box=list(t['box'])) for t in ann['targets']]
        regions = [list(r) for r in ann['regions']]
    records.append({'image_id': path.stem, 'width': original.shape[1], 'height': original.shape[0],
                    'targets': targets, 'regions': regions, 'predictions': preds, 'tiles': tiles})
    print(path.stem, len(tiles), 'tiles', round(sum(t['seconds'] for t in tiles), 2), 's', flush=True)

mem = psutil.Process().memory_info()
out.write_text(json.dumps({'config': {'weights': Path(weights).name, 'scene_width': scene_width, 'mode': mode,
                                      'size': size, 'overlap': overlap, 'infer': infer},
                           'peak_working_set_mb': getattr(mem, 'peak_wset', mem.rss) / 2 ** 20,
                           'cap_hits': cap_hits, 'records': records}), encoding='utf-8')
print('done', out.name)
