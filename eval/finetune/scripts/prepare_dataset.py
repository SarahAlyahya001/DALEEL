"""Build the YOLOv8n fine-tuning dataset (HERIDAL persons + VisDrone persons and cars).

Usage: prepare_dataset.py <raw_dir> <heridal_val_dir> <weights_dir> <repo_root> <out_dir>

Rules (recorded in prep_manifest.json):
  * Splits are fixed per source image before tiling: HERIDAL training split without the locations
    BLA and RAK -> train; BLA and RAK -> val; HERIDAL test split -> never used. VisDrone: the
    sequence-split subset in visdrone_subset_manifest.json (train / val sequences).
  * Scale: HERIDAL images resized to 1920 px wide (1080p-equivalent frame, the DALEEL input);
    VisDrone images kept at native size.
  * Tiles: 640 x 640, stride 480 (25% overlap), last tile aligned to the image edge.
  * Labels use the COCO class indices of the deployed models (0 person, 2 car); data.yaml keeps
    all 80 COCO names so the detection head is fine-tuned, not replaced.
  * An object cut by a tile edge is labelled if >= 50% of its area is inside the tile and the
    clipped box is >= 4 px on each side; otherwise its visible part is filled with the tile mean.
  * VisDrone ignored regions (category 0) and score-0 boxes are filled with the tile mean.
    VisDrone pedestrian/people -> person, car -> car (as in eval/visdrone.py); other categories
    (van, truck, bus, ...) are background, consistent with the evaluation protocol.
  * Unlabelled targets. HERIDAL labels persons only: every image was reviewed visually and the
    images showing cars are excluded (raw/car_review.json; the pretrained model could not detect these
    cars, so an automatic check was not usable). VisDrone: tiles containing a pretrained-YOLOv8s
    (imgsz 1920) person/car prediction >= 0.60 that overlaps no annotated object or ignored region
    (IoU < 0.10) are excluded and saved to a review sheet.
  * Tiles without targets are kept at 0.3 x the number of positive tiles per source (seeded).
"""
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

raw, heridal_val, weights, repo, out = map(Path, sys.argv[1:6])
# HERIDAL images excluded after the visual review for unlabelled cars (car_review.json)
review_file = raw / 'car_review.json'
EXCLUDE = set(json.loads(review_file.read_text())['exclude']) if review_file.exists() else set()
sys.path.insert(0, str(repo))
from eval.visdrone import GT_MAPPING  # noqa: E402  (1, 2 -> person; 4 -> car)

TILE, STRIDE, HERIDAL_WIDTH, NEG_RATIO, SEED = 640, 480, 1920, .3, 20261019
CLS = {'person': 0, 'car': 2}
rng = random.Random(SEED)
checker = YOLO(str(weights / 'yolov8s.pt'))
COCO_NAMES = checker.names
stats = Counter()
review = {'heridal_car': [], 'visdrone_unlabelled': []}
used_sources = {'train': set(), 'val': set()}


def starts(length):
    if length <= TILE:
        return [0]
    return list(range(0, length - TILE, STRIDE)) + [length - TILE]


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def check_predictions(image):
    r = checker.predict(image, conf=.40, classes=[0, 2], verbose=False, device='cpu', imgsz=1920)[0]
    return [(COCO_NAMES[int(b.cls.item())], float(b.conf.item()), b.xyxy[0].tolist()) for b in r.boxes]


def write_tiles(image, objects, masks, flags, source, split, kind):
    """objects: (class, box); masks: boxes to fill; flags: boxes that exclude a tile."""
    h, w = image.shape[:2]
    positives, negatives = [], []
    for y in starts(h):
        for x in starts(w):
            tile = image[y:y + TILE, x:x + TILE].copy()
            th, tw = tile.shape[:2]
            fill = tile.reshape(-1, 3).mean(axis=0)
            labels, flagged = [], False
            for box in masks:
                cx1, cy1 = max(box[0] - x, 0), max(box[1] - y, 0)
                cx2, cy2 = min(box[2] - x, tw), min(box[3] - y, th)
                if cx2 > cx1 and cy2 > cy1:
                    tile[int(cy1):int(np.ceil(cy2)), int(cx1):int(np.ceil(cx2))] = fill
            for box in flags:
                if min(box[2] - x, tw) > max(box[0] - x, 0) and min(box[3] - y, th) > max(box[1] - y, 0):
                    flagged = True
            for name, box in objects:
                cx1, cy1 = max(box[0] - x, 0), max(box[1] - y, 0)
                cx2, cy2 = min(box[2] - x, tw), min(box[3] - y, th)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                inside = (cx2 - cx1) * (cy2 - cy1) / ((box[2] - box[0]) * (box[3] - box[1]))
                if inside >= .5 and cx2 - cx1 >= 4 and cy2 - cy1 >= 4:
                    labels.append(f'{CLS[name]} {(cx1 + cx2) / 2 / tw:.6f} {(cy1 + cy2) / 2 / th:.6f} '
                                  f'{(cx2 - cx1) / tw:.6f} {(cy2 - cy1) / th:.6f}')
                else:
                    tile[int(cy1):int(np.ceil(cy2)), int(cx1):int(np.ceil(cx2))] = fill
            item = (f'{source}_{x}_{y}', tile, labels)
            if flagged:
                stats[f'{kind}_{split}_excluded_tiles'] += 1
                review['heridal_car' if kind == 'heridal' else 'visdrone_unlabelled'].append(
                    cv2.resize(tile, (160, 160)))
                continue
            (positives if labels else negatives).append(item)
    keep = positives + rng.sample(negatives, min(len(negatives), round(NEG_RATIO * max(1, len(positives)))))
    for name, tile, labels in keep:
        cv2.imwrite(str(out / 'images' / split / f'{name}.jpg'), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / 'labels' / split / f'{name}.txt').write_text('\n'.join(labels))
        stats[f'{kind}_{split}_tiles'] += 1
        stats[f'{kind}_{split}_positive_tiles'] += bool(labels)
        for line in labels:
            stats[f'{kind}_{split}_{"person" if line[0] == "0" else "car"}_labels'] += 1


for split in ('train', 'val'):
    for sub in ('images', 'labels'):
        (out / sub / split).mkdir(parents=True, exist_ok=True)

# HERIDAL: persons only; cars are not annotated in this dataset
heridal = [(p, 'train') for p in sorted((raw / 'heridal_train').rglob('*.jpg'))] + \
          [(p, 'val') for p in sorted(heridal_val.rglob('*.jpg'))]
for path, split in heridal:
    assert not path.stem.startswith('test_'), 'HERIDAL test images must not be used'
    location = path.stem.split('_')[1]
    assert (location in ('BLA', 'RAK')) == (split == 'val'), path.stem
    xml = next(path.parent.parent.rglob(path.stem + '.xml'), None)
    if xml is None:  # no annotation file: unknown content, so never used as an empty image
        stats[f'heridal_{split}_images_without_annotation_excluded'] += 1
        continue
    if path.stem in EXCLUDE:  # unlabelled cars found in the visual review
        stats[f'heridal_{split}_images_excluded_unlabelled_car'] += 1
        continue
    original = cv2.imread(str(path))
    s = HERIDAL_WIDTH / original.shape[1]
    image = cv2.resize(original, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    objects = [('person', [float(o.find('bndbox').findtext(k)) * s for k in ('xmin', 'ymin', 'xmax', 'ymax')])
               for o in ET.parse(xml).getroot().iter('object')]
    stats[f'heridal_{split}_images'] += 1
    used_sources[split].add(path.stem)
    write_tiles(image, objects, [], [], path.stem, split, 'heridal')
    print('heridal', split, path.stem, flush=True)

# VisDrone: sequence-split subset; persons and cars kept together
manifest = json.loads((raw / 'visdrone' / 'visdrone_subset_manifest.json').read_text())
for split in ('train', 'val'):
    for member in manifest['images_by_split'][split]:
        path = raw / 'visdrone' / member
        image = cv2.imread(str(path))
        ann = Path(str(path).replace('images', 'annotations').replace('.jpg', '.txt'))
        objects, masks, labelled = [], [], []
        for line in ann.read_text(encoding='utf-8-sig').splitlines():
            x, y, w, h, score, cat = map(int, line.strip().rstrip(',').split(',')[:6])
            box = [x, y, x + w, y + h]
            labelled.append(box)
            if cat == 0 or score == 0:
                masks.append(box)
            elif cat in GT_MAPPING:
                objects.append((GT_MAPPING[cat], box))
        unlabelled = [box for name, conf, box in check_predictions(image)
                      if conf >= .60 and all(iou(box, b) < .10 for b in labelled)]
        stats[f'visdrone_{split}_images'] += 1
        used_sources[split].add(path.stem)
        write_tiles(image, objects, masks, unlabelled, path.stem, split, 'visdrone')
        print('visdrone', split, path.stem, flush=True)

assert not used_sources['train'] & used_sources['val'], 'source image in both splits'
names = '\n'.join(f'  {i}: {COCO_NAMES[i]}' for i in range(len(COCO_NAMES)))
(out / 'data.yaml').write_text(f'path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n')
for key, tiles in review.items():
    if tiles:
        tiles = tiles[:120] + [np.zeros_like(tiles[0])] * ((-len(tiles[:120])) % 10)
        cv2.imwrite(str(out.parent / f'review_{key}.jpg'),
                    np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))
(out / 'prep_manifest.json').write_text(json.dumps({
    'rules': __doc__, 'seed': SEED, 'counts': dict(sorted(stats.items())),
    'sources': {k: sorted(v) for k, v in used_sources.items()}}, indent=1))
print(json.dumps(dict(sorted(stats.items())), indent=1))
