"""Write the Colab fine-tuning notebook: make_notebook.py <out.ipynb> <zip_sha256> [n|s]

'n' (default) reproduces DALEEL_finetune_yolov8n.ipynb exactly as used for experiment 1. 's' writes
the YOLOv8s notebook for the two-model comparison (ACCEPTANCE_CRITERIA.md, addendum 2): same dataset
package and training arguments; the app's yolov8s.pt is uploaded to Drive next to the zip.
"""
import json
import sys

M = sys.argv[3] if len(sys.argv) > 3 else 'n'
APP_SHA256 = {'n': 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36',
              's': '1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a'}
cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': text.strip('\n').splitlines(True)})


if M == 's':
    md(r'''
# DALEEL - YOLOv8s fine-tuning for the two-model comparison (experiment 2)

This notebook fine-tunes the **same YOLOv8s weights that DALEEL uses** with exactly the data and
settings used for the fine-tuned YOLOv8n (experiment 1), so that the two models can be compared and
combined fairly. The YOLOv8n is not retrained.

**What it uses**
* `DALEEL_finetune/ft_dataset.zip` on your Google Drive - the same file as in experiment 1 (same
  training and validation tiles, same person/car labels).
* `DALEEL_finetune/yolov8s.pt` on your Google Drive - the app's YOLOv8s weights (checked by hash).
* A **free** Colab GPU. Nothing in this notebook buys compute units or starts paid resources.

**What it produces** (in `DALEEL_finetune/runs/yolov8s_sar_v1/` on your Drive)
* `weights/last.pt` (updated every epoch - used to resume), `weights/best.pt`, and snapshots
  `weights/epoch10.pt`, `epoch20.pt`, ...
* `results.csv` (per-epoch training and validation metrics) and `pilot_yolov8s.json`.

**How to run**: run the cells from top to bottom. Cell 3 restarts the session once after installing
the pinned libraries; after the restart continue with cell 4. If Colab disconnects during training,
reconnect and run cells 1, 2, 3, 4, 5, 6 and then cell 8 again: training resumes from the last epoch.
''')
else:
    md(r'''
# DALEEL - YOLOv8n fine-tuning for aerial person detection (experiment 1)

This notebook fine-tunes the **same YOLOv8n weights that DALEEL uses** on aerial search-and-rescue
imagery, so that persons seen from above are detected, while cars are kept.

**What it uses**
* `DALEEL_finetune/ft_dataset.zip` on your Google Drive (prepared on the laptop; contains the training
  tiles, the validation tiles, `data.yaml`, `prep_manifest.json` and the app's `weights/yolov8n.pt`).
* A **free** Colab GPU. Nothing in this notebook buys compute units or starts paid resources.

**What it produces** (in `DALEEL_finetune/runs/yolov8n_sar_v1/` on your Drive)
* `weights/last.pt` (updated every epoch - used to resume), `weights/best.pt`, and snapshots
  `weights/epoch10.pt`, `epoch20.pt`, ... for later selection on the laptop.
* `results.csv` (per-epoch training and validation metrics) and `pilot.json`.

**How to run**: run the cells from top to bottom. Cell 3 restarts the session once after installing
the pinned libraries; after the restart continue with cell 4. If Colab disconnects during training,
reconnect and run cells 1, 2, 4, 5 and then cell 8 again: training resumes from the last epoch.
''')

md('## 1. Check that a GPU is attached')
code(r'''
import subprocess, sys
print('Python', sys.version.split()[0])
r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit('No GPU. Use Runtime > Change runtime type > Hardware accelerator: T4 GPU (free), then run again.')
print('GPU:', r.stdout.strip())
if not (3, 9) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit('The pinned libraries need Python 3.9-3.12. Use Runtime > Change runtime type > '
                     'Runtime version and choose a version with Python 3.12 or 3.11, then run again.')
''')

md('## 2. Connect Google Drive and locate the dataset')
code(r'''
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
DRIVE_DIR = Path('/content/drive/MyDrive/DALEEL_finetune')
ZIP = DRIVE_DIR / 'ft_dataset.zip'
assert ZIP.exists(), f'Upload ft_dataset.zip to {DRIVE_DIR} first.'
print(ZIP, round(ZIP.stat().st_size / 2**20), 'MB')
# Integrity check against the checksum printed on the laptop (ft_dataset.sha256.txt)
import hashlib
h = hashlib.sha256()
with open(ZIP, 'rb') as f:
    for block in iter(lambda: f.read(1 << 24), b''):
        h.update(block)
EXPECTED_SHA256 = '__ZIP_SHA256__'
print('sha256', h.hexdigest())
assert h.hexdigest() == EXPECTED_SHA256, 'ft_dataset.zip differs from the laptop copy: upload it again.'
''' + ('''assert (DRIVE_DIR / 'yolov8s.pt').exists(), f'Upload yolov8s.pt to {DRIVE_DIR} first.'
''' if M == 's' else ''))

md('''
## 3. Install the library versions used by the DALEEL app (run once per new session)
The same versions as the laptop (`requirements.txt`) are used so that the trained weights load in
the app. The session restarts automatically at the end of this cell; then continue with cell 4.
''')
code(r'''
import os, importlib.util
need = True
try:
    import torch, ultralytics, numpy
    need = not (torch.__version__.startswith('2.3.1') and ultralytics.__version__ == '8.2.48'
                and numpy.__version__ == '1.26.4')
except Exception:
    pass
if need:
    !pip -q install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    !pip -q install ultralytics==8.2.48 numpy==1.26.4
    print('Installed. Restarting the session - continue with cell 4 afterwards.')
    os.kill(os.getpid(), 9)
else:
    print('Pinned versions already installed.')
''')

md('## 4. Verify versions and GPU')
code(r'''
import torch, ultralytics, numpy, sys
print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| numpy', numpy.__version__)
assert torch.__version__.startswith('2.3.1') and ultralytics.__version__ == '8.2.48' and numpy.__version__ == '1.26.4'
assert torch.cuda.is_available(), 'CUDA is not available'
print('GPU:', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GB')
''')

md('## 5. Copy and check the dataset')
CELL5 = r'''
import hashlib, json, shutil, zipfile
from pathlib import Path
DRIVE_DIR = Path('/content/drive/MyDrive/DALEEL_finetune')
assert DRIVE_DIR.exists(), 'Google Drive is not mounted: run cell 2 again, then this cell.'
DS = Path('/content/ds')                      # fixed path: resume relies on it
if not (DS / 'data.yaml').exists():
    shutil.copy(DRIVE_DIR / 'ft_dataset.zip', '/content/ft_dataset.zip')
    with zipfile.ZipFile('/content/ft_dataset.zip') as z:
        z.extractall(DS)
# absolute dataset path for Ultralytics
text = (DS / 'data.yaml').read_text().splitlines()
text = [f'path: {DS}' if line.startswith('path:') else line for line in text]
(DS / 'data.yaml').write_text('\n'.join(text) + '\n')

# same starting weights as the app (hash recorded in the DALEEL VisDrone evaluation)
APP_N_SHA256 = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
WEIGHTS = DS / 'weights' / 'yolov8n.pt'
assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == APP_N_SHA256, 'weights differ from the app'

for split in ('train', 'val'):
    imgs = list((DS / 'images' / split).glob('*.jpg'))
    labels = list((DS / 'labels' / split).glob('*.txt'))
    classes = {line.split()[0] for p in labels for line in p.read_text().splitlines() if line.strip()}
    assert classes <= {'0', '2'}, classes
    print(f'{split}: {len(imgs)} tiles, {len(labels)} label files, classes {sorted(classes)}')
m = json.loads((DS / 'prep_manifest.json').read_text())
assert not set(m['sources']['train']) & set(m['sources']['val']), 'source image in both splits'
assert not any(s.startswith('test_') for s in m['sources']['train'] + m['sources']['val'])
print(json.dumps(m['counts'], indent=1))
'''
if M == 's':
    CELL5 = CELL5.replace(
        """APP_N_SHA256 = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
WEIGHTS = DS / 'weights' / 'yolov8n.pt'
assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == APP_N_SHA256, 'weights differ from the app'""",
        f"""APP_S_SHA256 = '{APP_SHA256['s']}'
WEIGHTS = DS / 'weights' / 'yolov8s.pt'
if not WEIGHTS.exists():
    shutil.copy(DRIVE_DIR / 'yolov8s.pt', WEIGHTS)
assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == APP_S_SHA256, 'yolov8s.pt differs from the app: upload it again'""")
    assert 'APP_S_SHA256' in CELL5
code(CELL5)

md('''
## 6. Training settings
Fine-tuning starts from the app's COCO YOLOv8n and keeps all 80 COCO classes, so the detection head is
not replaced; only persons (class 0) and cars (class 2) are labelled. `flipud` is enabled because aerial
images have no fixed "up". Validation during training uses the validation **tiles**; the final choice of
checkpoint, inference mode and threshold is made on the laptop with **complete** validation images.
''')
code(r'''
RUN_NAME = 'yolov8n_sar_v1'
PROJECT = str(DRIVE_DIR / 'runs')            # on Drive: checkpoints survive disconnects
EPOCHS = 100                                 # may be reduced after the pilot (cell 7)
TRAIN_ARGS = dict(data=str(DS / 'data.yaml'), imgsz=640, batch=16, epochs=EPOCHS, patience=25,
                  optimizer='SGD', lr0=0.01, cos_lr=False, flipud=0.5, fliplr=0.5, mosaic=1.0,
                  close_mosaic=10, seed=0, deterministic=True, workers=2, save_period=10,
                  project=PROJECT, name=RUN_NAME, exist_ok=True, plots=True)
''')

md('''
## 7. Pilot: time one epoch (not saved to Drive)
Measures the real speed of this GPU and prints the estimated time for the full run. If the estimate
exceeds your free-session time, training still works: it resumes after reconnecting (cell 8).
''')
code(r'''
import time, json
from ultralytics import YOLO
t0 = time.time()
YOLO(str(WEIGHTS)).train(**dict(TRAIN_ARGS, epochs=1, project='/content/pilot', name='pilot',
                                save_period=-1, plots=False))
epoch_s = time.time() - t0
estimate_h = epoch_s * EPOCHS / 3600
pilot = {'gpu': torch.cuda.get_device_name(0), 'epoch_seconds': round(epoch_s, 1),
         'planned_epochs': EPOCHS, 'estimated_hours_full_run': round(estimate_h, 2)}
(DRIVE_DIR / 'pilot.json').write_text(json.dumps(pilot, indent=1))
print(json.dumps(pilot, indent=1))
print('Early stopping (patience 25) may end the run sooner.')
''')

md('''
## 8. Full training (resumes automatically)
If `last.pt` already exists on Drive for this run, training continues from it; otherwise it starts
from the app's weights. Re-run this cell after any disconnect (after cells 1, 2, 4 and 5).
''')
code(r'''
from ultralytics import YOLO
last = Path(PROJECT) / RUN_NAME / 'weights' / 'last.pt'
if last.exists():
    print('Resuming from', last)
    try:
        YOLO(str(last)).train(resume=True)
    except AssertionError as e:          # raised when the run has already finished
        print('Training already finished - nothing to resume. Continue with cell 9.', e)
else:
    YOLO(str(WEIGHTS)).train(**TRAIN_ARGS)''')

md('''
**Note on free GPU time:** the free tier limits session length and daily GPU use. If Colab says the
GPU is unavailable, wait (usually several hours) and resume later; do not buy compute units.
''')

md('## 9. Summary and files to download')
code(r'''
import pandas as pd, hashlib
run = Path(PROJECT) / RUN_NAME
df = pd.read_csv(run / 'results.csv'); df.columns = [c.strip() for c in df.columns]
best = df.loc[df['metrics/mAP50(B)'].idxmax()]
print('epochs completed:', len(df))
print('best tile-validation epoch:', int(best['epoch']), '| mAP50', round(best['metrics/mAP50(B)'], 4),
      '| precision', round(best['metrics/precision(B)'], 4), '| recall', round(best['metrics/recall(B)'], 4))
for p in sorted((run / 'weights').glob('*.pt')):
    print(f'{p.name:12s} {round(p.stat().st_size / 2**20, 1):6.1f} MB  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}')
print('\nDownload the whole folder', run, '(right-click > Download in the Drive web page).')
''')

if M == 's':  # same cells; only the model, run name, fixed epoch count and pilot file differ
    swaps = [("starts from the app's COCO YOLOv8n", "starts from the app's COCO YOLOv8s"),
             ("RUN_NAME = 'yolov8n_sar_v1'", "RUN_NAME = 'yolov8s_sar_v1'"),
             ("EPOCHS = 100                                 # may be reduced after the pilot (cell 7)",
              "EPOCHS = 60                                  # same as the fine-tuned YOLOv8n (pre-registered)"),
             ("(DRIVE_DIR / 'pilot.json')", "(DRIVE_DIR / 'pilot_yolov8s.json')"),
             ("If the estimate\nexceeds your free-session time, training still works",
              "If the estimate\nexceeds your free-session time, training still works over several sessions")]
    for old, new in swaps:
        hits = 0
        for c in cells:
            text = ''.join(c['source'])
            if old in text:
                hits += 1
                c['source'] = text.replace(old, new).splitlines(True)
        assert hits == 1, old
if len(sys.argv) > 2:  # fill in the dataset checksum
    for c in cells:
        c['source'] = [line.replace('__ZIP_SHA256__', sys.argv[2]) for line in c['source']]
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('cells:', len(cells))
