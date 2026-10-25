# Model setup

DALEEL uses two detection models, YOLOv8n and YOLOv8s, fine-tuned for person and car detection
in aerial search-and-rescue imagery (see [DATASET_EVALUATION.md](DATASET_EVALUATION.md)). They
are not stored in the repository; they are attached to the project's GitHub Release
`v2.0.0`, and `scripts/setup_weights.py` downloads and verifies them. The application itself
never downloads model files, and it runs only these two files: there is no fallback to the
general pretrained models.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `yolov8n_sar_v1.pt` | 6,566,033 | `21f16d2d72ec76bf020eeeddbc3b59836fa534af43fc88ec3723ce2b7af72e7d` |
| `yolov8s_sar_v1.pt` | 22,605,073 | `90cda60aea2091c62a388954fe192104f9c5dbce96397bee00cbf4d650c40af4` |

Run the commands below from the project folder (the folder that contains `run.py`) after
installing the dependencies.

## Download and verify

```powershell
.\.venv\Scripts\python.exe scripts/setup_weights.py
```

Missing files are downloaded from the release and kept only if their size and SHA-256 hash
match. An existing file is never replaced; if it does not match, the script stops and reports
the difference.

## Use existing model files

Copy `yolov8n_sar_v1.pt` and `yolov8s_sar_v1.pt` into `weights/`, then verify them without
network access or modification:

```powershell
.\.venv\Scripts\python.exe scripts/setup_weights.py --verify-only
```

`--weights-dir` selects a different folder for either command.

## Check that both models load

```powershell
.\.venv\Scripts\python.exe scripts/setup_weights.py --verify-only --load-check
```

This loads both files through DALEEL's detector with network access blocked. No inference
is performed.

## Model locations and settings

By default, the application reads `weights/yolov8n_sar_v1.pt` and `weights/yolov8s_sar_v1.pt`
relative to the project folder. The environment variables `DALEEL_WEIGHTS_N` and
`DALEEL_WEIGHTS_S` can point to other locations. Before each mission is analyzed, both files are
checked against the sizes and hashes above; a mission whose model files do not match fails
with a message naming the file. The `weights/` folder and `*.pt` files are excluded from version
control.

The detection settings are fixed: both models analyze every 15th video frame at an input size
of 1280 pixels and keep detections with a confidence above 0.42. `DALEEL_IMGSZ` (640, 960 or
1280) exists for experimentation only; the evaluation in DATASET_EVALUATION.md applies to 1280.
