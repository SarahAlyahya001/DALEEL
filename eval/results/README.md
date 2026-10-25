# Evaluation results

This folder contains the published results of the detection evaluation described in
[`docs/DATASET_EVALUATION.md`](../../docs/DATASET_EVALUATION.md). All results come from a
single complete run over the 548 images of the VisDrone2019-DET validation split.

| File | Contents |
| --- | --- |
| `detection_metrics.json` | Complete metrics for every output and pass, with the configuration, ignore counts, detection-cap hits, degenerate predictions and timing summaries |
| `detection_metrics.csv` | Per-class counts, precision, recall, F1 and AP for the ranking and operational passes |
| `environment.json` | Hardware, software and session records of the run |
| `dataset_manifest.json` | Size and SHA-256 of every evaluated image and annotation file |
| `per_image_metrics.jsonl.gz` | Per-image predictions, processed target annotations and timings |
| `precision_recall_curves/` | Precision–recall curves at IoU 0.50 for YOLOv8n (A) and YOLOv8s (B), as CSV and SVG |

The files contain dataset identifiers, relative manifest paths, metrics, curves and
reproducibility metadata. They contain no images, raw annotation files, model weights or
local paths.

## Per-image predictions

`per_image_metrics.jsonl.gz` is a gzip copy of the run's per-image export,
`per_image_metrics.jsonl`. Decompressed, the file is unchanged:

- Size: 59,537,770 bytes (about 56.8 MiB); compressed: 15,053,687 bytes.
- SHA-256 of the decompressed file:
  `91f752f1e14ca48bf63b5c6c52a5b645907e5a12f110fdf7342a967577ab62b1`.
- Produced by `python -B -X utf8 eval/benchmark.py run` with Python 3.10.11 and the
  recorded configuration, in run
  `8214c72b7390-2026-09-16T174353.397978_0000-610ad571`.

To restore the uncompressed file:

```powershell
python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('eval/results/per_image_metrics.jsonl.gz'), open('eval/results/per_image_metrics.jsonl', 'xb'))"
```

## How the results were produced and checked

The aggregate metrics, environment record, manifest and curves were computed from the run's
548 saved per-image checkpoints; the JSONL file is the per-image export of the same
checkpoints. The run directory (about 344 MB) is not included. With it, the outputs can be
regenerated without inference:

```powershell
python -B -X utf8 eval/benchmark.py aggregate --run-dir <run-directory> --output-dir <new-output-directory>
```

An offline regeneration of this kind, with network access blocked and the evaluation source
hashes verified, reproduced all 13 published files, including the JSONL, byte for byte.

Without the run directory, `python eval/verify_saved_results.py` re-scores the saved
predictions in the JSONL file and requires 52 counts to equal `detection_metrics.json`
exactly.
