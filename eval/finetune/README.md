# Fine-tuning and evaluation records

The training data, protocol and results are explained in
[docs/DATASET_EVALUATION.md](../../docs/DATASET_EVALUATION.md). This folder keeps the material needed
to check or repeat that work. The datasets themselves are not included; obtain HERIDAL
(https://zenodo.org/records/5662351) and VisDrone2019-DET (https://github.com/VisDrone/VisDrone-Dataset)
under their terms.

| Path | Contents |
| --- | --- |
| `PROTOCOL.md` | Acceptance criteria and comparison rules, each part written before the results it governs (references to the internal experiment log point to the dated lab record) |
| `splits/` | VisDrone training subset and held-out validation sequences, and the visual review of HERIDAL images with unannotated cars |
| `notebooks/` | The Google Colab notebooks used to train YOLOv8n and YOLOv8s (free tier, Tesla T4) |
| `scripts/prepare_dataset.py`, `make_notebook.py` | Building the 640 × 640 training tiles and the notebooks |
| `scripts/val_run.py` | Saving each model's predictions on complete images (confidence above 0.001) |
| `scripts/ens_eval.py`, `review_order.py`, `decompose.py` | The one-model and two-model comparison, the review-order analysis, and the overlap explanation |
| `scripts/parity.py`, `system_eval.py` | The check that the application reproduces the evaluated configuration, and the final evaluation |
| `results/` | Machine-readable results, including `runtime/` (complete application, 60-second video) and `journey.json` (upgrade and first-use check) |

Scripts that import the application expect `DALEEL_REPO` to point to the project folder.
