# Detection models: training and evaluation

DALEEL runs two detection models, YOLOv8n and YOLOv8s, on every 15th video frame and keeps the
person and car detections of both. This document describes how the two models were fine-tuned,
how the integrated configuration was chosen and evaluated, and, for reference, the evaluation of
the original pretrained configuration that motivated the fine-tuning.

## Summary

| Measure | Original configuration (pretrained) | DALEEL version 2 (both fine-tuned models) |
| --- | ---: | ---: |
| HERIDAL test persons found (recall)¹ | 14 of 337 (4.15%) | 216 of 337 (64.1%) |
| HERIDAL test person precision | 70.0% | 78.8% |
| HERIDAL test false person detections per image | 0.06 | 0.57 |
| HERIDAL test person AP@0.5 | 16.05% | 71.2% |
| VisDrone validation car AP@0.5 | 55.27% (YOLOv8n) | 85.0% |
| 60-second 1080p video, complete application, system-test laptop | 150.0 s, 656 MB | 163.1 s, 650 MB |

¹ The original configuration analyzed the full-resolution HERIDAL images. Version 2 analyzed images
first resized to 1,920 px wide. With that same resizing, the original YOLOv8n found 13 of 337
persons (see [Final evaluation](#final-evaluation-of-daleel-version-2)).

- Most of the gain over the original configuration comes from fine-tuning: in its own test run the
  fine-tuned YOLOv8n alone found 200 of the 337 persons (59.3%). The configurations are compared on
  validation data, not on the test split.
- Both models analyze every 15th frame with one confidence threshold (above 0.42). Overlapping
  detections of the same class (IoU ≥ 0.50, one-to-one) become one candidate labeled
  **Both models**; every other qualifying detection of either model is kept as **One model**.
- On the validation images, the combination did not show a recall advantage over the better
  single model at the same false-positive rate, and it costs more computation than either model.
- Detections reported by both models were correct more often than detections reported by one
  model (validation: 65 of 77 vs 11 of 57; test: 156 of 177 vs 60 of 97). These are
  group-specific results, not a calibrated probability for an individual observation.

## Training data and splits

The roles of the images were fixed before training:

- **HERIDAL validation:** the two locations BLA and RAK. They occur in no other role: no BLA or RAK
  image was trained on, and neither location is in the test split.
- **HERIDAL training and test:** these follow the dataset's own train/test split, which is not
  separated by location. Eight of the 12 test locations (BRA, BRK, BRS, CAP, MED, SB, TRS, ZRI)
  also occur, with different images, among the training images. Only BLI, CAB, GRO and MOB are
  test-only locations.
- **VisDrone:** the 208 sequences of the training split were divided by sequence into 166 training
  and 42 held-out sequences (no overlap), so the held-out vehicle check uses unseen sequences.
  The official validation split (548 images, 76 sequence numbers) was not used for training or
  selection. However, 19 of its 76 sequence numbers also occur among the 166 training sequences
  (and 5 among the 42 held-out ones). A visual check of four such sequences showed the same scenes,
  so the final vehicle evaluation includes scenes that were also trained on. The sequence lists are in `eval/finetune/splits/`.

| Source | Role | Images | Annotated targets |
| --- | --- | ---: | --- |
| HERIDAL training split, without locations BLA and RAK | Training | 1,321 (3,503 tiles) | 4,229 persons in tiles |
| VisDrone2019-DET training split, 166 of 208 sequences, at most 6 frames each | Training | 768 (3,877 tiles) | 35,367 persons, 35,580 cars in tiles |
| HERIDAL training split, locations BLA and RAK | Validation: thresholds and model comparison | 93 | 86 persons |
| VisDrone2019-DET training split, the other 42 sequences | Validation: vehicle check | 194 | persons and cars |
| HERIDAL test split | Final evaluation, persons | 101 | 337 persons |
| VisDrone2019-DET validation split | Final evaluation, vehicles | 548 | 13,980 cars, 13,922 persons |

- HERIDAL annotates persons only. Of 1,490 downloaded training images outside BLA and RAK, 37
  without an annotation file and 132 that visibly contain unannotated cars were excluded; car
  predictions in the 10 validation images that contain real cars are not counted as false alarms.
- VisDrone *pedestrian* and *people* are mapped to *person* and *car* to *car*; ignored regions are
  masked. The 548-image validation split was not used for training or selection.
- Training images were resized to 1920 px wide (HERIDAL, the size of a 1080p frame) or kept at their
  native size (VisDrone) and cut into 640 × 640 tiles with 25% overlap.

**Training.** Both models were fine-tuned from the COCO-pretrained Ultralytics weights with the COCO
class list kept (person = 0, car = 2), with identical settings: 60 epochs, image size 640, batch 16,
SGD (lr0 0.01), vertical flip 0.5, mosaic closed for the last 10 epochs, seed 0. Training ran on the
free tier of Google Colab (NVIDIA Tesla T4): about 3.5 hours for YOLOv8n and 4 hours for YOLOv8s.
Ultralytics 8.2.48, PyTorch 2.3.1. The released files are the epoch-60 weights re-saved without the
`dill` module, with identical tensors.

## Selection protocol

The acceptance criteria and the comparison rules were written before the corresponding training
runs. Every configuration has one tuned parameter, its confidence threshold, chosen on the HERIDAL
validation images as the highest person recall with precision of at least 50% (grid 0.01 to 0.90).
Acceptance: recall ≥ 50%, precision ≥ 50%, at most 1.0 false positive per image (person false
positives plus car predictions), the processing-time and memory limits of NFR-01 and NFR-02, and car
AP@0.5 not more than 2 points below the pretrained YOLOv8n on the 194 held-out VisDrone images.

Matching follows the repository protocol below (one-to-one, IoU ≥ 0.50). HERIDAL images are
resized to 1920 px wide, the size of a 1080p frame, and analyzed as full frames at 1,280 px.

### Dual-model merging and scoring protocol

1. Each model predicts persons (COCO class 0) and cars (class 2) on the full frame at imgsz 1280 with
   its own NMS (IoU 0.7, at most 300 detections).
2. Each model keeps the detections with confidence above the threshold *t*; within each class they
   are ordered by descending confidence.
3. For each class, all YOLOv8n-YOLOv8s pairs with IoU ≥ 0.50 are sorted by descending IoU (ties by
   the two list positions) and accepted one-to-one. A pair becomes one candidate with the box of the
   higher-scoring model (YOLOv8n on an exact tie), score = the higher score, and both model scores;
   its label is **Both models**.
4. Every unpaired detection of either model is kept with its own score; its label is **One model**.
5. Operating point: *t* = 0.42. Precision, recall and false positives per image are counted at this
   threshold. HERIDAL annotates persons only. On the validation images, false positives per image
   follow the acceptance criterion: person false positives plus car predictions outside the 10 images
   that show real vehicles (none occurred at 0.42). On the test split, false positives are person
   false positives, and car predictions are reported separately.
6. AP@0.5: steps 1 to 4 with *t* = 0.001, ranked by the merged score, matched one-to-one at IoU ≥ 0.50
   and summarized with the 101-point interpolated precision-recall curve (`average_precision` in
   `eval/detection_metrics.py`). Zero-area boxes count as false positives.

Steps 2 to 4 are `fuse()` in `daleel/pipeline/detector.py`, the code the application runs. To repeat
the final evaluation (with the datasets prepared as for training and `DALEEL_REPO` set to this
folder): `eval/finetune/scripts/val_run.py` saves each model's predictions at confidence 0.001,
`ens_eval.py` produces the validation comparison, `parity.py` checks the application against it,
and `system_eval.py` runs the application's detector on the test or VisDrone validation images
(operating point and AP in one run).

## Comparison of one and two fine-tuned models (HERIDAL validation, 93 images, 86 persons)

| Configuration | Threshold | Recall | Precision | FP per image | AP@0.5 | Recall at 0.50 FP per image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n alone | 0.27 | 87.2% | 55.6% | 0.65 | 72.8% | 84.9% |
| YOLOv8s alone | 0.46 | 87.2% | 67.0% | 0.40 | 81.4% | 87.2% |
| Both models (DALEEL version 2) | 0.42 | 88.4% | 56.7% | 0.62 | 79.9% | 83.7% |

- At 0.50 false positives per image, both models together minus YOLOv8s alone: -3.5 percentage
  points, 95% paired bootstrap interval [-8.1, 0.0] (2,000 resamples of the 93 images). YOLOv8s
  alone minus YOLOv8n alone: +2.3 points, [-3.3, +8.1]. Neither difference is demonstrated.
- At their own thresholds, the two models found 72 persons in common and 3 each that the other
  missed.
- Model agreement within the combination: Both models 65 of 77 correct (84.4%, Wilson 95% interval
  74.7% to 90.9%); One model 11 of 57 (19.3%, 11.1% to 31.3%).
- An offline follow-up ranked the same candidates for review. Agreement-first ordering found correct
  targets slightly sooner than confidence-only ordering, but not sooner than YOLOv8s alone ranked
  by confidence. DALEEL therefore keeps the chronological order of observations.
- Car check on the 194 held-out VisDrone images (AP@0.5): YOLOv8n 76.6%, YOLOv8s 80.6%, both
  79.8%, pretrained YOLOv8n 48.2%.

## Final evaluation of DALEEL version 2

The settings were fixed before this run. The HERIDAL test split was not held back for this run
alone, and its earlier use is as follows:

1. **Original pretrained configuration.** The test split was used to evaluate it (14 of 337
   persons). That result, and an inspection of the persons the pretrained models missed, led to
   the decision to fine-tune.
2. **Exploratory tiling run.** It was set aside unread (`PROTOCOL.md`).
3. **Fine-tuned YOLOv8n alone.** Recall 59.3%, precision 76.6% at 0.27.
4. **This configuration.** The run reported below.

Thresholds and the choice between configurations used the validation images only, and no setting
was changed because of a test result. The test figures are therefore less independent than results
on never-used images.

**Image preparation.** For the fine-tuned runs, each HERIDAL test image is resized to 1,920 px wide
(a 1080p frame) and then analyzed at imgsz 1280. The original configuration's reference result
(14 of 337) analyzed the 4,000-px images directly at imgsz 1280. Run through the same 1,920-px
preparation, the pretrained YOLOv8n found 13 of 337 persons:

| Measure | Value |
| --- | ---: |
| Precision | 76.5% |
| Person false positives per image | 0.04 |
| AP@0.5 | 13.9% |

Before the run, the application's own detector (`daleel/pipeline/detector.py`) was checked against
the evaluated configuration on all validation images, candidate by candidate (class, box, scores and
agreement label), with no difference:

| Validation set | Candidates | Scored | Not scored |
| --- | ---: | --- | --- |
| HERIDAL BLA + RAK (93 images) | 144 | 76 correct and 58 false person detections | 10 car detections, all in the validation images that show real vehicles (HERIDAL annotates persons only) |
| VisDrone held-out (194 images) | 8,471 | 8,022: persons 2,695 correct, 582 false; cars 3,950 correct, 795 false | 449 candidates in VisDrone ignored regions (155 person, 294 car) |

The 8,022 scored candidates are also the candidate list of the review-order analysis.

**Persons, HERIDAL test split (101 images, 337 persons).** False positives are persons only; HERIDAL
has no vehicle annotations, so the 11 car predictions are not scored.

| | TP | FP | FN | Precision | Recall | FP per image | AP@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DALEEL version 2 | 216 | 58 | 121 | 78.8% | 64.1% | 0.57 | 71.2% |
| Both models agree | 156 correct | 21 false | | 88.1% (82.5% to 92.1%) | | | |
| One model | 60 correct | 37 false | | 61.9% (51.9% to 70.9%) | | | |

**Vehicles, VisDrone2019-DET validation split (548 images).**

| Class | TP | FP | FN | Precision | Recall | AP@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cars, DALEEL version 2 | 11,235 | 2,038 | 2,745 | 84.6% | 80.4% | 85.0% |
| Persons, DALEEL version 2 | 7,306 | 1,429 | 6,616 | 83.6% | 52.5% | 64.3% |
| Cars, original pretrained YOLOv8n (reference) | 6,345 | 1,700 | 7,635 | 78.9% | 45.4% | 55.3% |

Model agreement on cars: Both models 9,748 of 10,622 correct (91.8%), One model 1,487 of 2,651
(56.1%). The VisDrone training sequences come from the same camera platforms and cities as the
validation split, so part of the vehicle result reflects this similarity.

AP is computed with the fusion at confidence 0.001, ranked by the fused score.

## Processing time and memory

The same 60-second 1080p test video (120 sampled frames; built from the demonstration flight by
`scripts/measure_runtime.py`, SHA-256 `ad964076...0a92`) was processed by each configuration through
the complete application, one after another, on the system-test laptop (Intel Core i7-1355U, 16 GB
RAM, Windows 11, Python 3.10.11; other applications open, background load not controlled). Time is
measured from submitting the upload form until the status reports Complete (Waitress server, sign-in,
upload, MP4 validation, analysis, evidence images, storage; the browser is not included); memory is
the peak working set of the server process.

| Configuration | Model calls | Submission to Complete | Peak memory |
| --- | --- | ---: | ---: |
| Original pretrained configuration (2026-10-14 release) | 120 + 105 | 150.0 s | 656 MB |
| Fine-tuned YOLOv8n alone (> 0.27) | 120 | 61.6 s | 565 MB |
| Fine-tuned YOLOv8s alone (> 0.46) | 120 | 122.7 s | 641 MB |
| DALEEL version 2, both models (> 0.42) | 120 + 120 | 163.1 s | 650 MB |

The single-model rows use the application's own code with only the detector replaced, for this
comparison. `scripts/measure_runtime.py`, run as documented, measured 162.1 s and 650 MB for version 2.
NFR-01 requires five minutes on a laptop with a four-core x64 CPU and 8 GB RAM. Scaled by the speed
ratio of about 1.55 measured between the system-test laptop and the evaluation laptop (Intel Core
i5-8265U, 8 GB), the estimate is about 253 s.

**NFR-01 and NFR-02 status:**

- **System-test laptop:** both are met by these measurements.
- **Minimum specification:** both remain unverified until `scripts/measure_runtime.py` has been
  run on a minimum-specification laptop. The 253 s figure is an estimate, not a measurement.

## Limitations

- HERIDAL shows Mediterranean wilderness and VisDrone mostly populated areas; neither contains
  desert scenes. A labelled desert evaluation is still needed.
- The validation set contains 86 persons; small differences between configurations cannot be
  resolved, and the test recall (64.1%) is lower than the validation recall (88.4%).
- The HERIDAL test split had been used during development (see above), and 8 of its 12 locations
  also occur in the training images.
- Part of the vehicle result reflects the similarity of the VisDrone training and validation
  scenes: 19 of the 76 validation sequence numbers also occur among the training sequences.
- The review-order analysis is an offline ranking, not a measurement of analyst time.

## Original pretrained configuration (reference evaluation)

Until version 2, DALEEL used the COCO-pretrained YOLOv8n (confidence above 0.30) on every sampled
frame and the pretrained YOLOv8s (above 0.50) only on frames in which YOLOv8n found a candidate,
labeling each YOLOv8n candidate Dual-Model or Screening-Only. The evaluation of that configuration
on the VisDrone validation split is kept below unchanged; its machine-readable results are in
[`eval/results/`](../eval/results/), and `eval/legacy_pipeline.py` keeps the original matching rule
so that the results remain reproducible. On search-and-rescue imagery that configuration found
14 of the 337 persons in the HERIDAL test split (4.15% recall), which is why the models were
fine-tuned.

### Summary

On the complete VisDrone2019-DET validation split, under the two-class protocol described
below:

- YOLOv8s achieved higher average precision than YOLOv8n for both classes, with a mean
  inference time per image roughly 2.4 times longer.
- At the application's thresholds, YOLOv8n (DALEEL's candidate source) reached 80.23%
  precision and 32.25% recall; YOLOv8s alone reached 87.43% precision and 28.26% recall.
- Dual-Model candidates had a precision of 88.74%, compared with 60.35% for Screening-Only
  candidates. Because both groups are retained, DALEEL's complete candidate set and its
  metrics are identical to those of YOLOv8n.
- On these mostly populated scenes, running YOLOv8s only on frames with YOLOv8n candidates
  would have avoided 2.92% of YOLOv8s calls.

These results describe still-image detection. They do not measure tracking, telemetry
association, video sampling, mission processing time or performance across desert
conditions.

### Dataset

The benchmark is the complete validation split of VisDrone2019-DET: 548 RGB aerial images
and their annotations. It is a general aerial detection benchmark rather than a desert search
dataset. The data is not included in this repository. It can be obtained from the
[VisDrone authors](https://github.com/VisDrone/VisDrone-Dataset) or from the
[Ultralytics mirror](https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip).
Before inference, the evaluation verifies every image and annotation by size and SHA-256
against [`eval/results/dataset_manifest.json`](../eval/results/dataset_manifest.json)
(manifest digest `d6755dcbd5d0da44faf608c8efd2e20c2856b1df3637e0aa8cb0b0de480d6087`).

The VisDrone categories *pedestrian* (1) and *people* (2) are mapped to *person*, and *car*
(4) to *car*. Only the COCO classes *person* (0) and *car* (2) are predicted; other vehicle
categories are outside this evaluation.

**Attribution.** P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu and H. Ling, "Detection and
tracking meet drones challenge," *IEEE Transactions on Pattern Analysis and Machine
Intelligence*, 44(11), 7380–7399. Review the
[author repository](https://github.com/VisDrone/VisDrone-Dataset) and the
[dataset terms](https://aiskyeye.com/data-protection/) before use or redistribution. The
recorded terms describe academic use under CC BY-NC-SA 3.0; the terms page names VisDrone2021,
while the download page identifies the DET data as the same as VisDrone2019.

### Protocol

**Evaluated outputs.**

| Output | Definition |
| --- | --- |
| A | YOLOv8n alone, confidence above 0.30 |
| B | YOLOv8s alone, confidence above 0.50 |
| C / D | The Dual-Model and Screening-Only parts of A, using A's true and false positive assignments |
| E | DALEEL's deployed output: all YOLOv8n candidates with their support labels |

A separate ranking pass with confidence above 0.001 is used for average precision.

**Ground truth.** Boxes are clipped to the image bounds. Ground truth and predictions whose
area is at least 50% covered by the union of *ignored region* (category 0) rectangles are
excluded, without double-counting overlapping regions. Target annotations with score zero
are ignored: valid same-class ground truth is matched first, and an unmatched prediction may
then be ignored by a same-class score-zero annotation when its overlap with the prediction's
area reaches the evaluation IoU. Non-target categories are neither positives nor blanket
ignore regions.

**Matching.** Predictions are processed in descending confidence order and matched to the
unused same-class ground-truth box with the highest IoU. Stable image, prediction and
annotation identifiers break ties. A true positive requires IoU ≥ 0.50.

**Average precision.** AP is the precision envelope sampled at 101 recall points, at IoU
thresholds 0.50:0.05:0.95. Unreached recall contributes zero. AP for a class without ground
truth is undefined and excluded from macro averages.

**Operational support.** The Dual-Model label uses the application's one-to-one matching
between YOLOv8n and YOLOv8s boxes at IoU ≥ 0.50, with the YOLOv8n box as the canonical
box. C and D partition A's fixed assignments, so they have no separate false-negative
populations.

**Counting.** Ignored predictions are counted separately and excluded from precision
denominators. Undefined ratios are reported as null. Combined precision, recall and F1 are
computed from summed counts; macro values average the per-class metrics.

**Inference settings.** CPU, image size 1280, NMS IoU 0.7, at most 300 detections per image,
class-aware NMS, no test-time augmentation and no half precision. Both models ran on every
image so that each could be evaluated on its own. Each predict call was timed in full;
image decoding, serialization and support matching are excluded. Twenty warm-up calls (five
per model and pass) are excluded from timing. Per-image checkpoints are written atomically,
hash-checked and tied to a configuration fingerprint, and results are published only when
all 548 checkpoints exist.

This is a two-class adaptation, not the official ten-class VisDrone protocol.

### Results

#### Operational detection (IoU ≥ 0.50)

Eligible ground truth: 13,922 *person* and 13,980 *car* annotations (27,902 in total).
Ignored regions excluded 131 target annotations, and the split contains no score-zero target
annotations.

| Model and class | GT | Scored predictions | TP | FP | FN | Precision % | Recall % | F1 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n person | 13922 | 3171 | 2654 | 517 | 11268 | 83.70 | 19.06 | 31.05 |
| YOLOv8n car | 13980 | 8045 | 6345 | 1700 | 7635 | 78.87 | 45.39 | 57.62 |
| YOLOv8n combined | 27902 | 11216 | 8999 | 2217 | 18903 | 80.23 | 32.25 | 46.01 |
| YOLOv8s person | 13922 | 1897 | 1792 | 105 | 12130 | 94.46 | 12.87 | 22.66 |
| YOLOv8s car | 13980 | 7120 | 6092 | 1028 | 7888 | 85.56 | 43.58 | 57.74 |
| YOLOv8s combined | 27902 | 9017 | 7884 | 1133 | 20018 | 87.43 | 28.26 | 42.71 |

The two models use different thresholds (YOLOv8n above 0.30, YOLOv8s above 0.50), so the
higher precision of YOLOv8s at these operating points should not be read as a direct model
comparison; the average precision results below compare the models across all thresholds.

Raw predictions: 11,295 (YOLOv8n) and 9,053 (YOLOv8s). Ignored predictions: 79 for YOLOv8n
(16 person, 63 car) and 36 for YOLOv8s (5 person, 31 car). Macro precision/recall/F1:
81.28/32.22/44.33% for YOLOv8n and 90.01/28.22/40.20% for YOLOv8s. DALEEL's deployed output
(E) equals A in every count and metric.

#### Average precision (ranking pass, confidence above 0.001)

| Model and class | AP@0.50 % | mAP@0.50:0.95 % |
| --- | ---: | ---: |
| YOLOv8n person | 35.10 | 15.27 |
| YOLOv8n car | 55.27 | 35.76 |
| YOLOv8n macro | 45.18 | 25.51 |
| YOLOv8s person | 43.97 | 20.00 |
| YOLOv8s car | 62.15 | 42.14 |
| YOLOv8s macro | 53.06 | 31.07 |

E duplicates A by construction. The complete ranking counts, precision, recall, F1 and
per-IoU AP values are in `detection_metrics.json`; operational results are computed
separately, not by filtering the ranking outputs. The limit of 300 detections per image was
reached on 419 images for YOLOv8n and 345 images for YOLOv8s in the ranking pass, which limits
ranking recall and AP; neither operational model reached it. The internal NMS candidate limit
(30,000) is recorded but was not separately instrumented, and no NMS time-limit warning
occurred.

#### Support labels

| Class and support | Scored predictions | TP | FP | Precision % | Share of A TP % | Share of A FP % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Person, Dual-Model | 1692 | 1602 | 90 | 94.68 | 60.36 | 17.41 |
| Person, Screening-Only | 1479 | 1052 | 427 | 71.13 | 39.64 | 82.59 |
| Car, Dual-Model | 6162 | 5368 | 794 | 87.11 | 84.60 | 46.71 |
| Car, Screening-Only | 1883 | 977 | 906 | 51.89 | 15.40 | 53.29 |
| Combined, Dual-Model | 7854 | 6970 | 884 | 88.74 | 77.45 | 39.87 |
| Combined, Screening-Only | 3362 | 2029 | 1333 | 60.35 | 22.55 | 60.13 |

Dual-Model precision exceeded Screening-Only precision by 23.55 percentage points for
*person*, 35.23 for *car* and 28.39 combined. The combined figures exclude 24 ignored
Dual-Model and 55 ignored Screening-Only predictions; the raw group totals are 7,878 and
3,417. The support label therefore identifies a more precise subset of the candidates; it
does not make the complete candidate set more precise. Screening-Only candidates still
contain 2,029 true positives, which is why they are retained and shown to the analyst.

#### Inference time

| Pass and model | Wall time mean / median / P95 (s) | CPU time mean / median / P95 (s) |
| --- | --- | --- |
| Ranking, YOLOv8n | 0.682 / 0.610 / 1.172 | 0.669 / 0.609 / 1.120 |
| Ranking, YOLOv8s | 1.616 / 1.432 / 2.745 | 1.580 / 1.422 / 2.609 |
| Operational, YOLOv8n | 0.673 / 0.607 / 1.117 | 0.661 / 0.609 / 1.078 |
| Operational, YOLOv8s | 1.579 / 1.418 / 2.582 | 1.551 / 1.406 / 2.516 |

Each model was measured on all 548 images in each pass (1,096 measured calls per model),
after five excluded warm-up calls per model and pass. The 2,192 measured calls took
2,492.95 seconds of wall time and 2,444.53 seconds of process CPU time; the longest call took
3.96 seconds, and no warnings were raised. Wall time is the primary measure; CPU time is
reported for diagnosis. Consistency checks on timing units, operational box geometry, the
equality of A and E, and the support-partition counts all passed.

In deployment, YOLOv8s would have run on 532 of the 548 images (97.08%), avoiding 16 calls
(2.92%) compared with always running both models. Both models physically ran on every image
in this experiment, so this is a calculated saving, not a measured one. The resulting
conditional sum of predict times averages 2.202 seconds per image; it excludes support
matching, tracking and storage and is not an end-to-end mission time.

The complete inference phase, including warm-up, decoding and checkpointing, took 2,580.64
seconds (43.01 minutes) with a peak Windows working set of 539.75 MiB; memory used during the
subsequent aggregation is not included in that peak. The evaluation ran on an Intel Core
i5-8265U laptop CPU (4 cores, 8 logical processors) with 8 GB of RAM, on AC power with the
Balanced power plan, using Python 3.10.11, Ultralytics 8.2.48, PyTorch 2.3.1 and one PyTorch
inference thread. Background load and temperature were not controlled, so the times are
specific to that laptop and session.

#### Degenerate predictions

Degenerate boxes (zero or negative clipped area) keep their original values and count as
false positives at every AP threshold. They occurred only in the ranking pass: one *person*
prediction from YOLOv8n, and two *person* and thirteen *car* predictions from YOLOv8s, a
total of 16 predictions in 15 images. The operational outputs contain none. Coordinates,
scores and prediction identifiers are in `detection_metrics.json`.

| Image ID | Model | Class | Count |
| --- | --- | --- | ---: |
| 0000026_04500_d_0000033 | YOLOv8s | car | 1 |
| 0000069_02480_d_0000007 | YOLOv8s | person | 1 |
| 0000086_00000_d_0000001 | YOLOv8n | person | 1 |
| 0000163_00001_d_0000001 | YOLOv8s | car | 1 |
| 0000165_04325_d_0000105 | YOLOv8s | car | 1 |
| 0000194_00998_d_0000124 | YOLOv8s | car | 1 |
| 0000216_00520_d_0000001 | YOLOv8s | person | 1 |
| 0000244_03000_d_0000007 | YOLOv8s | car | 1 |
| 0000276_04401_d_0000529 | YOLOv8s | car | 2 |
| 0000280_02201_d_0000623 | YOLOv8s | car | 1 |
| 0000287_00601_d_0000762 | YOLOv8s | car | 1 |
| 0000335_00197_d_0000044 | YOLOv8s | car | 1 |
| 0000335_01765_d_0000052 | YOLOv8s | car | 1 |
| 0000335_02549_d_0000056 | YOLOv8s | car | 1 |
| 0000335_05097_d_0000069 | YOLOv8s | car | 1 |

### Reproducibility records

`detection_metrics.json` and `environment.json` record the full configuration, package
versions, hardware and the SHA-256 hashes of the four evaluation source files of the
2026-10-14 release. `eval/detection_metrics.py` and `eval/visdrone.py` are unchanged. The
support rule and extraction order recorded for `daleel/pipeline/detector.py` are kept unchanged
in `eval/legacy_pipeline.py`, because the application's detector now implements the dual-model
configuration. `eval/detection_run.py` has since gained the `--dataset` and `--output-dir`
options and reads the rule from `eval/legacy_pipeline.py`; the protocol and settings are
unchanged.

After the run, an offline aggregation with network access blocked read all 548 saved
checkpoints, verified the evaluation source hashes and regenerated every output. All 13
files matched the published files byte for byte. No models were loaded and no inference ran.
The run directory with its checkpoints and call logs (about 344 MB) is not included in the
repository. These hashes cover complete files; the manifest digest given under
[Dataset](#dataset) is a separate digest of the list of raw files.

| File under `eval/results/` | Bytes | SHA-256 |
| --- | ---: | --- |
| dataset_manifest.json | 237319 | `7765616bee595c840f14104c750c0b65a14adb6ee74da2f2b438a45278f8cfca` |
| detection_metrics.csv | 1556 | `84c38be9b5dbcf3eb9d6d6f6adeea3e160ec53f8b9bb943d0db448d78861bb59` |
| detection_metrics.json | 22751 | `22191e854023d3615c9cfd778d72b3506e15bc0744637b90cf22d88e42783409` |
| environment.json | 6971 | `8c961cd0f475900dd85af575370aa2e3d3df57fe83843ea6b1f28617ba053db9` |
| per_image_metrics.jsonl | 59537770 | `91f752f1e14ca48bf63b5c6c52a5b645907e5a12f110fdf7342a967577ab62b1` |
| precision_recall_curves/A_car.csv | 2271 | `729c8819304fb319eae086fa652ee283a772c401e18a07be3e10687e24a0274b` |
| precision_recall_curves/A_car.svg | 2123 | `26e92814d027fe2d3380153ac8bf018dce4e6f0f859f31d937b94c2494c65172` |
| precision_recall_curves/A_person.csv | 1979 | `dd1c340edb10939328da1447b58b26545e65aaac2a7a0a93688ab09ed7929366` |
| precision_recall_curves/A_person.svg | 2149 | `67956e481b92c8125f36ac059702d78d8ceb4a5da85834a87e93c9e2b8e1d184` |
| precision_recall_curves/B_car.csv | 2329 | `c9c6f5439e1f203f4bf1a4faeab130638265f978292ae2d85de889e924093055` |
| precision_recall_curves/B_car.svg | 2115 | `6d0aacc774d8be16e0821b025846fd201d4aad1efaa9dea454ae35af936bed36` |
| precision_recall_curves/B_person.csv | 2095 | `36d7b4c6e22bfa91499126b25c19ebd8bc805f3c4be96424af1a33d3d986fa85` |
| precision_recall_curves/B_person.svg | 2141 | `37d3111e3e6b58b84999ced4204c6abf45be5eb7420bf639e3511def8ec8b5c8` |

`per_image_metrics.jsonl` is stored as `per_image_metrics.jsonl.gz`; the hash above applies to
the decompressed file. See [`eval/results/README.md`](../eval/results/README.md).

### Verify or repeat the evaluation

From the repository root:

```powershell
python eval/verify_saved_results.py
```

This checks the hash of the decompressed saved predictions and re-scores them, requiring 52
counts, covering both models and the support partitions, to equal `detection_metrics.json`.
It needs no dataset, model files or inference and writes no files.

To repeat the evaluation, place the pretrained Ultralytics v8.2.0 files `yolov8n.pt`
(6,549,796 bytes, SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`) and
`yolov8s.pt` (22,588,772 bytes, SHA-256
`1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`) from
https://github.com/ultralytics/assets/releases/tag/v8.2.0 in `weights/`, obtain the dataset as
described under [Dataset](#dataset), and run:

```powershell
python -X utf8 eval/benchmark.py run --dataset path/to/VisDrone2019-DET-val --output-dir eval/rerun
```

The dataset folder must contain `images/` and `annotations/`, and every file must match the
recorded manifest. The output folder must not already contain results. On the laptop
described above, inference took about 43 minutes.

### Limitations

- VisDrone contains general aerial scenes, mostly of populated areas, rather than desert
  search imagery.
- The protocol is a two-class adaptation, not the official ten-class VisDrone scoring.
- Still images do not measure tracking, telemetry association, video sampling, mission
  processing time or the range of desert conditions in which DALEEL would be used.
- Ranking recall and AP are limited where the 300-detection cap was reached.
- Timing is specific to one laptop session with uncontrolled background load.
