# Person-detection improvement experiment: acceptance criteria

Written on 2026-10-19, before any validation result was produced. The criteria and the selection
rule below are not to be changed after the validation results are seen.

## Motivation (documented baseline)

On the HERIDAL test split (101 images, 337 persons) the current DALEEL configuration
(full-frame YOLOv8n at imgsz 1280, confidence > 0.30; YOLOv8s > 0.50 for support labels)
found 14 persons: recall 4.15%, precision 70.0%. Results: `heridal_test_baseline_results.json`.
These test results motivated the investigation and are not used to choose any setting.

## Data roles

- **Validation (selection):** HERIDAL training-split location groups BLA and RAK (93 images).
  Neither location appears in the test split; both are excluded from any later fine-tuning.
- **Test (final, once):** HERIDAL test split, used once for the selected configuration and the
  baseline. A tiling run on the test split made before this protocol was quarantined unread
  (`quarantine_test_runs/`).
- **Input scenarios:** each image is resized to 1920 px wide (1080p-equivalent frame) and to
  3840 px wide (4K-equivalent frame) before a configuration is applied.

## Selection rule (fixed in advance)

For every configuration, the operating confidence threshold is chosen on the validation set as
the threshold that **maximizes person recall subject to precision >= 0.50**. Configurations are
then compared at their own operating thresholds.

## Acceptance criteria (person class, validation set, chosen operating threshold)

| ID | Criterion | Minimum to integrate | Rationale |
|----|-----------|----------------------|-----------|
| AC1 | Recall (IoU >= 0.50) | >= 0.50 | Project-defined minimum: at least half of the persons visible in a frame are surfaced. The reference HERIDAL-trained system reported a detection rate of 88.9% (Bozic-Stulic et al., 2019). |
| AC2 | Precision | >= 0.50 | At most one false alarm per correct detection; too many false positives cause detector suggestions to be mistrusted and ignored (Pyrro et al., 2021). |
| AC3 | False positives per image (persons + cars) | <= 1.0 | Limits the review workload per sampled frame. |
| AC4 | Processing time | 60-second 1080p video (120 sampled frames) within 5 minutes on the minimum-specification laptop (NFR-01), i.e. <= 2.5 s per sampled frame there, estimated as <= 1.6 s per frame on the test laptop using the measured speed ratio of about 1.55 | NFR-01 |
| AC5 | Peak application memory | <= 4 GB | NFR-02 |
| AC6 | Vehicle detection | Car AP@0.5 on VisDrone not lower than the current 55.27% (YOLOv8n) by more than 2 percentage points | Cars remain a required target class (C-03) |

A configuration that improves on 4.15% recall but fails AC1 or AC2 is **not** accepted as
evidence that the detector is useful. Among configurations that meet all criteria, the
simplest one is preferred: fewer models, then fewer inference calls per frame.
If no configuration meets AC1 and AC2, a focused SAR fine-tuning experiment is proposed instead.

## Limits of this evidence

HERIDAL shows Mediterranean wilderness, not desert. Meeting these criteria shows usefulness on
SAR imagery at similar scale; desert performance requires a separate labelled desert evaluation.

## Addendum (2026-10-19, before any fine-tuning result): selection of the fine-tuned YOLOv8n

Training data: HERIDAL training split without BLA and RAK, plus a sequence-split VisDrone training
subset (`visdrone_subset_manifest.json`). No SARD. Validation and test images are never trained on.

Candidates: the `best.pt` and `last.pt` checkpoints, each with three inference paths on
1080p-equivalent complete frames: (a) full frame at imgsz 1280 (the current app path), (b) full frame
at imgsz 1920, (c) 640-px tiles with 20% overlap and class-wise NMS.

Selection (validation only):
1. Person operating threshold per candidate: the pre-registered rule (highest recall with
   precision >= 0.50) on complete HERIDAL validation images (BLA + RAK).
2. Car check on the 194 complete images of the held-out VisDrone validation sequences at the same
   threshold: car AP@0.5 not more than 2 percentage points below the pretrained YOLOv8n on the
   same images and inference path.
3. Among candidates meeting AC1-AC5 and the car check, choose the simplest (full frame before
   tiles; lower imgsz before higher), then the higher validation recall.
4. The two-model design is re-assessed only if YOLOv8s is fine-tuned in a follow-up experiment.

Final evaluation (once, after selection): HERIDAL test split (persons), the official VisDrone
validation split (cars, AC6, repository protocol), CPU time and peak memory on the laptop for the
60-second 1080p test video through the application path, and the desert demonstration check.

## Addendum 2 (2026-10-23, before any YOLOv8s fine-tuning): two-model comparison

**Purpose.** Determine whether a second fine-tuned model (YOLOv8s) makes a measurable contribution
when combined with the fine-tuned YOLOv8n. One focused comparison; the criteria below are fixed before
YOLOv8s is trained and are not changed after its results are seen.

**Test-set exposure.** The HERIDAL test split was used once on 2026-10-23 for the fine-tuned YOLOv8n
alone (EXPERIMENT_LOG section 11). No decision in this addendum uses test results. After the design
is fixed on validation data, the selected design is run once on the test split and reported together
with this prior exposure.

**Models**
- *n*: the fine-tuned YOLOv8n `yolov8n_sar_v1` (converted copy, SHA-256 21f16d2d...e7d). Not retrained.
- *s*: YOLOv8s fine-tuned from the app's pretrained `yolov8s.pt` (SHA-256 1f47a78b...a78a) on the same
  dataset package (`ft_dataset.zip`, SHA-256 8859b4f6...1596), i.e. the same source-level splits,
  person/car definitions and exclusions, with the same training arguments as *n* (imgsz 640, batch 16,
  60 epochs, patience 25, SGD lr0 0.01, flipud 0.5, mosaic 1.0, close_mosaic 10, seed 0). If the
  free-tier budget cannot complete 60 epochs, the shorter run is reported as a deviation.
- Checkpoint of *s*: `best.pt` or `last.pt`, whichever has the higher person AP@0.5 on the complete
  HERIDAL validation images at the common path (tie: `last.pt`).

**Common inference path.** Full frame at imgsz 1280 on 1080p-equivalent frames (the path selected for
*n*, and the app's path); each model applies its own NMS (IoU 0.7, at most 300 detections), as in the app.

**Candidates**
- A: *n* alone.
- B: *s* alone.
- C: combination. Both models process every sampled frame, including frames where *n* finds nothing.
  Fusion is fixed in advance and not tuned:
  1. At a confidence threshold t (the one tuned parameter), each model contributes its detections with
     score > t. Fusion is done at that threshold, so "both models" means that both models reported the
     object at the operating threshold, not that a low-score partner box existed.
  2. Class-aware duplicate handling: *n* and *s* detections of the same class are paired one-to-one by
     greedy descending IoU, IoU >= 0.50 (the app's existing consensus rule).
  3. A pair becomes one detection: the box of the higher-scoring member, score = the higher of the two
     scores, agreement = "both models".
  4. Unpaired detections from either model are kept with their own score and agreement "YOLOv8n only"
     or "YOLOv8s only", so *s* can recover targets that *n* misses.
  5. For AP only, the fusion is computed at t = 0.001 and ranked by the fused score.
  (Clarified 2026-10-23, before YOLOv8s training: steps 1 and 5 replace "fusion at 0.001, then one
  threshold on the fused score", which would have counted low-score partner boxes as agreement.)
  Agreement is recorded as supporting evidence. Disagreement is not used to remove or down-rank a
  detection. Temporal aggregation (Tracker) and human review are unchanged.

**Comparable tuning.** Every candidate has exactly one tuned parameter, its confidence threshold,
chosen on the same validation data with the same rule as before: the highest person recall with
precision >= 0.50 (grid 0.01-0.90, step 0.01).

**Validation data.** HERIDAL BLA + RAK (93 complete images resized to 1920 px wide, 86 persons) as the
primary set; the 194 complete held-out VisDrone images (persons and cars) as the secondary set.
False positives per image (FP/img) are counted as for AC3: person false positives plus car predictions,
except car predictions in the 10 reviewed validation images that show real cars.

**Measures**
- M1. Person recall at matched false-positive rates of 0.25, 0.50 and 1.00 FP/img: for each candidate,
  the lowest threshold whose FP/img does not exceed the level. Primary: 0.50 FP/img.
- M2. Recall, precision, FP/img at each candidate's operating threshold; person AP@0.5.
- M3. Unique correct detections: at the operating thresholds, persons found by *n* only, by *s* only
  and by both (A vs B); and the number of C's correct detections that come only from *s*.
- M4. Agreement as evidence (C at its operating threshold): precision of "both models" detections
  vs single-model detections with 95% Wilson intervals, overall and within fused-score bands
  [thr, 0.50), [0.50, 0.75), [0.75, 1].
- M5. CPU time and peak memory for the 60-second 1080p video through the app's frame loop, each
  candidate, on the same laptop, run one after another (AC4, AC5).
- M6. Car AP@0.5 on the 194 held-out VisDrone images (check as in the first addendum).

**Decision rules**
- *Measurable contribution of the second model*: claimed only if C's person recall at 0.50 FP/img on
  HERIDAL validation exceeds that of the better single model (A or B; the one with the higher recall at
  0.50 FP/img, tie: A) and the 95% paired bootstrap interval of the difference (2,000 resamples of the
  93 images, seed 20261023, thresholds held at the values found on the full validation set; percentile
  interval) lies entirely above zero, and C
  meets AC1-AC5 and the car check. With 86 validation persons a small real gain may be undetectable;
  an interval that includes zero is reported as "no demonstrated benefit", not as "no benefit".
  The VisDrone persons comparison is reported as secondary evidence and does not decide.
- *Agreement is useful additional evidence*: claimed only if "both models" detections have higher
  precision than single-model detections with non-overlapping 95% Wilson intervals, and not lower
  precision in any score band that holds at least 10 detections of each group.
- *Design*: if C meets the contribution rule, C is integrated. Otherwise B replaces A only if B meets
  the same rule against A; otherwise A (YOLOv8n alone) remains the supported design, and the
  measured contribution of YOLOv8s is reported as found.

## Addendum 3 (2026-10-24, before any ordering result is computed): agreement-based review ordering

**Status: exploratory.** Addendum 2's conclusion stands: the combination did not demonstrate a recall
advantage (EXPERIMENT_LOG section 19). This addendum asks a different question: does putting detections
that both models agree on first in the review queue surface correct targets sooner than ordering by
confidence alone, or than either model on its own? The question arose from M4 on the *same* HERIDAL
validation candidates; the agreement precision there (0.844 vs 0.193, and the band counts) was known
when this addendum was written, so a HERIDAL result cannot be confirmatory. The held-out VisDrone images,
on which the agreement pattern has not been examined, are the independent check. One bounded comparison;
if the rule below is not met, no further ordering rules are tried.

**Test-set exposure.** The HERIDAL test split was used once (2026-10-23) for the fine-tuned YOLOv8n alone.
It is not used in this addendum; nothing here is tuned on test results.

**Frozen inputs (saved predictions only; no training, no new inference).** SHA-256:
- HERIDAL validation (93 images): *n* `ft_runs/sar_v1/heridal/best_full1280.json` 97f0d7a1...507c,
  *s* `ft_runs/sar_v1_s/heridal/full1280.json` 57da0c82...ef40.
- VisDrone held-out (194 images, 42 sequences): *n* `ft_runs/sar_v1/visdrone/best_full1280.json`
  3cd5ae40...6c0d, *s* `ft_runs/sar_v1_s/visdrone/full1280.json` f2ef4c5e...ae73.
- Reviewed car images `car_review.json` 9cd679d8...5640.

**Frozen candidate sets** (thresholds and duplicate handling from addendum 2, not re-tuned):
- C: detections of *n* and *s* with score > 0.42, fused by the addendum 2 rule (same class, greedy
  descending IoU >= 0.50, one-to-one; box of the higher-scoring member; score = the higher score;
  agreement "both models", "YOLOv8n only" or "YOLOv8s only"). Every candidate stays in the queue;
  agreement only changes the order and never removes a candidate.
- A: *n* with score > 0.27. B: *s* with score > 0.46 (their addendum 2 operating thresholds).
- Person and car candidates both enter the queue, as both are reviewed in the app.

**Correctness labels** are assigned once per candidate set, before and independent of any ordering:
the repository matching (`match_image`, IoU 0.50, one-to-one, per image and class) applied to each
image's full candidate set. A candidate is *correct* if it is a TP; one-to-one matching means each correct
candidate is a different annotated target, so a duplicate on an already found target is not correct.
Candidates in ignored regions or matched to ignored targets are removed from the queue (as in AP).
HERIDAL labels persons only: car candidates in the 10 reviewed images with real cars are removed from the
queue; other car candidates are incorrect. VisDrone labels both classes: TPs of either class are correct.
Targets T: non-ignored persons (HERIDAL) or persons and cars (VisDrone).

**Orderings** (one queue over all images of a dataset):
- O1 confidence-only: C by score, descending.
- O2 agreement-first: C, all "both models" candidates first, then all single-model candidates; within
  each group by score, descending.
- O3: A by score, descending. O4: B by score, descending.
- Exact score ties in every ordering: image id, then class, then box coordinates, ascending (no
  information about correctness).

**Measures** (found(k) = correct candidates among the first k reviewed; L = length of the C queue):
- Primary: normalized area under the found-vs-reviewed curve over the common horizon k = 1..L,
  AUC = sum_k found(k) / (L * T). A queue shorter than L keeps its final value.
- Correct targets found at fixed budgets k = ceil(f * L), f = 0.10, 0.25, 0.50, 0.75, 1.00.
- Candidates reviewed to find m correct targets, m = ceil(g * M), g = 0.50, 0.75, 0.90, where M is the
  smallest final number of correct candidates among the four queues.
- Differences O2 - O1, O2 - O3, O2 - O4 for every measure (for "reviewed", fewer is better).

**Uncertainty, accounting for related images.** Cluster bootstrap, 2,000 resamples, seed 20261024,
percentile 95% intervals. Clusters: VisDrone, the sequence (first field of the image name; 42 clusters).
HERIDAL, capture-time blocks: within each location group in EXIF capture order, a new block starts when an
image is more than 30 s after the block's first image (consecutive images are seconds apart and overlap).
Sensitivity: 60 s blocks. Each resample draws clusters with replacement, includes all their images (copies
are distinct images), rebuilds every queue, recomputes L, budgets and m, and keeps each candidate's label.

**Decision rule (fixed).** Agreement-first ordering is reported as giving a useful advantage only if, on
*both* datasets, the AUC differences O2 - O1, O2 - O3 and O2 - O4 all have 95% intervals entirely above
zero (30 s blocks for HERIDAL). Otherwise the result is reported as no demonstrated advantage. If the rule
is met, the corresponding dual-model review workflow is proposed and its full-application runtime is
measured on a copy of the application; the application and the report stay unchanged until then.

**Limits.** This is an offline ranking of per-image detections, not a user study: "candidates reviewed"
is a proxy for review effort, and no analyst time saving is claimed. The app reviews observations
aggregated over frames by the tracker; the same person in consecutive HERIDAL images counts as separate
targets here.
