"""Two fine-tuned person/car detectors on every sampled frame, with model-agreement labels."""

from dataclasses import dataclass
import math
from pathlib import Path

from .weights import weight_problem

COCO_KEEP = {0: "person", 2: "car"}
BOTH_MODELS = "Both models"
ONE_MODEL = "One model"
AGREEMENT_LABELS = (BOTH_MODELS, ONE_MODEL)
DETECTION_CONF = 0.42  # One threshold for both models, fixed on validation data (docs/DATASET_EVALUATION.md).
PAIR_IOU = 0.50


@dataclass(frozen=True)
class Candidate:
    """One detection of one sampled frame, reported by one model or by both."""
    target_class: str
    box: tuple[float, float, float, float]  # The box of the higher-scoring reporting model.
    yolov8n_score: float | None = None
    yolov8s_score: float | None = None

    def __post_init__(self):
        if self.yolov8n_score is None and self.yolov8s_score is None:
            raise ValueError("A candidate needs the score of at least one model.")

    @property
    def score(self) -> float:
        """The higher score of the models that reported this detection."""
        return max(s for s in (self.yolov8n_score, self.yolov8s_score) if s is not None)

    @property
    def agreement(self) -> str:
        both = self.yolov8n_score is not None and self.yolov8s_score is not None
        return BOTH_MODELS if both else ONE_MODEL


@dataclass(frozen=True)
class FrameDetections:
    candidates: list[Candidate]


def iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def fuse(n_dets, s_dets) -> list[Candidate]:
    """Pair same-class detections one-to-one by descending IoU (>= 0.50); keep every detection.

    Each input is a list of (class, box, score) in the model's output order. A pair becomes one
    candidate with the box of the higher-scoring model (YOLOv8n on an exact tie); unpaired
    detections of either model are kept with their own score.
    """
    fused = []
    for cls in COCO_KEEP.values():
        n = [det for det in n_dets if det[0] == cls]
        s = [det for det in s_dets if det[0] == cls]
        pairs = sorted((-overlap, i, j) for i, (_, nb, _) in enumerate(n) for j, (_, sb, _) in enumerate(s)
                       if (overlap := iou(nb, sb)) >= PAIR_IOU)
        used_n, used_s = set(), set()
        for _, i, j in pairs:
            if i not in used_n and j not in used_s:
                used_n.add(i)
                used_s.add(j)
                box = n[i][1] if n[i][2] >= s[j][2] else s[j][1]
                fused.append(Candidate(cls, box, n[i][2], s[j][2]))
        fused += [Candidate(cls, box, yolov8n_score=score) for i, (_, box, score) in enumerate(n) if i not in used_n]
        fused += [Candidate(cls, box, yolov8s_score=score) for j, (_, box, score) in enumerate(s) if j not in used_s]
    return fused


def local_weights(weights_n, weights_s):
    """Check BOTH paths before constructing either YOLO object: no downloads."""
    paths = []
    for value in (weights_n, weights_s):
        if not value or not Path(value).is_file():
            raise ValueError("Both local YOLOv8n and YOLOv8s weight files are required. "
                             "Run python scripts/setup_weights.py.")
        paths.append(str(Path(value).resolve()))
    return paths


class DualDetector:
    """Runs YOLOv8n and YOLOv8s on the same frame and merges their detections."""

    def __init__(self, weights_n, weights_s, conf=DETECTION_CONF, imgsz=1280, specs=None):
        paths = local_weights(weights_n, weights_s)
        for path, spec in zip(paths, specs or ()):
            # The application runs only the evaluated fine-tuned files (daleel/pipeline/weights.py).
            if (problem := weight_problem(path, spec)) is not None:
                raise ValueError(problem + " Run python scripts/setup_weights.py.")
        from ultralytics import YOLO  # No import or weight access in the factory.
        self.model_n, self.model_s = (YOLO(path) for path in paths)
        for model in (self.model_n, self.model_s):
            names = model.names
            if names.get(0) != "person" or names.get(2) != "car" or len(names) != 80:
                raise ValueError("Only models with the COCO class list (person = 0, car = 2) are supported.")
        self.conf, self.imgsz = conf, imgsz

    @staticmethod
    def _extract(result):
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in COCO_KEEP:
                continue
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            score = float(box.conf.item())
            if (len(xyxy) != 4 or not all(math.isfinite(v) for v in xyxy)
                    or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("The model returned invalid candidate data.")
            detections.append((COCO_KEEP[class_id], xyxy, score))
        # Highest score first, as evaluated; the sort is stable, so equal scores keep model order.
        return sorted(detections, key=lambda d: -d[2])

    def _predict(self, model, frame):
        return self._extract(model.predict(frame, conf=self.conf, classes=[0, 2], verbose=False,
                                           device="cpu", imgsz=self.imgsz)[0])

    def detect(self, frame, on_call=lambda model: None):
        """Both models process every sampled frame, including frames where YOLOv8n finds nothing."""
        on_call("yolov8n")
        n_dets = self._predict(self.model_n, frame)
        on_call("yolov8s")
        s_dets = self._predict(self.model_s, frame)
        return FrameDetections(fuse(n_dets, s_dets))
