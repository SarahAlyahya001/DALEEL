"""Representative images on disk; never retain a video/frame history."""

from pathlib import Path

import cv2


def write_representative(frame, track, evidence_dir):
    root = Path(evidence_dir)
    root.mkdir(exist_ok=True)
    target = root / f"track-{track.track_id}.jpg"
    temporary = root / f"track-{track.track_id}.pending.jpg"
    annotated = frame.copy()
    candidate = track.representative.candidate
    x1, y1, x2, y2 = (round(v) for v in candidate.box)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(annotated, f"{candidate.target_class} {candidate.agreement}",
                (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    if not cv2.imwrite(str(temporary), annotated):
        raise OSError("Representative evidence could not be written.")
    temporary.replace(target)
    return f"frames/{target.name}"
