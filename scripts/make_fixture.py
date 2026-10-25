"""Generate a small test video + matching DJI-style SRT fixture.

Usage:
    python scripts/make_fixture.py [--image path/to/photo.jpg] [--out DIR]

With --image, each frame is built from the given photo (use a photo that
contains people or cars, such as the Ultralytics sample image bus.jpg, so that
the pretrained models have targets to detect). Without it, a synthetic sand
background is generated, which exercises the full workflow including the
empty-results state. The SRT has one cue per second of video, so it passes the
application's upload validation.
"""
import argparse
import os

import cv2
import numpy as np

FPS = 30
SECONDS = 4
W, H = 960, 540
BASE_LAT, BASE_LON = 24.774265, 46.738586  # Riyadh area


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(srt_path):
    """Write one cue per second of video; every cue ends one second after it starts."""
    with open(srt_path, "w", encoding="utf-8") as fh:
        for i in range(SECONDS):
            t0, t1 = float(i), float(i + 1)
            lat = BASE_LAT + i * 0.000180  # ~20 m/s northwards
            lon = BASE_LON + i * 0.000050
            fh.write(
                f"{i + 1}\n{srt_time(t0)} --> {srt_time(t1)}\n"
                f'<font size="28">SrtCnt : {i + 1}, DiffTime : 1000ms\n'
                f"2026-07-19 10:00:{i:02d}.000\n"
                f"[iso : 100] [shutter : 1/2000] [fnum : 280] "
                f"[latitude: {lat:.6f}] [longitude: {lon:.6f}] "
                f"[rel_alt: 60.0 abs_alt: 680.0] </font>\n\n"
            )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="optional source photo for the frames")
    ap.add_argument("--out", default="fixtures")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.image:
        src = cv2.imread(args.image)
        if src is None:
            raise SystemExit(f"Could not read image: {args.image}")
        src = cv2.resize(src, (W, H))
    else:
        src = None

    video_path = os.path.join(args.out, "fixture.mp4")
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(7)
    total = FPS * SECONDS
    for i in range(total):
        if src is not None:
            frame = src.copy()
            # slight jitter so consecutive frames are not bit-identical
            dx = int(3 * np.sin(i / 10))
            frame = np.roll(frame, dx, axis=1)
        else:
            frame = np.full((H, W, 3), (140, 175, 205), np.uint8)  # sand tone (BGR)
            noise = rng.integers(-12, 12, (H, W, 1), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()

    srt_path = os.path.join(args.out, "fixture.srt")
    write_srt(srt_path)

    print(f"Wrote {video_path} ({total} frames @ {FPS} fps) and {srt_path}")


if __name__ == "__main__":
    main()
