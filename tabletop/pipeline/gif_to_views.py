"""gif_to_views - extract front/left/back stills from a turntable GIF/video.

A spinning turntable goes 0deg -> 360deg. We sample:
  front = 0%  of the sequence
  left  = 25% (a quarter turn)
  back  = 50% (half turn)

Works on GIF (via PIL) and common video files (via imageio if available).
Direction (CW vs CCW) only swaps which quarter-frame is "left" — the model is
fairly robust; pass --reverse if the side looks mirrored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def extract(path: str, out_dir: str, reverse: bool = False) -> dict:
    from PIL import Image, ImageSequence
    p = Path(path)
    frames = []
    if p.suffix.lower() in (".gif", ".webp", ".apng"):
        im = Image.open(p)
        frames = [f.convert("RGBA").copy() for f in ImageSequence.Iterator(im)]
    else:
        # Video (mp4/mov/webm/avi/...). Try imageio first, then OpenCV — the
        # latter has bundled ffmpeg codecs and reads mp4/h264 reliably.
        import numpy as np
        frames = []
        try:
            import imageio.v3 as iio
            vid = iio.imread(p, index=None)  # (T,H,W,C)
            frames = [Image.fromarray(np.asarray(fr)).convert("RGBA")
                      for fr in vid]
        except Exception:
            frames = []
        if len(frames) < 2:
            try:
                import cv2
                cap = cv2.VideoCapture(str(p))
                frames = []
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(fr).convert("RGBA"))
                cap.release()
            except Exception as e:
                raise RuntimeError(f"cannot read video {path}: {e}")
    n = len(frames)
    if n < 2:
        raise RuntimeError(f"{path}: only {n} frame(s) — need a turntable sequence")

    quarter = 0.75 if reverse else 0.25
    idx = {
        "front": 0,
        "left":  int(round(quarter * n)) % n,
        "back":  int(round(0.50 * n)) % n,
    }
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    paths = {}
    for view, i in idx.items():
        fp = od / f"{view}.png"
        frames[i].save(fp)
        paths[view] = str(fp)
    return {"frames": n, "indices": idx, "paths": paths}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="turntable GIF / video")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reverse", action="store_true",
                    help="object spins clockwise — take left from 75%")
    args = ap.parse_args()
    info = extract(args.input, args.out_dir, args.reverse)
    import json
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
