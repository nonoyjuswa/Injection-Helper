#!/usr/bin/env python3
"""PrivARcy Video Frame Extractor

Saves frames from one video (or every video in a folder) as images.

Examples
--------
  py privarcy_video_frames.py                          # uses the default folders below
  py privarcy_video_frames.py clip.mp4                 # one frame per second (default)
  py privarcy_video_frames.py clip.mp4 --seconds 0.5   # one frame every 0.5 s
  py privarcy_video_frames.py clip.mp4 --every 10      # every 10th frame
  py privarcy_video_frames.py clip.mp4 --count 40      # 40 frames spread evenly
  py privarcy_video_frames.py videos/ -o out/ --max-side 1920 --min-sharpness 60
  py privarcy_video_frames.py clip.mp4 --start 5 --end 30 --format png

Output names are  <videoname>_f000123.jpg  (frame number), so they stay unique and
work with the helper's "profiles are matched by file name" rule.

Dependencies:  pip install opencv-python numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.webm', '.mts', '.3gp'}

# ---- Default folders (edit if yours change) ---------------------------------
DEFAULT_VIDEO_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Videos"
DEFAULT_OUT_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Backgrounds"


def list_videos(path: Path):
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted((f for f in path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS),
                      key=lambda f: f.name.lower())
    return []


def sharpness(frame) -> float:
    """Variance of the Laplacian: higher = sharper, low = motion-blurred."""
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def pick_indices(total, fps, start_f, end_f, every, seconds, count):
    """Frame numbers to save."""
    if count:
        return set(np.linspace(start_f, max(start_f, end_f - 1), count).round().astype(int).tolist())
    step = every if every else max(1, round(fps * (seconds if seconds else 1.0)))
    return set(range(start_f, end_f, step))


def extract(video: Path, out_dir: Path, args) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f'  ! cannot open {video.name}')
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start_f = int(args.start * fps)
    end_f = int(args.end * fps) if args.end else total
    if end_f <= 0 or (total and end_f > total):
        end_f = total or 10 ** 9
    if start_f >= end_f:
        print(f'  ! start is after the end of {video.name}')
        cap.release()
        return 0

    wanted = pick_indices(total, fps, start_f, end_f, args.every, args.seconds, args.count)
    last = max(wanted) if wanted else -1
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    idx = start_f
    saved = skipped = 0
    ext = '.png' if args.format == 'png' else '.jpg'
    params = ([cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == '.png'
              else [cv2.IMWRITE_JPEG_QUALITY, args.quality])

    while idx <= last:
        if not cap.grab():                       # end of video (or broken frame count)
            break
        if idx in wanted:
            ok, frame = cap.retrieve()           # decode only the frames we keep
            if ok:
                if args.min_sharpness and sharpness(frame) < args.min_sharpness:
                    skipped += 1
                else:
                    if args.max_side:
                        h, w = frame.shape[:2]
                        s = args.max_side / max(h, w)
                        if s < 1:
                            frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                    dest = out_dir / f'{video.stem}_f{idx:06d}{ext}'
                    if cv2.imwrite(str(dest), frame, params):
                        saved += 1
        idx += 1
    cap.release()
    extra = f', {skipped} skipped as blurry' if skipped else ''
    print(f'  {video.name}: {saved} frame(s) saved{extra}  ({fps:.1f} fps, {total} frames)')
    return saved


def main():
    ap = argparse.ArgumentParser(description='Extract frames from videos into images.')
    ap.add_argument('input', nargs='?', default=DEFAULT_VIDEO_DIR, help='video file or folder of videos')
    ap.add_argument('-o', '--out', default=DEFAULT_OUT_DIR, help='output folder')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--seconds', type=float, help='save one frame every N seconds (default 1.0)')
    g.add_argument('--every', type=int, help='save every Nth frame')
    g.add_argument('--count', type=int, help='save N frames spread evenly across the video')
    ap.add_argument('--start', type=float, default=0.0, help='start time in seconds')
    ap.add_argument('--end', type=float, default=0.0, help='end time in seconds (0 = to the end)')
    ap.add_argument('--format', choices=('jpg', 'png'), default='jpg')
    ap.add_argument('--quality', type=int, default=95, help='JPEG quality (default 95)')
    ap.add_argument('--max-side', type=int, default=0, help='downscale so the longer side is at most N px')
    ap.add_argument('--min-sharpness', type=float, default=0.0,
                    help='skip blurry frames below this Laplacian variance (try 50-100)')
    args = ap.parse_args()

    videos = list_videos(Path(args.input))
    if not videos:
        sys.exit(f'No videos found at: {args.input}')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'{len(videos)} video(s) -> {out_dir}')
    total = sum(extract(v, out_dir, args) for v in videos)
    print(f'Done: {total} image(s) saved.')


if __name__ == '__main__':
    main()