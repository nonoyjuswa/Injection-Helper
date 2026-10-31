#!/usr/bin/env python3
"""PrivARcy Pre-label Helper

Companion to privarcy_injection_helper.py.  Two steps:

  1) train  - train a YOLO detector on the synthetic composites made by the
              injection helper (uses the data.yaml it writes).
  2) label  - run that detector over your REAL images and save draft YOLO
              labels (.txt), ready to be corrected in CVAT / Make Sense /
              labelImg / Label Studio.  Nothing is uploaded anywhere.

Usage (Windows):
    py privarcy_prelabel.py train
    py privarcy_prelabel.py label
    py privarcy_prelabel.py train --epochs 80 --model yolov8s.pt
    py privarcy_prelabel.py label --source "D:\\real_images" --conf 0.2
    py privarcy_prelabel.py --help

Dependencies:  pip install -r privarcy_prelabel_requirements.txt
(Ultralytics pulls in PyTorch.  For an NVIDIA GPU, install the CUDA build of
PyTorch first - see https://pytorch.org/get-started/locally/ )

IMPORTANT: a model trained only on synthetic images will miss or mis-box some
real objects.  Treat its output as DRAFTS and review every image.  Do not use
the same images for final evaluation that you pre-labelled with this model.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs" / "privarcy"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

# ---- Default folders (edit these to match your machine) ---------------------
BASE = r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection"
DEFAULT_SYNTH_DIR = BASE + r"\Output"          # output of the injection helper
DEFAULT_REAL_DIR = BASE + r"\Real"             # your real, unlabeled images
DEFAULT_PRELABEL_DIR = BASE + r"\Prelabeled"   # where draft labels are written


# =============================================================================
#  Helpers
# =============================================================================
def die(msg):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def need_ultralytics():
    try:
        from ultralytics import YOLO              # noqa: F401
        return YOLO
    except ImportError:
        die("Ultralytics is not installed.\n"
            "Run:  pip install -r privarcy_prelabel_requirements.txt")


def pick_device(arg):
    if arg and arg != 'auto':
        return arg
    try:
        import torch
        return '0' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def list_images(folder, recursive=True):
    root = Path(folder)
    if not root.is_dir():
        return []
    it = root.rglob('*') if recursive else root.iterdir()
    return sorted((f for f in it if f.is_file() and f.suffix.lower() in IMAGE_EXTS),
                  key=lambda f: str(f).lower())


def load_yaml(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def class_names(names):
    """Ultralytics names (dict or list) -> list ordered by class id."""
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def find_latest_weights():
    cands = list(RUNS_DIR.glob('*/weights/best.pt'))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


# =============================================================================
#  Step 1: train on synthetic data
# =============================================================================
def make_split(data_yaml, val_fraction, seed):
    """The injection helper writes train: images / val: images (same files).
    Make a real train/val split so validation numbers are less meaningless.
    Returns the path of the yaml to train with."""
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get('path') or Path(data_yaml).parent)
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()
    img_dir = root / str(cfg.get('train', 'images'))
    imgs = list_images(img_dir, recursive=False)
    if not imgs:
        die(f"No images found in {img_dir}.\n"
            f"Generate composites first, with 'Save YOLO labels' ticked in the injection helper.")
    if not (root / 'labels').is_dir():
        die(f"No 'labels' folder next to the images in {root}.\n"
            f"Re-run the injection helper with 'Save YOLO labels' ticked.")
    if val_fraction <= 0 or len(imgs) < 10:
        return Path(data_yaml)                      # too few images to split sensibly

    rng = random.Random(seed)
    shuffled = imgs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val, train = shuffled[:n_val], shuffled[n_val:]
    split_dir = root / 'splits'
    split_dir.mkdir(exist_ok=True)
    (split_dir / 'train.txt').write_text('\n'.join(str(p) for p in train) + '\n', encoding='utf-8')
    (split_dir / 'val.txt').write_text('\n'.join(str(p) for p in val) + '\n', encoding='utf-8')
    names = class_names(cfg['names'])
    out = split_dir / 'data_split.yaml'
    out.write_text('path: ' + root.as_posix() + '\n'
                   'train: ' + (split_dir / 'train.txt').as_posix() + '\n'
                   'val: ' + (split_dir / 'val.txt').as_posix() + '\nnames:\n'
                   + ''.join(f'  {i}: {n}\n' for i, n in enumerate(names)), encoding='utf-8')
    print(f"Split: {len(train)} train / {len(val)} val images  ({out})")
    return out


def cmd_train(a):
    YOLO = need_ultralytics()
    synth = Path(a.synthetic)
    data_yaml = synth / 'data.yaml'
    if not data_yaml.is_file():
        die(f"data.yaml not found in {synth}.\n"
            f"Generate composites with the injection helper first (tick 'Save YOLO labels').")
    data = make_split(data_yaml, a.val_fraction, a.seed)
    device = pick_device(a.device)
    print(f"Training {a.model} on {device.upper() if device == 'cpu' else 'GPU ' + device} "
          f"for up to {a.epochs} epochs ...")
    if device == 'cpu':
        print("  (CPU training is slow. A GPU or Google Colab is much faster.)")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)
    model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device,
                workers=a.workers, patience=a.patience, seed=a.seed,
                project=str(RUNS_DIR), name=a.name)
    best = None
    try:
        best = Path(model.trainer.best)
    except Exception:
        pass
    if best is None or not best.is_file():
        best = find_latest_weights()
    print("\nDone.")
    if best:
        print(f"Best weights: {best}")
        print("Next:  py privarcy_prelabel.py label")
    else:
        print(f"Training finished; look for weights under {RUNS_DIR}")


# =============================================================================
#  Step 2: pre-label real images
# =============================================================================
def copy_image(src, dest):
    """Copy an image; bake in EXIF rotation so label coordinates match what every tool displays."""
    try:
        with Image.open(src) as im:
            orient = im.getexif().get(0x0112, 1)
            if orient not in (None, 1):
                fixed = ImageOps.exif_transpose(im)
                if dest.suffix.lower() in ('.jpg', '.jpeg'):
                    fixed.convert('RGB').save(dest, quality=95, subsampling=0)
                else:
                    fixed.save(dest)
                return
    except Exception:
        pass
    shutil.copy2(src, dest)


def unique_dest(dir_, src, root, used):
    """Flat file name; if two sub-folders hold the same name, prefix with the folder."""
    name = src.name
    if name.lower() in used:
        try:
            rel = src.relative_to(root)
            name = '__'.join(rel.parts)
        except ValueError:
            name = f'{src.parent.name}__{src.name}'
        stem, ext, n = Path(name).stem, Path(name).suffix, 2
        while name.lower() in used:
            name, n = f'{stem}_{n}{ext}', n + 1
    used.add(name.lower())
    return dir_ / name


def cmd_label(a):
    YOLO = need_ultralytics()
    weights = Path(a.weights) if a.weights else find_latest_weights()
    if not weights or not Path(weights).is_file():
        die("No trained weights found. Run 'py privarcy_prelabel.py train' first, "
            "or pass --weights path\\to\\best.pt")
    src_root = Path(a.source)
    images = list_images(src_root, recursive=not a.no_recursive)
    if not images:
        die(f"No images found in {src_root}")

    out = Path(a.out)
    img_dir, lbl_dir = out / 'images', out / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    names = class_names(model.names)

    # Sanity check against the synthetic classes.txt so class IDs stay consistent.
    synth_classes = Path(a.synthetic) / 'classes.txt'
    if synth_classes.is_file():
        expected = [l.strip() for l in synth_classes.read_text(encoding='utf-8').splitlines() if l.strip()]
        if expected != names:
            print("WARNING: the model's classes differ from the synthetic classes.txt:\n"
                  f"  model : {names}\n  synth : {expected}\n"
                  "  Class IDs may not match your synthetic dataset!\n")

    (out / 'classes.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    (out / 'data.yaml').write_text(
        'path: ' + out.as_posix() + '\ntrain: images\nval: images\nnames:\n'
        + ''.join(f'  {i}: {n}\n' for i, n in enumerate(names)), encoding='utf-8')

    # Copy first, so the labels always match the exact files in the output folder.
    print(f"Copying {len(images)} image(s) to {img_dir} ...")
    used, dests = set(), []
    for src in images:
        d = unique_dest(img_dir, src, src_root, used)
        copy_image(src, d)
        dests.append(d)

    device = pick_device(a.device)
    print(f"Pre-labelling with {weights.name} on {'CPU' if device == 'cpu' else 'GPU ' + device} "
          f"(confidence >= {a.conf}) ...")
    rows, total_boxes, empty = [], 0, 0
    t0 = time.time()
    results = model.predict(source=[str(d) for d in dests], stream=True, conf=a.conf, iou=a.iou,
                            imgsz=a.imgsz, device=device, max_det=a.max_det, verbose=False)
    for i, r in enumerate(results, 1):
        p = Path(r.path)
        xywhn = r.boxes.xywhn.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, (x, y, w, h) in zip(cls, xywhn)]
        (lbl_dir / (p.stem + '.txt')).write_text(('\n'.join(lines) + '\n') if lines else '', encoding='utf-8')
        n = len(lines)
        total_boxes += n
        empty += n == 0
        min_conf = float(conf.min()) if n else 0.0
        review = 'EMPTY' if n == 0 else ('low-confidence' if min_conf < a.review_below else '')
        rows.append([p.name, n, ','.join(sorted({names[c] for c in cls})), f'{min_conf:.2f}' if n else '', review])
        if i % 25 == 0 or i == len(dests):
            print(f"  {i}/{len(dests)}", end='\r', flush=True)
    print()

    rows.sort(key=lambda r: (r[4] == '', r[3] == '', r[3]))        # most doubtful first
    with (out / 'review_report.csv').open('w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['image', 'boxes', 'classes', 'lowest_confidence', 'needs_review'])
        wr.writerows(rows)

    flagged = sum(1 for r in rows if r[4])
    print(f"\nDone in {time.time() - t0:.0f}s.")
    print(f"  {len(dests)} images, {total_boxes} draft boxes, {empty} image(s) with no detection.")
    print(f"  {flagged} image(s) flagged for extra attention (see review_report.csv).")
    print(f"  Output: {out}")
    print("\nNext: open the 'images' + 'labels' folders in CVAT / Make Sense / labelImg, "
          "fix or delete wrong boxes, and add missed ones. These are DRAFTS - review every image.")


# =============================================================================
#  CLI
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description='PrivARcy pre-label helper: train on synthetic composites, then draft labels for real images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest='cmd')

    t = sub.add_parser('train', help='train YOLO on the injection helper output',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t.add_argument('--synthetic', default=DEFAULT_SYNTH_DIR, help='injection helper Output folder (has data.yaml)')
    t.add_argument('--model', default='yolov8n.pt', help='starting weights (yolov8n.pt = small/fast, yolov8s.pt = better)')
    t.add_argument('--epochs', type=int, default=50)
    t.add_argument('--imgsz', type=int, default=640)
    t.add_argument('--batch', type=int, default=16, help='lower this if you run out of memory (-1 = auto)')
    t.add_argument('--workers', type=int, default=2, help='data loader workers (use 0 if Windows gives errors)')
    t.add_argument('--patience', type=int, default=20, help='stop early after this many epochs without improvement')
    t.add_argument('--val-fraction', type=float, default=0.1, help='share of synthetic images held out for validation (0 = none)')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--device', default='auto', help="auto, cpu, or GPU index like 0")
    t.add_argument('--name', default='synthetic', help='run name under runs/privarcy/')
    t.set_defaults(func=cmd_train)

    l = sub.add_parser('label', help='draft YOLO labels for real images',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    l.add_argument('--source', default=DEFAULT_REAL_DIR, help='folder of real, unlabeled images')
    l.add_argument('--out', default=DEFAULT_PRELABEL_DIR, help='where images/ labels/ classes.txt are written')
    l.add_argument('--weights', default='', help='trained weights (.pt); default = newest runs/privarcy/*/weights/best.pt')
    l.add_argument('--synthetic', default=DEFAULT_SYNTH_DIR, help='synthetic Output folder (only used to check classes.txt)')
    l.add_argument('--conf', type=float, default=0.25, help='min confidence for a box (lower = more boxes, more false ones)')
    l.add_argument('--iou', type=float, default=0.6, help='NMS overlap threshold')
    l.add_argument('--imgsz', type=int, default=640)
    l.add_argument('--max-det', type=int, default=50)
    l.add_argument('--review-below', type=float, default=0.5, help='flag images whose weakest box is below this confidence')
    l.add_argument('--device', default='auto')
    l.add_argument('--no-recursive', action='store_true', help='do not search sub-folders of --source')
    l.set_defaults(func=cmd_label)
    return p


def main():
    p = build_parser()
    a = p.parse_args()
    if not getattr(a, 'func', None):
        p.print_help()
        return
    a.func(a)


if __name__ == '__main__':
    main()