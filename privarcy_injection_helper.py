#!/usr/bin/env python3
"""PrivARcy Mock-up Injection Helper  (v2)

Creates synthetic composites by placing transparent mock-up images (ID cards,
bank cards, passports, ...) onto a tabletop in a background photo, using a
four-point perspective transform like Photoshop's "Perspective Warp".

What is new in v2
-----------------
* REAL-WORLD SIZES.  Every background profile stores the real size (cm) of the
  marked tabletop area, and every mock-up class has a real size (cm), e.g.
  ID-1 card = 8.56 x 5.40 cm.  Objects are placed on a metric plane and then
  projected through the tabletop homography, so a card really looks card-sized
  and gets smaller when it is farther from the camera.  (v1 used "fractions of
  the table", which distorted sizes/aspect whenever the table was not square.)
* Metric rotation (no skew), overlap tests in real units, spacing in cm.
* Redesigned GUI: tabs, background status list, zoom/pan marking window with a
  measuring grid and a draggable real-size "ghost" card, live test placement.
* Generation runs in a background thread (window never freezes) with Cancel.
* Better blending: premultiplied-alpha warping, anti-aliased edges, soft
  contact/ambient shadows, optional lighting match and camera blur/grain.
* EXIF orientation handled, settings remembered between runs.

Dependencies:  pip install pillow opencv-python numpy      (Tkinter ships with Python)
Run:           py privarcy_injection_helper.py

Marking order (clockwise on screen): 1 far-left, 2 far-right, 3 near-right, 4 near-left.
Width = edge 1->2, Depth = edge 1->4.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import traceback
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont

APP_TITLE = "PrivARcy Mock-up Injection Helper"
APP_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = APP_DIR / "privarcy_settings.json"
LOG_PATH = APP_DIR / "privarcy_injection_errors.log"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
RES = getattr(Image, "Resampling", Image)

# ---- Default folder paths (edit these if your folders change) ---------------
DEFAULT_BG_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Backgrounds"
DEFAULT_MOCK_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Mockups"
DEFAULT_OUT_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Output"
DEFAULT_PROFILE_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\helper\profiles"

# ---- Real-world mock-up sizes in cm (width x height, orientation-agnostic) --
# A mock-up gets a size when its FILE NAME contains the keyword
# (e.g. "id_card_01.png" -> "id"; "my_passport.png" -> "passport").  The longest
# matching keyword wins.  Editable in the GUI ("Mock-ups" tab).
DEFAULT_SIZES = {
    "id": [8.56, 5.40],          # ISO/IEC 7810 ID-1 (national IDs, PhilSys, UMID, ...)
    "card": [8.56, 5.40],        # bank / credit / debit cards (ID-1)
    "license": [8.56, 5.40],
    "licence": [8.56, 5.40],
    "passport": [12.5, 8.8],     # ISO/IEC 7810 ID-3, closed booklet
    "certificate": [21.0, 29.7], # A4 sheet (birth / marriage certificates) - ADJUST to your real paper
}

DEFAULT_SETTINGS = {
    "bg_dir": DEFAULT_BG_DIR, "mock_dir": DEFAULT_MOCK_DIR,
    "out_dir": DEFAULT_OUT_DIR, "profile_dir": DEFAULT_PROFILE_DIR,
    "count": "3", "objects": "1", "dupes": False,
    "rot": "25", "margin": "1.0", "gap": "1.0", "jitter": "0",
    "keep_aspect": False, "fallback": "8.56",
    "shadow": True, "light": True, "strength": "50", "fx": True,
    "quality": "95", "seed": "", "labels": True,
    "sizes": DEFAULT_SIZES,
}

# ---- UI colours --------------------------------------------------------------
BG = '#f3f4f6'
CARD = '#ffffff'
ACCENT = '#2563eb'
ACCENT_D = '#1d4ed8'
TEXT = '#111827'
SUB = '#6b7280'
C_OK, C_WARN, C_BAD = '#15803d', '#b45309', '#b91c1c'


# =============================================================================
#  Image / file helpers
# =============================================================================
def list_images(folder):
    try:
        p = Path(folder)
        if not p.is_dir():
            return []
        return sorted((f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS),
                      key=lambda f: f.name.lower())
    except OSError:
        return []


def list_mockups(folder):
    """Mock-up images, including those in sub-folders (sub-folder name = class)."""
    root = Path(folder)
    try:
        if not root.is_dir():
            return []
        files = [f for f in root.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    except OSError:
        return []
    return sorted(files, key=lambda f: str(f).lower())


def mockup_class(path, root):
    """Class = first sub-folder under the mock-up root; files directly in the root are 'unclassified'."""
    try:
        rel = Path(path).relative_to(root)
    except ValueError:
        return 'unclassified'
    return rel.parts[0] if len(rel.parts) > 1 else 'unclassified'


def class_map(mockups, root):
    return {m: mockup_class(m, root) for m in mockups}


def yolo_lines(infos, W, H, class_ids):
    lines = []
    for i in infos:
        q = np.asarray(i['quad'])
        x0, x1 = max(0.0, q[:, 0].min()), min(float(W), q[:, 0].max())
        y0, y1 = max(0.0, q[:, 1].min()), min(float(H), q[:, 1].max())
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append(f"{class_ids[i['class']]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return lines


def image_size(path):
    """(width, height) of an image *after* EXIF orientation is applied."""
    with Image.open(path) as im:
        w, h = im.size
        try:
            o = im.getexif().get(0x0112, 1)
        except Exception:
            o = 1
    return (h, w) if o in (5, 6, 7, 8) else (w, h)


def load_rgb(path):
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        return im.convert('RGB')


@lru_cache(maxsize=24)
def _load_mockup_cached(path_str, mtime):
    im = Image.open(path_str)
    im.load()
    im = im.convert('RGBA')
    mask = im.getchannel('A').point(lambda v: 255 if v > 8 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError('Mock-up image is empty / fully transparent.')
    im = im.crop(bbox)
    m = max(im.size)
    if m > 3000:                               # huge mock-ups are never needed at full size
        s = 3000.0 / m
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), RES.LANCZOS)
    return im


def load_mockup(path):
    """Mock-up as RGBA, trimmed to its visible pixels so padding never changes its real size."""
    p = Path(path)
    return _load_mockup_cached(str(p), p.stat().st_mtime_ns)


def open_path(path):
    path = str(path)
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)                 # type: ignore[attr-defined]
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception:
        messagebox.showinfo('Location', path)


def log_error(task, item, error):
    try:
        with LOG_PATH.open('a', encoding='utf-8') as f:
            f.write(f'\n[{datetime.datetime.now().isoformat(timespec="seconds")}] {task}: {item}\n'
                    f'{error}\n{traceback.format_exc()}\n')
    except Exception:
        pass


# =============================================================================
#  Geometry (metric tabletop plane)
#  Plane coordinates are centimetres: origin = corner 1 (far-left),
#  +x -> corner 2 (far-right),  +y -> corner 4 (near-left, toward the camera).
# =============================================================================
def signed_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0


def is_convex(pts):
    n, sign = len(pts), 0
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cx, cy = pts[(i + 2) % n]
        cr = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cr) < 1e-9:
            return False
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def validate_table(table_px, table_cm):
    """Return an error string, or None when the tabletop definition is usable."""
    if not table_px or len(table_px) != 4:
        return 'Mark exactly four tabletop corners.'
    if not table_cm or len(table_cm) != 2:
        return 'Enter the real width and depth (cm) of the marked area.'
    try:
        tw, td = float(table_cm[0]), float(table_cm[1])
    except (TypeError, ValueError):
        return 'Table width/depth must be numbers (cm).'
    if not (1 <= tw <= 2000 and 1 <= td <= 2000):
        return 'Table width/depth must be between 1 and 2000 cm.'
    if not is_convex(table_px):
        return ('The four corners must form a convex four-sided shape (no crossing lines). '
                'Order: far-left, far-right, near-right, near-left.')
    if signed_area(table_px) <= 0:
        return ('The corners are in counter-clockwise order, which would mirror the mock-ups. '
                'Order: far-left, far-right, near-right, near-left (clockwise on screen).')
    return None


def build_homography(table_px, table_cm):
    tw, td = float(table_cm[0]), float(table_cm[1])
    plane = np.float32([[0, 0], [tw, 0], [tw, td], [0, td]])
    return cv2.getPerspectiveTransform(plane, np.float32(table_px))


def apply_h(H, pts):
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H).reshape(-1, 2)


def footprint_plane(cx, cy, w, h, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [(cx + dx * c - dy * s, cy + dx * s + dy * c)
            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def convex_polys_overlap(a, b):
    """Separating-axis test for two convex polygons (touching counts as overlap)."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            nx, ny = y1 - y2, x2 - x1
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def estimate_aspect(table_px, W, H):
    """Estimate depth/width of the marked rectangle from perspective alone.

    Returns (ratio, method) or (None, None).  method is 'solved' when the focal
    length could be recovered from the rectangle itself, or 'assumed' when the
    view is level/centred (equation degenerate) and a typical phone focal length
    (0.75 x the longer image side) was assumed - less reliable.
    Assumes square pixels, principal point at the image centre, un-cropped photo.
    """
    try:
        plane = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        Hm = cv2.getPerspectiveTransform(plane, np.float32(table_px)).astype(np.float64)
        cx, cy = W / 2.0, H / 2.0
        h1, h2 = Hm[:, 0], Hm[:, 1]
        m = max(W, H)
        a = (h1[0] - cx * h1[2]) * (h2[0] - cx * h2[2]) + (h1[1] - cy * h1[2]) * (h2[1] - cy * h2[2])
        b = h1[2] * h2[2]
        f, method = None, 'solved'
        if abs(b) > 1e-14:
            f2 = -a / b
            if np.isfinite(f2) and f2 > 0 and 0.3 * m <= math.sqrt(f2) <= 6 * m:
                f = math.sqrt(f2)
        if f is None:
            f, method = 0.75 * m, 'assumed'
        n1 = np.linalg.norm([(h1[0] - cx * h1[2]) / f, (h1[1] - cy * h1[2]) / f, h1[2]])
        n2 = np.linalg.norm([(h2[0] - cx * h2[2]) / f, (h2[1] - cy * h2[2]) / f, h2[2]])
        r = float(n2 / n1)
        return (r, method) if 0.1 <= r <= 10 else (None, None)
    except Exception:
        return None, None


class ObstacleMask:
    """Low-resolution raster of the 'avoid' polygons for fast overlap tests."""

    def __init__(self, W, H, obstacles):
        self.s = min(1.0, 900.0 / max(W, H))
        self.w, self.h = max(1, int(W * self.s)), max(1, int(H * self.s))
        self.mask = np.zeros((self.h, self.w), np.uint8)
        for poly in obstacles or []:
            if len(poly) >= 3:
                cv2.fillPoly(self.mask, [np.round(np.array(poly) * self.s).astype(np.int32)], 255)
        if self.mask.any():
            self.mask = cv2.dilate(self.mask, np.ones((3, 3), np.uint8))
        self.any = bool(self.mask.any())

    def hits(self, quad):
        if not self.any:
            return False
        tmp = np.zeros_like(self.mask)
        cv2.fillPoly(tmp, [np.round(np.asarray(quad) * self.s).astype(np.int32)], 255)
        return cv2.countNonZero(cv2.bitwise_and(tmp, self.mask)) > 0


# =============================================================================
#  Real-world size resolution
# =============================================================================
def norm_name(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def match_size_key(name, sizes):
    n = norm_name(name)
    best = None
    for key in sizes:
        k = norm_name(key)
        if k and k in n and (best is None or len(k) > len(norm_name(best))):
            best = key
    return best


def resolve_real_size(path, px_size, sizes, fallback_long_cm, keep_aspect, cls=None):
    """Real-world (w_cm, h_cm) for a mock-up, oriented like the image.

    Returns (w_cm, h_cm, matched_key_or_None, [warnings]).
    """
    fw, fh = px_size
    landscape = fw >= fh
    warns = []
    key = match_size_key(cls, sizes) if cls and cls != 'unclassified' else None   # folder name first
    if key is None:
        key = match_size_key(Path(path).stem, sizes)                              # then file name
    img_aspect = max(fw, fh) / max(1, min(fw, fh))
    if key:
        a, b = float(sizes[key][0]), float(sizes[key][1])
        long_s, short_s = max(a, b), min(a, b)
        if keep_aspect:
            short_s = long_s / img_aspect
        else:
            real_aspect = long_s / max(short_s, 1e-6)
            if abs(real_aspect / img_aspect - 1) > 0.12:
                warns.append(f'image aspect {img_aspect:.2f} differs from real {real_aspect:.2f} - '
                             f'it will be stretched (check the PNG crop, or tick "Keep image aspect")')
    else:
        long_s = float(fallback_long_cm)
        short_s = long_s / img_aspect
        warns.append('no size keyword matched - using fallback size')
    w, h = (long_s, short_s) if landscape else (short_s, long_s)
    return w, h, key, warns


# =============================================================================
#  Rendering
# =============================================================================
class PlacementError(RuntimeError):
    pass


def prefilter(pre, quad):
    """Shrink (area-average) the premultiplied source to about its on-screen size first."""
    sh, sw = pre.shape[:2]
    q = np.asarray(quad)
    tw = max(np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[3])))
    th = max(np.hypot(*(q[3] - q[0])), np.hypot(*(q[2] - q[1])))
    nw = int(max(2, min(sw, math.ceil(tw))))
    nh = int(max(2, min(sh, math.ceil(th))))
    if nw < sw or nh < sh:
        pre = cv2.resize(pre, (nw, nh), interpolation=cv2.INTER_AREA)
    return pre


def warp_premult(pre, quad, box):
    """Perspective-warp a premultiplied RGBA float image into a region of the output."""
    x0, y0, x1, y1 = box
    sh, sw = pre.shape[:2]
    src = np.float32([[-0.5, -0.5], [sw - 0.5, -0.5], [sw - 0.5, sh - 0.5], [-0.5, sh - 0.5]])
    dst = (np.asarray(quad, dtype=np.float64) - np.array([x0, y0]) - 0.5).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(pre, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def scene_stats(canvas, table, obstacles, strength):
    """Brightness gain and colour tint that make a flat mock-up sit in the scene's light."""
    H, W = canvas.shape[:2]
    s = min(1.0, 400.0 / max(W, H))
    sm = cv2.resize(canvas, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    mask = np.zeros(sm.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(table) * s).astype(np.int32)], 255)
    for poly in obstacles or []:
        if len(poly) >= 3:
            cv2.fillPoly(mask, [np.round(np.array(poly) * s).astype(np.int32)], 0)
    region = sm[mask > 0] if mask.any() else sm.reshape(-1, 3)
    lum = float((region.astype(np.float32) * np.array([0.299, 0.587, 0.114], np.float32)).sum(1).mean())
    mean_all = sm.reshape(-1, 3).astype(np.float32).mean(0)
    k = strength / 100.0
    gain = 1.0 + k * (float(np.clip(lum / 140.0, 0.6, 1.3)) - 1.0)
    cast = np.clip(mean_all / max(float(mean_all.mean()), 1.0), 0.85, 1.15)
    tint = (cast ** (k * 0.6)).astype(np.float32)
    return {'gain': gain, 'tint': tint}


def composite_object(canvas, fg_img, quad, opt, scene, light, nprng):
    """Warp one mock-up onto canvas (uint8 RGB, modified in place) with shadow + realism."""
    Hh, Ww = canvas.shape[:2]
    quad = np.asarray(quad, dtype=np.float64)
    obj = math.sqrt(max(abs(signed_area(quad)), 1.0))
    pad = int(obj * 0.10) + 4
    x0 = max(0, int(math.floor(quad[:, 0].min())) - pad)
    y0 = max(0, int(math.floor(quad[:, 1].min())) - pad)
    x1 = min(Ww, int(math.ceil(quad[:, 0].max())) + pad + 1)
    y1 = min(Hh, int(math.ceil(quad[:, 1].max())) + pad + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    fg = np.asarray(fg_img, dtype=np.float32) / 255.0
    rgb, a = fg[..., :3], fg[..., 3:4]
    if scene:
        rgb = np.clip(rgb * scene['gain'] * scene['tint'], 0, 1)
    pre = np.concatenate([rgb * a, a], axis=2)            # premultiplied alpha -> no dark fringes
    pre = prefilter(pre, quad)
    layer = warp_premult(pre, quad, (x0, y0, x1, y1))
    if opt['fx']:
        layer = cv2.GaussianBlur(layer, (0, 0), float(nprng.uniform(0.35, 0.9)))
        ns = float(nprng.uniform(0.0, 3.0)) / 255.0
        if ns > 0:
            hh, ww = layer.shape[:2]
            noise = nprng.normal(0, ns, (hh, ww, 1)).astype(np.float32)
            chroma = nprng.normal(0, ns * 0.4, (hh, ww, 3)).astype(np.float32)
            layer[..., :3] += (noise + chroma) * layer[..., 3:4]
    alpha = np.ascontiguousarray(layer[..., 3])
    roi = canvas[y0:y1, x0:x1].astype(np.float32) / 255.0
    if opt['shadow']:
        for blur_f, off_f, opacity in ((0.006, 0.004, 0.28), (0.022, 0.016, 0.14)):
            M = np.float32([[1, 0, light[0] * obj * off_f], [0, 1, light[1] * obj * off_f]])
            sh = cv2.warpAffine(alpha, M, (alpha.shape[1], alpha.shape[0]),
                                flags=cv2.INTER_LINEAR, borderValue=0)
            sh = cv2.GaussianBlur(sh, (0, 0), max(0.6, obj * blur_f))
            roi *= (1.0 - opacity * sh)[..., None]
    roi = layer[..., :3] + roi * (1.0 - alpha[..., None])
    canvas[y0:y1, x0:x1] = np.clip(roi * 255.0 + 0.5, 0, 255).astype(np.uint8)


def pick_mockups(rng, mockups, n, first=None, dupes=False, class_of=None):
    """Random selection of n mock-ups for one image.

    Extras are drawn class-first (uniform over classes, preferring classes not yet
    in the image), then file-within-class, so classes with many files are not
    over-represented and different classes get mixed together.
    """
    chosen = [first] if first is not None else []
    cls = (lambda m: class_of[m]) if class_of else (lambda m: '')
    by = {}
    for m in mockups:
        by.setdefault(cls(m), []).append(m)
    while len(chosen) < n:
        cands = [(c, [m for m in ms if dupes or m not in chosen]) for c, ms in by.items()]
        cands = [(c, ms) for c, ms in cands if ms]
        if not cands:
            break
        used = {cls(m) for m in chosen}
        fresh = [x for x in cands if x[0] not in used]
        c, ms = rng.choice(fresh or cands)
        chosen.append(rng.choice(ms))
    return chosen


def compose_image(bg_path, profile, mockups, sizes, opt, rng):
    """Build one composite.  Returns (PIL RGB image, [info dict per placed object])."""
    canvas = np.array(load_rgb(bg_path), dtype=np.uint8)
    Hh, Ww = canvas.shape[:2]
    table = [tuple(p) for p in profile['table']]
    table_cm = profile.get('table_cm')
    err = validate_table(table, table_cm)
    if err:
        raise ValueError(err)
    tw, td = float(table_cm[0]), float(table_cm[1])
    Hm = build_homography(table, (tw, td))
    obstacles = profile.get('obstacles', [])
    obs = ObstacleMask(Ww, Hh, obstacles)
    scene = scene_stats(canvas, table, obstacles, opt['strength']) if opt['light'] else None
    ang = math.radians(rng.uniform(25, 155))
    light = (math.cos(ang), math.sin(ang))                 # shadows fall "downwards"
    nprng = np.random.default_rng(rng.randrange(2 ** 32))
    margin, gap, max_rot = opt['margin'], opt['gap'], opt['rot']

    items = []
    for mp in mockups:
        fg = load_mockup(mp)
        cls = mockup_class(mp, opt['mock_dir'])
        w, h, key, _ = resolve_real_size(mp, fg.size, sizes, opt['fallback'], opt['keep_aspect'], cls)
        if opt['jitter'] > 0:
            f = 1 + rng.uniform(-opt['jitter'], opt['jitter']) / 100.0
            w, h = w * f, h * f
        items.append((Path(mp), fg, w, h, cls))
    items.sort(key=lambda t: -(t[2] * t[3]))              # big things first -> easier packing

    placed_plane, placed = [], []
    for mp, fg, w, h, cls in items:
        quad, fit_possible = None, False
        for _ in range(400):
            angle = rng.uniform(-max_rot, max_rot) if max_rot > 0 else 0.0
            a = math.radians(angle)
            ca, sa = abs(math.cos(a)), abs(math.sin(a))
            hx, hy = (w * ca + h * sa) / 2, (w * sa + h * ca) / 2
            xlo, xhi = margin + hx, tw - margin - hx
            ylo, yhi = margin + hy, td - margin - hy
            if xlo > xhi or ylo > yhi:
                continue
            fit_possible = True
            cx, cy = rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)
            infl = footprint_plane(cx, cy, w + gap, h + gap, angle)
            if any(convex_polys_overlap(infl, p) for p in placed_plane):
                continue
            q = apply_h(Hm, footprint_plane(cx, cy, w, h, angle))
            if obs.hits(q):
                continue
            quad = q
            placed_plane.append(infl)
            break
        if quad is None:
            if not fit_possible:
                raise PlacementError(f'{mp.name} ({w:.1f} x {h:.1f} cm) is larger than the marked table area '
                                     f'({tw:g} x {td:g} cm). Re-check the table size in the profile.')
            raise PlacementError(f'Could not place {mp.name} ({w:.1f} x {h:.1f} cm) on the {tw:g} x {td:g} cm '
                                 f'area without overlaps. Shrink obstacle zones, lower "mock-ups per image" '
                                 f'or the spacing.')
        placed.append((mp, fg, quad, (w, h), cls))

    infos = []
    for mp, fg, quad, (w, h), cls in placed:
        composite_object(canvas, fg, quad, opt, scene, light, nprng)
        infos.append({'file': mp.name, 'class': cls, 'size_cm': (round(w, 2), round(h, 2)),
                      'quad': quad.tolist()})
    return Image.fromarray(canvas), infos


# =============================================================================
#  Profiles & settings
# =============================================================================
def profile_file(profile_dir, image_path):
    return Path(profile_dir) / (Path(image_path).stem + '.json')


def load_profile(profile_dir, image_path):
    p = profile_file(profile_dir, image_path)
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding='utf-8'))
    w, h = image_size(image_path)
    ow, oh = (data.get('size') or [w, h])[:2]
    sx, sy = (w / ow if ow else 1.0), (h / oh if oh else 1.0)
    if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:           # image was resized since marking
        data['table'] = [[x * sx, y * sy] for x, y in data.get('table', [])]
        data['obstacles'] = [[[x * sx, y * sy] for x, y in poly] for poly in data.get('obstacles', [])]
    data.pop('mockup_calibrations', None)                  # v1 sample-size quads are obsolete
    data.setdefault('table', [])
    data.setdefault('obstacles', [])
    data.setdefault('table_cm', None)
    return data


def save_profile(profile_dir, image_path, data):
    p = profile_file(profile_dir, image_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')


def profile_status(profile):
    """-> (code, label)"""
    if not profile or len(profile.get('table', [])) != 4:
        return 'none', '✖ Not marked'
    if not profile.get('table_cm'):
        return 'size', '⚠ Needs real size'
    if validate_table(profile['table'], profile['table_cm']):
        return 'bad', '⚠ Invalid corners'
    return 'ok', '✔ Ready'


def load_settings():
    s = json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        if SETTINGS_PATH.is_file():
            saved = json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
            for k in s:
                if k in saved:
                    s[k] = saved[k]
    except Exception as e:
        log_error('load settings', SETTINGS_PATH, e)
    return s


# =============================================================================
#  Background generation worker
# =============================================================================
class GenerationWorker(threading.Thread):
    """Runs the whole batch off the UI thread; talks to the UI through a queue."""

    def __init__(self, cfg, q, cancel):
        super().__init__(daemon=True)
        self.cfg, self.q, self.cancel = cfg, q, cancel

    def log(self, msg, level='info'):
        self.q.put(('log', msg, level))

    def run(self):
        try:
            self._run()
        except Exception as e:
            log_error('generation', self.cfg.get('out_dir'), e)
            self.q.put(('fatal', str(e)))

    def _run(self):
        c = self.cfg
        backgrounds = list_images(c['bg_dir'])
        mockups = list_mockups(c['mock_dir'])
        class_of = class_map(mockups, c['mock_dir'])
        classes = sorted(set(class_of.values()))
        class_ids = {n: i for i, n in enumerate(classes)}
        if not backgrounds:
            raise ValueError('No images found in the Backgrounds folder.')
        if not mockups:
            raise ValueError('No images found in the Mock-ups folder.')
        if c['objects'] > len(mockups) and not c['dupes']:
            raise ValueError(f'"Mock-ups per image" is {c["objects"]} but there are only {len(mockups)} mock-up '
                             f'files. Add more files, lower the number, or tick "Allow the same mock-up twice".')

        profiles = {}
        for bg in backgrounds:
            try:
                prof = load_profile(c['profile_dir'], bg)
            except Exception as e:
                self.log(f'Skipping {bg.name}: unreadable profile ({e})', 'warn')
                continue
            code, label = profile_status(prof)
            if code == 'ok':
                profiles[bg] = prof
            else:
                self.log(f'Skipping {bg.name}: {label[2:]}', 'warn')
        if not profiles:
            raise ValueError('No background has a complete profile. Mark the tabletop corners AND enter the '
                             'real table size for at least one background (tab 1).')
        self.log(f'{len(profiles)} background(s) ready, {len(mockups)} mock-up(s) in {len(classes)} class(es): '
                 + ', '.join(f'{n} ({sum(1 for v in class_of.values() if v == n)})' for n in classes), 'ok')
        for m in mockups:
            try:
                fg = load_mockup(m)
                w, h, key, warns = resolve_real_size(m, fg.size, c['sizes'], c['fallback'], c['keep_aspect'],
                                                     class_of[m])
                self.log(f'  [{class_of[m]}] {m.name}: {w:.2f} x {h:.2f} cm' + (f'  ({key})' if key else '')
                         + (f'  ! {"; ".join(warns)}' if warns else ''), 'warn' if warns else 'info')
            except Exception as e:
                self.log(f'  {m.name}: cannot load ({e})', 'err')

        seed = c['seed']
        rng = random.Random(int(seed) if str(seed).strip() else None)
        out = Path(c['out_dir'])
        out.mkdir(parents=True, exist_ok=True)
        img_dir = out / 'images' if c['labels'] else out
        lbl_dir = out / 'labels'
        if c['labels']:
            img_dir.mkdir(exist_ok=True)
            lbl_dir.mkdir(exist_ok=True)
            (out / 'classes.txt').write_text('\n'.join(classes) + '\n', encoding='utf-8')
            (out / 'data.yaml').write_text(
                'path: ' + out.as_posix() + '\ntrain: images\nval: images\nnames:\n'
                + ''.join(f'  {i}: {n}\n' for i, n in enumerate(classes)), encoding='utf-8')
        total = len(mockups) * c['count']
        done = ok = fail = 0
        keys = list(profiles.keys())
        self.q.put(('progress', 0, total, 0, 0))
        cancelled = False
        for mock in mockups:
            for k in range(c['count']):
                if self.cancel.is_set():
                    cancelled = True
                    break
                last_err, success = None, False
                for _ in range(6):                      # try other backgrounds if one is too crowded
                    bg = rng.choice(keys)
                    chosen = pick_mockups(rng, mockups, c['objects'], first=mock, dupes=c['dupes'], class_of=class_of)
                    try:
                        img, infos = compose_image(bg, profiles[bg], chosen, c['sizes'], c, rng)
                        name = f'{bg.stem}__{class_of[mock]}__{mock.stem}__{k + 1:03d}_{c["objects"]}objects'
                        dest, n = img_dir / f'{name}.jpg', 2
                        while dest.exists():
                            dest, n = img_dir / f'{name}_{n}.jpg', n + 1
                        img.save(dest, quality=c['quality'], optimize=True,
                                 subsampling=0 if c['quality'] >= 90 else 2)
                        if c['labels']:
                            (lbl_dir / (dest.stem + '.txt')).write_text(
                                '\n'.join(yolo_lines(infos, img.width, img.height, class_ids)) + '\n',
                                encoding='utf-8')
                        success = True
                        break
                    except PlacementError as e:
                        last_err = e
                    except Exception as e:
                        last_err = e
                        log_error(f'generate {mock.name} #{k + 1}', bg, e)
                        break
                done += 1
                if success:
                    ok += 1
                else:
                    fail += 1
                    self.log(f'Failed {mock.name} #{k + 1}: {last_err}', 'err')
                self.q.put(('progress', done, total, ok, fail))
            if cancelled:
                break
        self.q.put(('done', ok, fail, str(out), cancelled))


# =============================================================================
#  GUI pieces
# =============================================================================
def flat(pts):
    return [v for p in pts for v in p]


class SizeDialog(tk.Toplevel):
    """Small modal dialog to add / edit one real-world size entry."""

    def __init__(self, parent, title, key='', w='', h=''):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.configure(bg=BG)
        self.result = None
        self.v_key, self.v_w, self.v_h = tk.StringVar(value=key), tk.StringVar(value=str(w)), tk.StringVar(value=str(h))
        f = ttk.Frame(self, padding=16)
        f.pack()
        ttk.Label(f, text='File-name keyword').grid(row=0, column=0, sticky='w', pady=4)
        e = ttk.Entry(f, textvariable=self.v_key, width=22)
        e.grid(row=0, column=1, columnspan=2, pady=4, padx=(8, 0))
        ttk.Label(f, text='Real width (cm)').grid(row=1, column=0, sticky='w', pady=4)
        ttk.Entry(f, textvariable=self.v_w, width=10).grid(row=1, column=1, pady=4, padx=(8, 0), sticky='w')
        ttk.Label(f, text='Real height (cm)').grid(row=2, column=0, sticky='w', pady=4)
        ttk.Entry(f, textvariable=self.v_h, width=10).grid(row=2, column=1, pady=4, padx=(8, 0), sticky='w')
        ttk.Label(f, text='Any file whose name contains the keyword gets this size,\n'
                          'e.g. keyword "passport" matches "ph_passport_02.png".',
                  style='Sub.TLabel', justify='left').grid(row=3, column=0, columnspan=3, sticky='w', pady=(6, 10))
        bar = ttk.Frame(f)
        bar.grid(row=4, column=0, columnspan=3, sticky='e')
        ttk.Button(bar, text='Cancel', command=self.destroy).pack(side='right')
        ttk.Button(bar, text='OK', style='Accent.TButton', command=self.ok).pack(side='right', padx=6)
        self.bind('<Return>', lambda _e: self.ok())
        self.bind('<Escape>', lambda _e: self.destroy())
        e.focus_set()
        self.update_idletasks()
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def ok(self):
        key = self.v_key.get().strip()
        try:
            w = float(self.v_w.get().replace(',', '.'))
            h = float(self.v_h.get().replace(',', '.'))
        except ValueError:
            messagebox.showerror('Invalid', 'Width and height must be numbers (cm).', parent=self)
            return
        if not norm_name(key):
            messagebox.showerror('Invalid', 'Enter a keyword (letters/numbers).', parent=self)
            return
        if not (0.1 <= w <= 500 and 0.1 <= h <= 500):
            messagebox.showerror('Invalid', 'Sizes must be between 0.1 and 500 cm.', parent=self)
            return
        self.result = (key, w, h)
        self.destroy()


class PreviewWindow(tk.Toplevel):
    """Shows a test composite; 'Re-roll' draws a new random placement."""

    def __init__(self, parent, make_fn):
        super().__init__(parent)
        self.title('Test placement preview')
        self.geometry('1100x760')
        self.minsize(500, 400)
        self.configure(bg=BG)
        self.make_fn = make_fn
        self.img = None
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg='#111827', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        bar = ttk.Frame(self, padding=8)
        bar.grid(row=1, column=0, sticky='ew')
        self.info = tk.StringVar()
        ttk.Label(bar, textvariable=self.info, style='Sub.TLabel').pack(side='left', fill='x', expand=True)
        ttk.Button(bar, text='Close', command=self.destroy).pack(side='right')
        ttk.Button(bar, text='🎲  Re-roll', style='Accent.TButton', command=self.reroll).pack(side='right', padx=6)
        self.canvas.bind('<Configure>', lambda _e: self.show())
        self.after(50, self.reroll)

    def reroll(self):
        self.config(cursor='watch')
        self.update_idletasks()
        try:
            self.img, infos = self.make_fn()
            self.info.set('   '.join(f'[{i["class"]}] {i["file"]}: {i["size_cm"][0]} x {i["size_cm"][1]} cm' for i in infos))
        except Exception as e:
            log_error('test placement', '', e)
            messagebox.showerror('Could not place', str(e), parent=self)
            self.info.set('')
        finally:
            self.config(cursor='')
        self.show()

    def show(self):
        if self.img is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        s = min(cw / self.img.width, ch / self.img.height)
        im = self.img.resize((max(1, int(self.img.width * s)), max(1, int(self.img.height * s))), RES.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(im)
        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img)


class MarkingWindow(tk.Toplevel):
    """Mark the tabletop (4 corners), its real size, and objects to avoid."""

    def __init__(self, app, image_path):
        super().__init__(app)
        self.app = app
        self.path = Path(image_path)
        self.original = load_rgb(self.path)              # raises if unreadable -> caller handles
        self.W, self.H = self.original.size
        self.title(f'Mark tabletop – {self.path.name}')
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f'{min(1320, sw - 60)}x{min(840, sh - 100)}+20+20')
        self.minsize(900, 600)
        self.configure(bg=BG)

        self.table, self.obstacles, self.current = [], [], []
        self.mode = 'table'
        self.zoom, self.ox, self.oy, self.fit_zoom, self.fitted = 1.0, 0.0, 0.0, 1.0, True
        self.drag, self.pan = None, None
        self.dirty = False
        self.ghost_c = None
        self.tkimg = None
        self.mac = sys.platform == 'darwin'

        self.var_w, self.var_d = tk.StringVar(), tk.StringVar()
        self.show_grid = tk.BooleanVar(value=True)
        self.show_ghost = tk.BooleanVar(value=False)
        self.ghost_key = tk.StringVar()
        self.status = tk.StringVar()

        try:
            existing = load_profile(app.v_profiles.get(), self.path)
        except Exception as e:
            log_error('load profile', self.path, e)
            existing = None
        if existing:
            self.table = [tuple(p) for p in existing.get('table', [])]
            self.obstacles = [[tuple(p) for p in poly] for poly in existing.get('obstacles', [])]
            cm = existing.get('table_cm')
            if cm:
                self.var_w.set(f'{cm[0]:g}')
                self.var_d.set(f'{cm[1]:g}')
        self.mode = 'idle' if len(self.table) == 4 else 'table'

        self.build_ui()
        self.var_w.trace_add('write', lambda *_: self.on_size_edit())
        self.var_d.trace_add('write', lambda *_: self.on_size_edit())
        self.set_mode(self.mode)
        self.protocol('WM_DELETE_WINDOW', self.on_close)
        self.after(120, self.redraw)

    # ---------------------------------------------------------------- UI ----
    def build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg='#1f2937', highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, sticky='nsew')
        side = ttk.Frame(self, padding=(10, 8))
        side.grid(row=0, column=1, sticky='ns')
        bar = ttk.Frame(self, padding=(10, 8))
        bar.grid(row=1, column=0, columnspan=2, sticky='ew')

        f = ttk.LabelFrame(side, text='①  Tabletop area', padding=8)
        f.pack(fill='x', pady=(0, 8))
        ttk.Label(f, text='Click 4 corners clockwise:\nfar-left → far-right → near-right → near-left.\n'
                          'Drag a corner to fine-tune.', style='Sub.TLabel', justify='left').pack(anchor='w')
        ttk.Button(f, text='Mark / re-mark corners', command=self.start_table).pack(fill='x', pady=(6, 0))

        f = ttk.LabelFrame(side, text='②  Real size of that area', padding=8)
        f.pack(fill='x', pady=(0, 8))
        g = ttk.Frame(f)
        g.pack(fill='x')
        ttk.Label(g, text='Width (edge 1→2)').grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(g, textvariable=self.var_w, width=8).grid(row=0, column=1, padx=6)
        ttk.Label(g, text='cm').grid(row=0, column=2)
        ttk.Label(g, text='Depth (edge 1→4)').grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(g, textvariable=self.var_d, width=8).grid(row=1, column=1, padx=6)
        ttk.Label(g, text='cm').grid(row=1, column=2)
        ttk.Button(f, text='Estimate depth from perspective', command=self.estimate_depth).pack(fill='x', pady=(6, 0))
        ttk.Label(f, text='Best: measure the marked area with a tape measure. Use the ghost card below '
                          'to double-check the scale.', style='Sub.TLabel', wraplength=260,
                  justify='left').pack(anchor='w', pady=(6, 0))

        f = ttk.LabelFrame(side, text='③  Objects to avoid', padding=8)
        f.pack(fill='x', pady=(0, 8))
        self.lst = tk.Listbox(f, height=4, exportselection=False, activestyle='none', relief='flat',
                              highlightthickness=1, highlightbackground='#d1d5db')
        self.lst.pack(fill='x')
        self.lst.bind('<<ListboxSelect>>', lambda _e: self.redraw())
        r = ttk.Frame(f)
        r.pack(fill='x', pady=(6, 0))
        ttk.Button(r, text='Add zone', command=self.start_obstacle).pack(side='left', expand=True, fill='x')
        ttk.Button(r, text='Finish', command=self.finish_obstacle).pack(side='left', expand=True, fill='x', padx=4)
        ttk.Button(r, text='Delete', command=self.delete_obstacle).pack(side='left', expand=True, fill='x')

        f = ttk.LabelFrame(side, text='View & real-size check', padding=8)
        f.pack(fill='x')
        ttk.Checkbutton(f, text='Show measuring grid on the table', variable=self.show_grid,
                        command=self.redraw).pack(anchor='w')
        ttk.Checkbutton(f, text='Show real-size ghost (drag it):', variable=self.show_ghost,
                        command=self.redraw).pack(anchor='w', pady=(4, 0))
        keys = list(self.app.sizes.keys())
        self.ghost_key.set('card' if 'card' in keys else (keys[0] if keys else ''))
        cb = ttk.Combobox(f, textvariable=self.ghost_key, values=keys, state='readonly', width=22)
        cb.pack(fill='x', pady=(2, 6))
        cb.bind('<<ComboboxSelected>>', lambda _e: self.redraw())
        ttk.Button(f, text='Fit view', command=self.fit).pack(fill='x')
        ttk.Label(f, text='Wheel = zoom · middle-drag or Shift-drag = pan\nRight-click = undo point · Enter = finish zone',
                  style='Sub.TLabel', justify='left').pack(anchor='w', pady=(6, 0))

        ttk.Label(bar, textvariable=self.status, style='Sub.TLabel', wraplength=760,
                  justify='left').pack(side='left', fill='x', expand=True)
        ttk.Button(bar, text='Close', command=self.on_close).pack(side='right')
        ttk.Button(bar, text='💾  Save profile', style='Accent.TButton', command=self.save).pack(side='right', padx=6)
        ttk.Button(bar, text='🧪  Test placement', command=self.test_placement).pack(side='right')

        c = self.canvas
        mid, right = (3, 2) if self.mac else (2, 3)
        c.bind('<Button-1>', self.on_click)
        c.bind('<B1-Motion>', self.on_motion)
        c.bind('<ButtonRelease-1>', self.on_release)
        c.bind(f'<Button-{right}>', self.on_right)
        c.bind(f'<Button-{mid}>', self.on_pan_start)
        c.bind(f'<B{mid}-Motion>', self.on_pan_move)
        c.bind('<MouseWheel>', self.on_wheel)
        c.bind('<Button-4>', lambda e: self.on_wheel(e, 120))
        c.bind('<Button-5>', lambda e: self.on_wheel(e, -120))
        c.bind('<Configure>', lambda _e: self.redraw())
        c.bind('<Enter>', lambda _e: c.focus_set())
        c.bind('<Return>', lambda _e: self.finish_obstacle())
        c.bind('<Escape>', lambda _e: self.cancel_current())
        c.bind('<Delete>', lambda _e: self.delete_obstacle())
        c.bind('<f>', lambda _e: self.fit())
        self.refresh_list()

    # ---------------------------------------------------------- helpers ----
    def say(self, msg):
        self.status.set(msg)

    def mark_dirty(self):
        self.dirty = True

    def on_size_edit(self):
        self.mark_dirty()
        self.redraw()

    def size_cm(self):
        try:
            w = float(self.var_w.get().replace(',', '.'))
            d = float(self.var_d.get().replace(',', '.'))
            if w > 0 and d > 0:
                return w, d
        except ValueError:
            pass
        return None

    def homography(self):
        cm = self.size_cm()
        if len(self.table) != 4 or not cm or validate_table(self.table, cm):
            return None
        return build_homography(self.table, cm)

    def set_mode(self, mode):
        self.mode = mode
        msgs = {
            'table': 'Click the 4 corners of the placement area clockwise: ① far-left → ② far-right → '
                     '③ near-right → ④ near-left.',
            'idle': 'Corners set. Drag a corner to fine-tune, enter the real size on the right, add zones to avoid, '
                    'then Save.',
            'obstacle': 'Click around an object to avoid (3+ points). Click the first point, press Enter or '
                        '"Finish" to close the zone. Right-click undoes a point, Esc cancels.',
        }
        self.say(msgs[mode])
        self.redraw()

    def refresh_list(self):
        sel = self.lst.curselection()
        self.lst.delete(0, 'end')
        for i, poly in enumerate(self.obstacles):
            self.lst.insert('end', f'Zone {i + 1}  ({len(poly)} points)')
        if sel and sel[0] < len(self.obstacles):
            self.lst.selection_set(sel[0])

    # -------------------------------------------------------- view math ----
    def fit(self, redraw=True):
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        z = min((cw - 16) / self.W, (ch - 16) / self.H)
        self.fit_zoom = self.zoom = z
        self.ox, self.oy = (cw - self.W * z) / 2, (ch - self.H * z) / 2
        self.fitted = True
        if redraw:
            self.redraw()

    def to_image(self, x, y):
        return (max(0.0, min(self.W - 0.001, (x - self.ox) / self.zoom)),
                max(0.0, min(self.H - 0.001, (y - self.oy) / self.zoom)))

    def cv(self, p):
        return p[0] * self.zoom + self.ox, p[1] * self.zoom + self.oy

    # --------------------------------------------------------- drawing -----
    def redraw(self):
        if not hasattr(self, 'canvas') or not self.canvas.winfo_exists():
            return
        c = self.canvas
        c.delete('all')
        cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 20 or ch < 20:
            return
        if self.fitted:
            self.fit(redraw=False)
        z = self.zoom
        ix0, iy0 = max(0.0, -self.ox / z), max(0.0, -self.oy / z)
        ix1, iy1 = min(float(self.W), (cw - self.ox) / z), min(float(self.H), (ch - self.oy) / z)
        if ix1 - ix0 >= 1 and iy1 - iy0 >= 1:
            dw, dh = max(1, int(round((ix1 - ix0) * z))), max(1, int(round((iy1 - iy0) * z)))
            img = self.original.resize((dw, dh), RES.BILINEAR if z < 4 else RES.NEAREST,
                                       box=(ix0, iy0, ix1, iy1))
            self.tkimg = ImageTk.PhotoImage(img)
            c.create_image(int(round(ix0 * z + self.ox)), int(round(iy0 * z + self.oy)),
                           anchor='nw', image=self.tkimg)

        Hm, cm = self.homography(), self.size_cm()
        if Hm is not None and self.show_grid.get():
            tw, td = cm
            step = 10.0
            while tw / step > 24 or td / step > 24:
                step *= 2
            x = 0.0
            while x <= tw + 1e-6:
                a, b = apply_h(Hm, [(x, 0), (x, td)])
                c.create_line(*self.cv(a), *self.cv(b), fill='#22d3ee', dash=(3, 3))
                x += step
            y = 0.0
            while y <= td + 1e-6:
                a, b = apply_h(Hm, [(0, y), (tw, y)])
                c.create_line(*self.cv(a), *self.cv(b), fill='#22d3ee', dash=(3, 3))
                y += step
            c.create_text(10, 10, anchor='nw', fill='#22d3ee', font=('TkDefaultFont', 10, 'bold'),
                          text=f'grid = {step:g} cm')

        if self.table:
            pts = [self.cv(p) for p in self.table]
            if len(pts) >= 2:
                c.create_line(*flat(pts + ([pts[0]] if len(pts) == 4 else [])), fill='#22c55e', width=2)
            for i, p in enumerate(pts):
                c.create_oval(p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6, fill='#22c55e', outline='white', width=2)
                c.create_text(p[0] + 12, p[1] - 12, text=str(i + 1), fill='white', font=('TkDefaultFont', 11, 'bold'))
            if len(pts) == 4 and cm:
                mx, my = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
                c.create_text(mx, my - 14, text=f'W {cm[0]:g} cm', fill='#86efac', font=('TkDefaultFont', 10, 'bold'))
                mx, my = (pts[0][0] + pts[3][0]) / 2, (pts[0][1] + pts[3][1]) / 2
                c.create_text(mx - 34, my, text=f'D {cm[1]:g} cm', fill='#86efac', font=('TkDefaultFont', 10, 'bold'))

        sel = self.lst.curselection()
        for i, poly in enumerate(self.obstacles):
            pts = [self.cv(p) for p in poly]
            hot = bool(sel) and sel[0] == i
            c.create_polygon(*flat(pts), outline='#facc15' if hot else '#ef4444', fill='#ef4444',
                             stipple='gray25', width=3 if hot else 2)
            c.create_text(pts[0][0], pts[0][1] - 10, text=f'{i + 1}', fill='white')
        if self.current:
            pts = [self.cv(p) for p in self.current]
            if len(pts) > 1:
                c.create_line(*flat(pts), fill='#facc15', width=2)
            for p in pts:
                c.create_oval(p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4, fill='#facc15')

        if Hm is not None and self.show_ghost.get():
            q = self.ghost_quad(Hm)
            if q is not None:
                pts = [self.cv(p) for p in q]
                c.create_polygon(*flat(pts), outline='#facc15', fill='#facc15', stipple='gray25', width=2)
                w, h = self.ghost_size()
                c.create_text(sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4, fill='white',
                              text=f'{w:g} × {h:g} cm', font=('TkDefaultFont', 9, 'bold'))

    # ----------------------------------------------------------- ghost -----
    def ghost_size(self):
        v = self.app.sizes.get(self.ghost_key.get())
        if not v:
            return 8.56, 5.40
        return max(v), min(v)

    def ghost_quad(self, Hm):
        cm = self.size_cm()
        if cm is None:
            return None
        if self.ghost_c is None:
            self.ghost_c = (cm[0] / 2, cm[1] / 2)
        w, h = self.ghost_size()
        return apply_h(Hm, footprint_plane(self.ghost_c[0], self.ghost_c[1], w, h, 0))

    def to_plane(self, Hm, ip):
        return apply_h(np.linalg.inv(Hm), [ip])[0]

    # ----------------------------------------------------- mouse events ----
    def hit_corner(self, x, y):
        best, bi = 13.0, None
        for i, p in enumerate(self.table):
            cx, cy = self.cv(p)
            d = math.hypot(cx - x, cy - y)
            if d < best:
                best, bi = d, i
        return bi

    def on_click(self, e):
        self.canvas.focus_set()
        if e.state & 0x0001:                                # Shift = pan
            self.on_pan_start(e)
            return
        p = self.to_image(e.x, e.y)
        if len(self.table) == 4 and self.mode in ('table', 'idle'):
            i = self.hit_corner(e.x, e.y)
            if i is not None:
                self.drag = ('corner', i)
                return
        Hm = self.homography()
        if self.mode == 'idle' and self.show_ghost.get() and Hm is not None:
            q = self.ghost_quad(Hm)
            if q is not None and cv2.pointPolygonTest(np.float32(q).reshape(-1, 1, 2), p, False) >= 0:
                pp = self.to_plane(Hm, p)
                self.drag = ('ghost', (self.ghost_c[0] - pp[0], self.ghost_c[1] - pp[1]))
                return
        if self.mode == 'table':
            if len(self.table) < 4:
                self.table.append(p)
                self.mark_dirty()
                if len(self.table) == 4:
                    self.set_mode('idle')
                    self.say('Corners set. Now enter the real width and depth (right panel) – '
                             'tip: "Estimate depth" can fill the depth from the perspective.')
                    return
                self.say(f'Corner {len(self.table)} placed – click corner {len(self.table) + 1}.')
        elif self.mode == 'obstacle':
            if len(self.current) >= 3 and math.hypot(*(np.array(self.cv(self.current[0])) - (e.x, e.y))) < 12:
                self.finish_obstacle()
                return
            self.current.append(p)
        self.redraw()

    def on_motion(self, e):
        if self.pan:
            self.on_pan_move(e)
            return
        if not self.drag:
            return
        kind, val = self.drag
        p = self.to_image(e.x, e.y)
        if kind == 'corner':
            self.table[val] = p
            self.mark_dirty()
        elif kind == 'ghost':
            Hm = self.homography()
            if Hm is not None:
                pp = self.to_plane(Hm, p)
                self.ghost_c = (pp[0] + val[0], pp[1] + val[1])
        self.redraw()

    def on_release(self, _e):
        self.drag = None
        self.pan = None

    def on_right(self, _e):
        if self.current:
            self.current.pop()
        elif self.mode == 'table' and self.table and len(self.table) < 4:
            self.table.pop()
        self.redraw()

    def on_pan_start(self, e):
        self.pan = (e.x, e.y, self.ox, self.oy)

    def on_pan_move(self, e):
        if not self.pan:
            return
        x0, y0, ox0, oy0 = self.pan
        self.ox, self.oy = ox0 + (e.x - x0), oy0 + (e.y - y0)
        self.fitted = False
        self.redraw()

    def on_wheel(self, e, delta=None):
        d = delta if delta is not None else e.delta
        f = 1.2 if d > 0 else 1 / 1.2
        nz = max(self.fit_zoom * 0.5, min(40.0, self.zoom * f))
        ix, iy = (e.x - self.ox) / self.zoom, (e.y - self.oy) / self.zoom
        self.ox, self.oy, self.zoom = e.x - ix * nz, e.y - iy * nz, nz
        self.fitted = False
        self.redraw()

    # ------------------------------------------------------- actions -------
    def start_table(self):
        if len(self.table) == 4 and not messagebox.askyesno(
                'Re-mark corners', 'Clear the current tabletop corners and mark them again?', parent=self):
            return
        self.table, self.current = [], []
        self.mark_dirty()
        self.set_mode('table')

    def start_obstacle(self):
        if len(self.table) != 4:
            messagebox.showwarning('Tabletop first', 'Mark the 4 tabletop corners first.', parent=self)
            return
        self.current = []
        self.set_mode('obstacle')

    def finish_obstacle(self):
        if self.mode != 'obstacle':
            return
        if len(self.current) < 3:
            messagebox.showinfo('More points', 'A zone needs at least 3 points.', parent=self)
            return
        self.obstacles.append(self.current[:])
        self.current = []
        self.mark_dirty()
        self.refresh_list()
        self.set_mode('idle')
        self.say(f'Zone {len(self.obstacles)} saved. Use "Add zone" for another, or Save profile.')

    def cancel_current(self):
        self.current = []
        if self.mode == 'obstacle':
            self.set_mode('idle')
        self.redraw()

    def delete_obstacle(self):
        sel = self.lst.curselection()
        if not sel:
            return
        del self.obstacles[sel[0]]
        self.mark_dirty()
        self.refresh_list()
        self.redraw()

    def estimate_depth(self):
        if len(self.table) != 4:
            messagebox.showinfo('Corners needed', 'Mark the 4 tabletop corners first.', parent=self)
            return
        try:
            w = float(self.var_w.get().replace(',', '.'))
        except ValueError:
            messagebox.showinfo('Width needed', 'Type the real WIDTH (cm) first; the depth is estimated from it.',
                                parent=self)
            return
        r, method = estimate_aspect(self.table, self.W, self.H)
        if r is None:
            messagebox.showinfo('Cannot estimate',
                                'The corners are too inconsistent to estimate the depth. '
                                'Please measure it with a tape measure.', parent=self)
            return
        self.var_d.set(f'{w * r:.1f}')
        note = ('' if method == 'solved' else ' The camera looks level/centred, so a typical phone focal length '
                'was ASSUMED – treat this as a rough guess.')
        self.say(f'Estimated depth ≈ {w * r:.1f} cm (assumes an un-cropped photo).{note} '
                 f'Verify with the ghost card or a tape measure.')

    def validate(self):
        cm = self.size_cm()
        if cm is None and len(self.table) == 4:
            return 'Enter the real width and depth (cm) of the marked area.'
        return validate_table(self.table, cm)

    def profile_data(self):
        w, d = self.size_cm()
        return {'version': 2, 'image': str(self.path.resolve()), 'size': [self.W, self.H],
                'table': [list(p) for p in self.table], 'table_cm': [w, d],
                'obstacles': [[list(p) for p in poly] for poly in self.obstacles]}

    def save(self):
        err = self.validate()
        if err:
            messagebox.showerror('Cannot save yet', err, parent=self)
            return False
        r, method = estimate_aspect(self.table, self.W, self.H)
        w, d = self.size_cm()
        if r and method == 'solved' and not (0.6 <= (d / w) / r <= 1.6):
            if not messagebox.askyesno(
                    'Check the size', f'You entered {w:g} × {d:g} cm (depth/width = {d / w:.2f}), but the perspective '
                                      f'suggests about {r:.2f}.\n\nIf you measured it, or the photo is cropped, the '
                                      f'numbers may still be right. Did you swap width and depth?\n\nSave anyway?',
                    parent=self):
                return False
        try:
            save_profile(self.app.v_profiles.get(), self.path, self.profile_data())
        except Exception as e:
            log_error('save profile', self.path, e)
            messagebox.showerror('Save failed', str(e), parent=self)
            return False
        self.dirty = False
        self.say(f'Saved → {profile_file(self.app.v_profiles.get(), self.path)}')
        self.app.refresh_backgrounds()
        return True

    def test_placement(self):
        err = self.validate()
        if err:
            messagebox.showerror('Cannot test yet', err, parent=self)
            return
        try:
            opt = self.app.collect_settings()
        except ValueError as e:
            messagebox.showerror('Check settings', str(e), parent=self)
            return
        mockups = list_mockups(opt['mock_dir'])
        cmap = class_map(mockups, opt['mock_dir'])
        if not mockups:
            messagebox.showinfo('No mock-ups', f'No mock-up images found in:\n{opt["mock_dir"]}', parent=self)
            return
        prof = self.profile_data()
        n = opt['objects'] if opt['dupes'] else min(opt['objects'], len(mockups))

        def make():
            rng = random.Random()
            chosen = pick_mockups(rng, mockups, n, dupes=opt['dupes'], class_of=cmap)
            return compose_image(self.path, prof, chosen, opt['sizes'], opt, rng)

        PreviewWindow(self, make)

    def on_close(self):
        if self.dirty:
            r = messagebox.askyesnocancel('Unsaved changes', 'Save this background profile before closing?',
                                          parent=self)
            if r is None:
                return
            if r and not self.save():
                return
        self.app.marking = None
        self.destroy()


# =============================================================================
#  Main application window
# =============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry('1040x760')
        self.minsize(900, 660)
        self.settings = load_settings()
        self.sizes = {k: list(v) for k, v in self.settings['sizes'].items()}
        self.marking = None
        self.worker = None
        self.q = None
        self.cancel_ev = threading.Event()
        S = self.settings

        self.v_bg, self.v_mock = tk.StringVar(value=S['bg_dir']), tk.StringVar(value=S['mock_dir'])
        self.v_out, self.v_profiles = tk.StringVar(value=S['out_dir']), tk.StringVar(value=S['profile_dir'])
        sv = lambda k: tk.StringVar(value=str(S[k]))          # noqa: E731
        bv = lambda k: tk.BooleanVar(value=bool(S[k]))        # noqa: E731
        self.v_count, self.v_objects = sv('count'), sv('objects')
        self.v_rot, self.v_margin, self.v_gap, self.v_jitter = sv('rot'), sv('margin'), sv('gap'), sv('jitter')
        self.v_fallback, self.v_strength, self.v_quality, self.v_seed = sv('fallback'), sv('strength'), sv('quality'), sv('seed')
        self.v_dupes, self.v_keep, self.v_shadow = bv('dupes'), bv('keep_aspect'), bv('shadow')
        self.v_light, self.v_fx, self.v_labels = bv('light'), bv('fx'), bv('labels')

        self.setup_style()
        self.build_ui()
        self.protocol('WM_DELETE_WINDOW', self.on_close)
        self.v_fallback.trace_add('write', lambda *_: self.after_idle(self.refresh_mockups))
        self.v_keep.trace_add('write', lambda *_: self.after_idle(self.refresh_mockups))
        self.refresh_backgrounds()
        self.refresh_sizes()
        self.refresh_mockups()

    # ------------------------------------------------------------ style ----
    def setup_style(self):
        fams = set(tkfont.families())
        fam = next((f for f in ('Segoe UI', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans', 'Arial') if f in fams),
                   tkfont.nametofont('TkDefaultFont').actual('family'))
        for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
            try:
                tkfont.nametofont(name).configure(family=fam, size=10)
            except tk.TclError:
                pass
        self.configure(bg=BG)
        st = ttk.Style(self)
        try:
            st.theme_use('clam')
        except tk.TclError:
            pass
        st.configure('.', background=BG, foreground=TEXT, font=(fam, 10))
        st.configure('TFrame', background=BG)
        st.configure('TLabel', background=BG, foreground=TEXT)
        st.configure('Sub.TLabel', foreground=SUB)
        st.configure('H1.TLabel', font=(fam, 18, 'bold'))
        st.configure('TLabelframe', background=BG, bordercolor='#d1d5db', relief='solid')
        st.configure('TLabelframe.Label', background=BG, foreground=TEXT, font=(fam, 10, 'bold'))
        st.configure('TButton', padding=(12, 6), background='#e5e7eb', bordercolor='#d1d5db', focusthickness=0)
        st.map('TButton', background=[('active', '#d1d5db'), ('disabled', '#f3f4f6')],
               foreground=[('disabled', '#9ca3af')])
        st.configure('Accent.TButton', background=ACCENT, foreground='white', bordercolor=ACCENT)
        st.map('Accent.TButton', background=[('active', ACCENT_D), ('disabled', '#93c5fd')],
               foreground=[('disabled', '#eff6ff')])
        st.configure('TEntry', padding=4, fieldbackground='white')
        st.configure('TCombobox', padding=4, fieldbackground='white')
        st.configure('TCheckbutton', background=BG)
        st.configure('Treeview', rowheight=28, background='white', fieldbackground='white', bordercolor='#d1d5db')
        st.configure('Treeview.Heading', font=(fam, 10, 'bold'), padding=6, background='#e5e7eb')
        st.map('Treeview', background=[('selected', '#dbeafe')], foreground=[('selected', TEXT)])
        st.configure('TNotebook', background=BG, borderwidth=0)
        st.configure('TNotebook.Tab', padding=(18, 8), font=(fam, 10, 'bold'), background='#e5e7eb')
        st.map('TNotebook.Tab', background=[('selected', 'white')])
        st.configure('Horizontal.TProgressbar', troughcolor='#e5e7eb', background=ACCENT, thickness=14)
        self.mono = ('Consolas' if 'Consolas' in fams else 'DejaVu Sans Mono', 9)

    # --------------------------------------------------------------- UI ----
    def build_ui(self):
        head = ttk.Frame(self, padding=(18, 14, 18, 4))
        head.pack(fill='x')
        ttk.Label(head, text=APP_TITLE, style='H1.TLabel').pack(anchor='w')
        ttk.Label(head, text='Real-size mock-up placement with 4-point perspective:  1) mark backgrounds  →  '
                             '2) check sizes  →  3) generate',
                  style='Sub.TLabel').pack(anchor='w', pady=(2, 0))
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=14, pady=10)
        self.tab1, self.tab2, self.tab3 = (ttk.Frame(self.nb, padding=12) for _ in range(3))
        self.nb.add(self.tab1, text='1 · Backgrounds')
        self.nb.add(self.tab2, text='2 · Mock-ups & sizes')
        self.nb.add(self.tab3, text='3 · Generate')
        self.build_tab_backgrounds(self.tab1)
        self.build_tab_mockups(self.tab2)
        self.build_tab_generate(self.tab3)

    def path_row(self, parent, row, label, var, cmd):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(0, 10), pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky='ew', pady=3)
        ttk.Button(parent, text='Browse…', command=cmd).grid(row=row, column=2, padx=(6, 0), pady=3)
        parent.columnconfigure(1, weight=1)

    def field(self, parent, row, label, var, unit='', width=8, col=0):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky='w', pady=4, padx=(0, 8))
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky='w', pady=4)
        if unit:
            ttk.Label(parent, text=unit, style='Sub.TLabel').grid(row=row, column=col + 2, sticky='w', padx=(6, 0))

    def pick_dir(self, var, after=None):
        d = var.get().strip()
        p = filedialog.askdirectory(parent=self, initialdir=d if Path(d).is_dir() else None)
        if p:
            var.set(os.path.normpath(p))
            if after:
                after()

    # ---- tab 1 ----
    def build_tab_backgrounds(self, tab):
        top = ttk.Frame(tab)
        top.pack(fill='x')
        self.path_row(top, 0, 'Backgrounds folder', self.v_bg, lambda: self.pick_dir(self.v_bg, self.refresh_backgrounds))
        self.path_row(top, 1, 'Profiles folder', self.v_profiles, lambda: self.pick_dir(self.v_profiles, self.refresh_backgrounds))
        ttk.Label(tab, style='Sub.TLabel', wraplength=940, justify='left',
                  text='Each background needs a profile: the 4 tabletop corners, the REAL size of that area in cm '
                       '(so mock-ups come out true to scale), and optional zones to avoid. Double-click a row to mark it. '
                       'Profiles are matched to backgrounds by file name.').pack(anchor='w', pady=(6, 8))
        mid = ttk.Frame(tab)
        mid.pack(fill='both', expand=True)
        cols = (('name', 'Background', 330), ('res', 'Resolution', 110), ('status', 'Profile', 150),
                ('table', 'Table size', 120), ('obs', 'Avoid zones', 90))
        self.tv_bg = ttk.Treeview(mid, columns=[c[0] for c in cols], show='headings', selectmode='browse')
        for key, text, w in cols:
            self.tv_bg.heading(key, text=text)
            self.tv_bg.column(key, width=w, anchor='w' if key == 'name' else 'center')
        sb = ttk.Scrollbar(mid, orient='vertical', command=self.tv_bg.yview)
        self.tv_bg.configure(yscrollcommand=sb.set)
        self.tv_bg.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        for tag, col in (('ok', C_OK), ('size', C_WARN), ('bad', C_WARN), ('none', C_BAD)):
            self.tv_bg.tag_configure(tag, foreground=col)
        self.tv_bg.bind('<Double-1>', lambda _e: self.open_marking())
        bar = ttk.Frame(tab)
        bar.pack(fill='x', pady=(10, 0))
        ttk.Button(bar, text='✎  Mark / edit selected', style='Accent.TButton', command=self.open_marking).pack(side='left')
        ttk.Button(bar, text='Delete profile', command=self.delete_profile).pack(side='left', padx=6)
        ttk.Button(bar, text='⟳  Refresh', command=self.refresh_backgrounds).pack(side='left')
        self.lbl_bg = ttk.Label(bar, style='Sub.TLabel')
        self.lbl_bg.pack(side='right')

    def refresh_backgrounds(self):
        sel = self.tv_bg.selection()
        self.tv_bg.delete(*self.tv_bg.get_children())
        imgs = list_images(self.v_bg.get())
        ready = 0
        stems = [p.stem.lower() for p in imgs]
        for p in imgs:
            try:
                res = '%d × %d' % image_size(p)
            except Exception:
                res = '?'
            try:
                prof = load_profile(self.v_profiles.get(), p)
            except Exception:
                prof = None
            code, label = profile_status(prof)
            if stems.count(p.stem.lower()) > 1:
                label += '  (duplicate name!)'
            ready += code == 'ok'
            cm = prof.get('table_cm') if prof else None
            self.tv_bg.insert('', 'end', iid=str(p), tags=(code,), values=(
                p.name, res, label, f'{cm[0]:g} × {cm[1]:g} cm' if cm else '—',
                len(prof.get('obstacles', [])) if prof else '—'))
        for s in sel:
            if self.tv_bg.exists(s):
                self.tv_bg.selection_set(s)
        if not imgs:
            self.lbl_bg.config(text='No images found in the backgrounds folder.')
        else:
            self.lbl_bg.config(text=f'{ready} of {len(imgs)} ready')

    def open_marking(self):
        sel = self.tv_bg.selection()
        if not sel:
            messagebox.showinfo('Select a background', 'Select a background in the list first.')
            return
        if self.marking is not None and self.marking.winfo_exists():
            self.marking.lift()
            messagebox.showinfo('Already open', 'Finish or close the open marking window first.')
            return
        try:
            self.marking = MarkingWindow(self, sel[0])
        except Exception as e:
            log_error('open marking window', sel[0], e)
            self.marking = None
            messagebox.showerror('Could not open image', f'{e}\n\nSee log: {LOG_PATH}')

    def delete_profile(self):
        sel = self.tv_bg.selection()
        if not sel:
            return
        p = profile_file(self.v_profiles.get(), sel[0])
        if p.is_file() and messagebox.askyesno('Delete profile', f'Delete the profile for {Path(sel[0]).name}?'):
            p.unlink()
            self.refresh_backgrounds()

    # ---- tab 2 ----
    def build_tab_mockups(self, tab):
        top = ttk.Frame(tab)
        top.pack(fill='x')
        self.path_row(top, 0, 'Mock-ups folder', self.v_mock, lambda: self.pick_dir(self.v_mock, self.refresh_mockups))
        pw = ttk.PanedWindow(tab, orient='horizontal')
        pw.pack(fill='both', expand=True, pady=8)
        left = ttk.LabelFrame(pw, text='Mock-up files  (class = sub-folder name)', padding=8)
        right = ttk.LabelFrame(pw, text='Real-world sizes by file-name keyword', padding=8)
        pw.add(left, weight=3)
        pw.add(right, weight=2)

        cols = (('file', 'File', 190), ('px', 'Image (px)', 90), ('cls', 'Class (folder)', 110), ('size', 'Real size (cm)', 100),
                ('note', 'Note', 180))
        self.tv_mock = ttk.Treeview(left, columns=[c[0] for c in cols], show='headings', selectmode='browse')
        for key, text, w in cols:
            self.tv_mock.heading(key, text=text)
            self.tv_mock.column(key, width=w, anchor='w')
        sb = ttk.Scrollbar(left, orient='vertical', command=self.tv_mock.yview)
        self.tv_mock.configure(yscrollcommand=sb.set)
        self.tv_mock.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.tv_mock.tag_configure('warn', foreground=C_WARN)
        self.tv_mock.tag_configure('bad', foreground=C_BAD)

        self.tv_size = ttk.Treeview(right, columns=('k', 'w', 'h'), show='headings', selectmode='browse', height=8)
        for key, text, w in (('k', 'Keyword', 110), ('w', 'Width cm', 80), ('h', 'Height cm', 80)):
            self.tv_size.heading(key, text=text)
            self.tv_size.column(key, width=w, anchor='w' if key == 'k' else 'center')
        self.tv_size.pack(fill='both', expand=True)
        self.tv_size.bind('<Double-1>', lambda _e: self.edit_size())
        r = ttk.Frame(right)
        r.pack(fill='x', pady=(8, 0))
        ttk.Button(r, text='Add…', command=self.add_size).pack(side='left')
        ttk.Button(r, text='Edit…', command=self.edit_size).pack(side='left', padx=4)
        ttk.Button(r, text='Remove', command=self.remove_size).pack(side='left')
        ttk.Button(r, text='Reset', command=self.reset_sizes).pack(side='right')
        ttk.Label(right, style='Sub.TLabel', wraplength=330, justify='left',
                  text='Defaults: ISO ID-1 cards 8.56 × 5.40 cm, passport 12.5 × 8.8 cm. Orientation follows the '
                       'PNG (landscape stays landscape). The longest matching keyword wins.').pack(anchor='w', pady=(8, 0))

        self.lbl_classes = ttk.Label(tab, style='Sub.TLabel', wraplength=940, justify='left')
        self.lbl_classes.pack(anchor='w', pady=(0, 6))
        bot = ttk.Frame(tab)
        bot.pack(fill='x')
        ttk.Label(bot, text='Fallback when no keyword matches: long side').pack(side='left')
        ttk.Entry(bot, textvariable=self.v_fallback, width=7).pack(side='left', padx=6)
        ttk.Label(bot, text='cm').pack(side='left')
        ttk.Checkbutton(bot, text='Keep each image\'s own aspect ratio (scale by long side only)',
                        variable=self.v_keep).pack(side='left', padx=(20, 0))
        ttk.Button(bot, text='⟳  Refresh', command=self.refresh_mockups).pack(side='right')

    def safe_float(self, var, default):
        try:
            v = float(var.get().replace(',', '.'))
            return v if v > 0 else default
        except ValueError:
            return default

    def refresh_mockups(self):
        if not hasattr(self, 'tv_mock'):
            return
        self.tv_mock.delete(*self.tv_mock.get_children())
        fb = self.safe_float(self.v_fallback, 8.56)
        root = self.v_mock.get().strip()
        files = list_mockups(root)
        counts = {}
        for p in files:
            cls = mockup_class(p, root)
            counts[cls] = counts.get(cls, 0) + 1
            rel = str(p.relative_to(root)) if Path(root) in p.parents else p.name
            try:
                fg = load_mockup(p)
                w, h, key, warns = resolve_real_size(p, fg.size, self.sizes, fb, self.v_keep.get(), cls)
                self.tv_mock.insert('', 'end', tags=('warn',) if warns else (), values=(
                    rel, f'{fg.width} × {fg.height}', cls, f'{w:.2f} × {h:.2f}', '; '.join(warns)))
            except Exception as e:
                self.tv_mock.insert('', 'end', tags=('bad',), values=(rel, '?', cls, '—', str(e)))
        if hasattr(self, 'lbl_classes'):
            self.lbl_classes.config(text=(f'{len(counts)} class(es): ' + ',  '.join(
                f'{n} ({c})' for n, c in sorted(counts.items()))) if counts else 'No mock-ups found.')

    def refresh_sizes(self):
        self.tv_size.delete(*self.tv_size.get_children())
        for k in sorted(self.sizes):
            self.tv_size.insert('', 'end', iid=k, values=(k, f'{self.sizes[k][0]:g}', f'{self.sizes[k][1]:g}'))

    def add_size(self):
        d = SizeDialog(self, 'Add size')
        self.wait_window(d)
        if d.result:
            k, w, h = d.result
            self.sizes[k] = [w, h]
            self.after_sizes_changed()

    def edit_size(self):
        sel = self.tv_size.selection()
        if not sel:
            return
        k = sel[0]
        d = SizeDialog(self, 'Edit size', k, self.sizes[k][0], self.sizes[k][1])
        self.wait_window(d)
        if d.result:
            nk, w, h = d.result
            if nk != k:
                del self.sizes[k]
            self.sizes[nk] = [w, h]
            self.after_sizes_changed()

    def remove_size(self):
        sel = self.tv_size.selection()
        if sel:
            self.sizes.pop(sel[0], None)
            self.after_sizes_changed()

    def reset_sizes(self):
        if messagebox.askyesno('Reset sizes', 'Restore the default real-world sizes?'):
            self.sizes = {k: list(v) for k, v in DEFAULT_SIZES.items()}
            self.after_sizes_changed()

    def after_sizes_changed(self):
        self.refresh_sizes()
        self.refresh_mockups()
        self.save_settings()

    # ---- tab 3 ----
    def build_tab_generate(self, tab):
        tab.columnconfigure(0, weight=1, uniform='g')
        tab.columnconfigure(1, weight=1, uniform='g')
        tab.rowconfigure(6, weight=1)
        f = ttk.LabelFrame(tab, text='Output', padding=10)
        f.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        self.path_row(f, 0, 'Output folder', self.v_out, lambda: self.pick_dir(self.v_out))

        f = ttk.LabelFrame(tab, text='How many', padding=10)
        f.grid(row=1, column=0, sticky='nsew', padx=(0, 6), pady=(0, 8))
        self.field(f, 0, 'Composites per mock-up', self.v_count)
        self.field(f, 1, 'Mock-ups per image', self.v_objects)
        ttk.Checkbutton(f, text='Allow the same mock-up twice in one image', variable=self.v_dupes).grid(
            row=2, column=0, columnspan=3, sticky='w', pady=(4, 0))
        self.field(f, 3, 'Random seed (optional)', self.v_seed, width=12)
        self.field(f, 4, 'JPEG quality', self.v_quality, '(70–100)')
        ttk.Checkbutton(f, text='Save YOLO labels (class = mock-up sub-folder)', variable=self.v_labels).grid(
            row=5, column=0, columnspan=3, sticky='w', pady=(4, 0))

        f = ttk.LabelFrame(tab, text='Placement (real-world units)', padding=10)
        f.grid(row=1, column=1, sticky='nsew', padx=(6, 0), pady=(0, 8))
        self.field(f, 0, 'Random rotation ±', self.v_rot, 'degrees')
        self.field(f, 1, 'Distance from table edge', self.v_margin, 'cm')
        self.field(f, 2, 'Minimum gap between items', self.v_gap, 'cm')
        self.field(f, 3, 'Size variation ±', self.v_jitter, '%  (0 = exact real size)')

        f = ttk.LabelFrame(tab, text='Realism', padding=10)
        f.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Checkbutton(f, text='Soft contact + ambient shadows', variable=self.v_shadow).grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(f, text='Camera blur & grain', variable=self.v_fx).grid(row=0, column=1, sticky='w', padx=20)
        ttk.Checkbutton(f, text='Match scene lighting / colour, strength', variable=self.v_light).grid(row=0, column=2, sticky='w')
        ttk.Entry(f, textvariable=self.v_strength, width=5).grid(row=0, column=3, padx=6)
        ttk.Label(f, text='%', style='Sub.TLabel').grid(row=0, column=4)

        act = ttk.Frame(tab)
        act.grid(row=3, column=0, columnspan=2, sticky='ew')
        self.btn_gen = ttk.Button(act, text='▶   Generate composites', style='Accent.TButton', command=self.start_generation)
        self.btn_gen.pack(side='left')
        self.btn_cancel = ttk.Button(act, text='Cancel', command=self.cancel_generation, state='disabled')
        self.btn_cancel.pack(side='left', padx=6)
        ttk.Button(act, text='Open error log', command=self.open_log).pack(side='right')
        ttk.Button(act, text='Open output folder', command=lambda: self.open_out()).pack(side='right', padx=6)

        self.progress = ttk.Progressbar(tab, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(12, 4))
        self.lbl_status = ttk.Label(tab, text='Ready. Mark your backgrounds first (tab 1), then generate.', style='Sub.TLabel')
        self.lbl_status.grid(row=5, column=0, columnspan=2, sticky='w')
        lf = ttk.Frame(tab)
        lf.grid(row=6, column=0, columnspan=2, sticky='nsew', pady=(8, 0))
        self.txt = tk.Text(lf, height=8, state='disabled', wrap='word', bg='#111827', fg='#e5e7eb',
                           font=self.mono, relief='flat', padx=8, pady=6)
        sb = ttk.Scrollbar(lf, orient='vertical', command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.txt.tag_configure('ok', foreground='#86efac')
        self.txt.tag_configure('warn', foreground='#fcd34d')
        self.txt.tag_configure('err', foreground='#fca5a5')

    def log(self, msg, level='info'):
        self.txt.configure(state='normal')
        self.txt.insert('end', msg + '\n', level)
        self.txt.see('end')
        self.txt.configure(state='disabled')

    def open_log(self):
        if not LOG_PATH.exists():
            LOG_PATH.write_text('No errors logged yet.\n', encoding='utf-8')
        open_path(LOG_PATH)

    def open_out(self):
        p = Path(self.v_out.get())
        p.mkdir(parents=True, exist_ok=True)
        open_path(p)

    # ---------------------------------------------------------- settings ---
    def collect_settings(self):
        def num(var, label, lo, hi, integer=False):
            s = var.get().strip().replace(',', '.')
            try:
                v = float(s)
            except ValueError:
                raise ValueError(f'{label}: "{s}" is not a number.')
            if integer:
                v = int(round(v))
            if not (lo <= v <= hi):
                raise ValueError(f'{label} must be between {lo:g} and {hi:g}.')
            return v

        seed = self.v_seed.get().strip()
        if seed:
            try:
                int(seed)
            except ValueError:
                raise ValueError('Random seed must be a whole number (or empty).')
        return {
            'bg_dir': self.v_bg.get().strip(), 'mock_dir': self.v_mock.get().strip(),
            'out_dir': self.v_out.get().strip(), 'profile_dir': self.v_profiles.get().strip(),
            'count': num(self.v_count, 'Composites per mock-up', 1, 5000, True),
            'objects': num(self.v_objects, 'Mock-ups per image', 1, 20, True),
            'dupes': self.v_dupes.get(),
            'rot': num(self.v_rot, 'Random rotation', 0, 180),
            'margin': num(self.v_margin, 'Distance from table edge', 0, 100),
            'gap': num(self.v_gap, 'Minimum gap', 0, 100),
            'jitter': num(self.v_jitter, 'Size variation', 0, 30),
            'keep_aspect': self.v_keep.get(),
            'fallback': num(self.v_fallback, 'Fallback long side', 0.5, 200),
            'shadow': self.v_shadow.get(), 'light': self.v_light.get(), 'fx': self.v_fx.get(),
            'strength': num(self.v_strength, 'Lighting strength', 0, 100),
            'quality': num(self.v_quality, 'JPEG quality', 50, 100, True),
            'seed': seed, 'labels': self.v_labels.get(), 'sizes': {k: list(v) for k, v in self.sizes.items()},
        }

    def save_settings(self):
        data = {
            'bg_dir': self.v_bg.get(), 'mock_dir': self.v_mock.get(), 'out_dir': self.v_out.get(),
            'profile_dir': self.v_profiles.get(),
            'count': self.v_count.get(), 'objects': self.v_objects.get(), 'dupes': self.v_dupes.get(),
            'rot': self.v_rot.get(), 'margin': self.v_margin.get(), 'gap': self.v_gap.get(),
            'jitter': self.v_jitter.get(), 'keep_aspect': self.v_keep.get(), 'fallback': self.v_fallback.get(),
            'shadow': self.v_shadow.get(), 'light': self.v_light.get(), 'strength': self.v_strength.get(),
            'fx': self.v_fx.get(), 'quality': self.v_quality.get(), 'seed': self.v_seed.get(),
            'labels': self.v_labels.get(),
            'sizes': self.sizes,
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception as e:
            log_error('save settings', SETTINGS_PATH, e)

    # -------------------------------------------------------- generation ---
    def start_generation(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self.collect_settings()
        except ValueError as e:
            messagebox.showerror('Check settings', str(e))
            return
        if not Path(cfg['bg_dir']).is_dir() or not Path(cfg['mock_dir']).is_dir() or not cfg['out_dir']:
            messagebox.showerror('Folders', 'Choose valid Backgrounds, Mock-ups and Output folders.')
            return
        self.save_settings()
        self.txt.configure(state='normal')
        self.txt.delete('1.0', 'end')
        self.txt.configure(state='disabled')
        self.q = queue.Queue()
        self.cancel_ev = threading.Event()
        self.worker = GenerationWorker(cfg, self.q, self.cancel_ev)
        self.btn_gen.config(state='disabled')
        self.btn_cancel.config(state='normal')
        self.lbl_status.config(text='Starting…')
        self.worker.start()
        self.after(100, self.poll)

    def cancel_generation(self):
        self.cancel_ev.set()
        self.lbl_status.config(text='Cancelling after the current image…')

    def poll(self):
        finished = False
        try:
            while True:
                m = self.q.get_nowait()
                kind = m[0]
                if kind == 'log':
                    self.log(m[1], m[2])
                elif kind == 'progress':
                    _, done, total, ok, fail = m
                    self.progress.configure(maximum=max(total, 1), value=done)
                    self.lbl_status.config(text=f'Generating… {done}/{total}   ✔ {ok}   ✖ {fail}')
                elif kind == 'done':
                    _, ok, fail, out, cancelled = m
                    word = 'Cancelled' if cancelled else 'Finished'
                    self.lbl_status.config(text=f'{word}: {ok} created, {fail} failed → {out}')
                    self.log(f'{word}: {ok} created, {fail} failed.  Output: {out}', 'ok' if not fail else 'warn')
                    finished = True
                elif kind == 'fatal':
                    self.lbl_status.config(text='Stopped.')
                    self.log(m[1], 'err')
                    messagebox.showerror('Cannot generate', m[1])
                    finished = True
        except queue.Empty:
            pass
        if finished or not (self.worker and self.worker.is_alive()):
            if not self.q.empty():
                self.after(50, self.poll)
                return
            self.btn_gen.config(state='normal')
            self.btn_cancel.config(state='disabled')
            return
        self.after(100, self.poll)

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno('Generation running', 'Generation is still running. Quit anyway?'):
                return
            self.cancel_ev.set()
        if self.marking is not None and self.marking.winfo_exists() and self.marking.dirty:
            self.marking.lift()
            if not messagebox.askyesno('Unsaved profile', 'A marking window has unsaved changes. Quit anyway?'):
                return
        self.save_settings()
        self.destroy()


def main():
    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)        # crisp UI on high-DPI screens
        except Exception:
            pass
    App().mainloop()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(traceback.format_exc())
        raise