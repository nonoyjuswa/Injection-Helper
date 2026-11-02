from pathlib import Path
import shutil, sys, csv

OUT = Path(r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Corrected")
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

args = sys.argv[1:]
MOVE = "--move" in args                                   # default is copy
SEARCH = [Path(a) for a in args if not a.startswith("--")] or [OUT.parent]

Q = OUT / "_orphans"
(Q / "images").mkdir(parents=True, exist_ok=True)
(Q / "labels").mkdir(parents=True, exist_ok=True)

def files(folder, exts):
    return {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts} if folder.is_dir() else {}

# Pool the live folders and _orphans, so this works whether or not you already ran --move
images = {**files(OUT / "images", IMG_EXTS), **files(Q / "images", IMG_EXTS)}
labels = {**files(OUT / "labels", {'.txt'}), **files(Q / "labels", {'.txt'})}

no_label = sorted(images.keys() - labels.keys())          # images missing a .txt
no_image = sorted(labels.keys() - images.keys())          # labels missing an image

# Index every file elsewhere, skipping the four folders we already looked at
skip = [(OUT / "images").resolve(), (OUT / "labels").resolve(), Q.resolve()]
index = {}
for root in SEARCH:
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS | {'.txt'}:
            continue
        rp = p.resolve()
        if any(s in rp.parents for s in skip):
            continue
        index.setdefault((p.stem, 'txt' if p.suffix.lower() == '.txt' else 'img'), []).append(p)

def origin(stem):
    """Names look like  background__class__mockup__001_3objects"""
    parts = stem.split("__")
    return parts[:3] if len(parts) >= 4 else ["?", "?", "?"]

def transfer(src, dst):
    if dst.exists():
        return
    (shutil.move if MOVE else shutil.copy2)(src, dst)

rows = []
def handle(stems, missing_kind):
    for s in stems:
        bg, cls, mock = origin(s)
        hits = index.get((s, missing_kind), [])
        if hits:
            src = hits[0]
            dest_dir = Q / ("images" if missing_kind == 'img' else "labels")
            transfer(src, dest_dir / src.name)
            status, where = "found", str(src)
        else:
            status, where = "NOT FOUND", ""
        rows.append([s, f"missing {'image' if missing_kind == 'img' else 'label'}",
                     status, where, bg, cls, mock])
        print(f"{status:9} {s}  <- {where or 'no partner anywhere'}")

handle(no_image, 'img')     # labels whose image is missing -> look for the image
handle(no_label, 'txt')     # images whose label is missing -> look for the .txt

with (Q / "report.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["stem", "problem", "status", "partner_found_at", "background", "class", "mockup"])
    w.writerows(rows)

found = sum(r[2] == "found" for r in rows)
print(f"\n{found} of {len(rows)} partners found. Report: {Q / 'report.csv'}")