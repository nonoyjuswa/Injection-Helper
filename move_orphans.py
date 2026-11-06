from pathlib import Path
import shutil, sys

OUT = Path(r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Corrected")
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
MOVE = "--move" in sys.argv          # run with --move to quarantine orphans

images = {p.stem: p for p in (OUT / "images").iterdir() if p.suffix.lower() in IMG_EXTS}
labels = {p.stem: p for p in (OUT / "labels").iterdir() if p.suffix.lower() == ".txt"}

no_label = sorted(images.keys() - labels.keys())
no_image = sorted(labels.keys() - images.keys())
empty    = sorted(s for s, p in labels.items() if p.stat().st_size <= 1)   # label file with no boxes

print(f"{len(images)} images, {len(labels)} labels")
print(f"\nImages without a label ({len(no_label)}):")
for s in no_label: print("  ", images[s].name)
print(f"\nLabels without an image ({len(no_image)}):")
for s in no_image: print("  ", labels[s].name)
print(f"\nEmpty label files ({len(empty)}):")
for s in empty: print("  ", labels[s].name)

if MOVE:
    q = OUT / "_orphans"
    (q / "images").mkdir(parents=True, exist_ok=True)
    (q / "labels").mkdir(parents=True, exist_ok=True)
    for s in no_label: shutil.move(images[s], q / "images" / images[s].name)
    for s in no_image: shutil.move(labels[s], q / "labels" / labels[s].name)
    print(f"\nMoved orphans to {q}")