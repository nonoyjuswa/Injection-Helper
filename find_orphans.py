from pathlib import Path
from collections import Counter, defaultdict

OUT = Path(r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Corrected")
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

for sub in ("images", "labels"):
    items = list(( OUT / sub).iterdir())
    print(f"\n=== {sub}: {len(items)} items in folder ===")

    # 1) What kinds of things are in here?
    print("By extension:", dict(Counter(p.suffix.lower() or "(folder/none)" for p in items)))

    # 2) Anything that isn't a normal image/label file
    expected = IMG_EXTS if sub == "images" else {'.txt'}
    for p in items:
        if not p.is_file() or p.suffix.lower() not in expected:
            print("  unexpected:", p.name, "(folder)" if p.is_dir() else "")

    # 3) Same stem, different extension (the one my script would collapse)
    by_stem = defaultdict(list)
    for p in items:
        if p.is_file():
            by_stem[p.stem].append(p.name)
    for stem, names in by_stem.items():
        if len(names) > 1:
            print("  DUPLICATE STEM:", names)