from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
register_heif_opener()

for f in Path(r"C:\Users\Juswa\Documents\Thesis\DATASETS\REAL Images\malon").glob("*.heic"):
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    im.save(f.with_suffix(".jpg"), quality=95)