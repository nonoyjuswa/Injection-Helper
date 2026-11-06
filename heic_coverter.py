#!/usr/bin/env python3
"""Convert every non-JPG image in a folder to JPG, then delete the originals.

Handles: .heic .heif .png .webp .bmp .tif .tiff
Safety:  an original is deleted ONLY after its JPG was written and reopened
         successfully with the same pixel size. Existing JPGs are never overwritten.

Usage:
    py heic_coverter.py                       # uses FOLDER below
    py heic_coverter.py "C:\\path\\to\\folder"  # another folder
    py heic_coverter.py --keep                # convert but keep the originals
    py heic_coverter.py --recursive           # include sub-folders

Dependencies:  pip install pillow pillow-heif
Optional:      pip install send2trash   (deleted originals go to the Recycle Bin)
"""
import argparse
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

FOLDER = r"C:\Users\Juswa\Documents\Thesis\DATASETS\REAL Images\malon"
QUALITY = 95
CONVERT_EXTS = {'.heic', '.heif', '.png', '.webp', '.bmp', '.tif', '.tiff'}

try:
    from send2trash import send2trash
except ImportError:
    send2trash = None


def remove(path: Path):
    if send2trash:
        send2trash(str(path))
    else:
        path.unlink()


def convert(src: Path, dest: Path):
    with Image.open(src) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            rgba = im.convert('RGBA')                      # flatten transparency onto white
            flat = Image.new('RGB', rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.getchannel('A'))
            im = flat
        else:
            im = im.convert('RGB')
        size = im.size
        im.save(dest, 'JPEG', quality=QUALITY, optimize=True, subsampling=0)
    with Image.open(dest) as chk:                          # verify before touching the original
        chk.load()
        if chk.size != size:
            raise ValueError('size mismatch after saving')


def main():
    ap = argparse.ArgumentParser(description='Convert non-JPG images to JPG and delete the originals.')
    ap.add_argument('folder', nargs='?', default=FOLDER)
    ap.add_argument('--keep', action='store_true', help='keep the original files')
    ap.add_argument('--recursive', action='store_true', help='include sub-folders')
    args = ap.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        raise SystemExit(f'Folder not found: {root}')

    it = root.rglob('*') if args.recursive else root.iterdir()
    files = sorted(f for f in it if f.is_file() and f.suffix.lower() in CONVERT_EXTS)
    if not files:
        print('Nothing to convert.')
        return

    converted = deleted = skipped = failed = 0
    for i, src in enumerate(files, 1):
        dest = src.with_suffix('.jpg')
        if dest.exists():                                   # never overwrite an existing JPG
            dest = src.with_name(f'{src.stem}_{src.suffix[1:].lower()}.jpg')
            if dest.exists():
                print(f'[{i}/{len(files)}] SKIP   {src.name} ({dest.name} already exists)')
                skipped += 1
                continue
        try:
            convert(src, dest)
            converted += 1
            if not args.keep:
                remove(src)
                deleted += 1
            print(f'[{i}/{len(files)}] OK     {src.name} -> {dest.name}')
        except Exception as e:
            failed += 1
            print(f'[{i}/{len(files)}] FAILED {src.name}: {e}')
            if dest.exists():
                try:
                    dest.unlink()                           # remove a half-written file
                except OSError:
                    pass

    where = 'Recycle Bin' if send2trash else 'permanently'
    print(f'\nConverted: {converted}   Deleted originals: {deleted} ({where})   '
          f'Skipped: {skipped}   Failed: {failed}')


if __name__ == '__main__':
    main()