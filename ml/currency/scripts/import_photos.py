#!/usr/bin/env python3
"""Copies a folder of REAL phone photos of ONE kind of note into ml/currency/test/<class_folder>/ as real_001.jpg, real_002.jpg ...
(numbering continues after the highest existing number; the originals are copied, never changed or deleted; EXIF is kept).

  python import_photos.py ~/Desktop/currency_photos/new_2_front  sle_2
  python import_photos.py ~/Desktop/currency_photos/hands_paper  no_note

class_folder = sle_1 sle_2 sle_5 sle_10 sle_20 sll_1000 sll_2000 sll_5000 sll_10000 no_note
Only .jpg/.jpeg/.png are imported. HEIC is not supported (set the phone camera to JPEG)."""
import argparse, hashlib, pathlib, re, shutil, sys
from PIL import Image
from common import ROOT, LABELS

FOLDERS = set(LABELS["folder_to_class"]) | {"no_note"}

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("source"); ap.add_argument("class_folder"); a = ap.parse_args(argv)
    if a.class_folder not in FOLDERS: sys.exit(f"unknown class folder '{a.class_folder}'. Use one of: {sorted(FOLDERS)}")
    src = pathlib.Path(a.source).expanduser()
    if not src.is_dir(): sys.exit(f"not a folder: {src}")
    dest = ROOT / "test" / a.class_folder; dest.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dest.iterdir() if re.fullmatch(r"real_(\d+)\.(jpg|jpeg|png)", p.name.lower())]
    nxt = max([int(re.findall(r"\d+", p.name)[0]) for p in existing], default=0) + 1
    have = {hashlib.md5(p.read_bytes()).hexdigest() for p in existing}
    done = skipped = noexif = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            if p.is_file() and not p.name.startswith("."): print(f"skip (not jpg/png): {p.name}"); skipped += 1
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in have: print(f"skip (already imported): {p.name}"); skipped += 1; continue
        try: ex = Image.open(p).getexif()
        except Exception as e: print(f"skip (unreadable): {p.name} {e}"); skipped += 1; continue
        if not (ex.get(271) or ex.get(272)): noexif += 1; print(f"warning: no camera EXIF in {p.name} (screenshot / sent through a chat app / edited?)")
        out = dest / f"real_{nxt:03d}{p.suffix.lower().replace('jpeg', 'jpg')}"
        shutil.copy2(p, out); have.add(h); nxt += 1; done += 1
    print(f"imported {done} photo(s) into test/{a.class_folder}/ (skipped {skipped}, without camera EXIF {noexif}). "
          f"Now in that folder: {len([q for q in dest.iterdir() if q.name.lower().startswith('real_')])} real photos.")

if __name__ == "__main__":
    main()
