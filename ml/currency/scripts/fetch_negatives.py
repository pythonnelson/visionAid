#!/usr/bin/env python3
"""Downloads the negative ("not a banknote") images listed in negatives_ids.txt from COCO val2017.
COCO images are Flickr photos under Creative Commons licences (see cocodataset.org/#termsofuse);
they are used ONLY locally for training/testing and are not committed to git."""
import pathlib, urllib.request
ROOT = pathlib.Path(__file__).resolve().parent.parent
out = ROOT / "negatives_source"; out.mkdir(exist_ok=True)
for line in (ROOT / "negatives_ids.txt").read_text().split():
    p = out / f"{line}.jpg"
    if not p.exists():
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{line}.jpg", p)
print("negatives:", len(list(out.glob('*.jpg'))))
