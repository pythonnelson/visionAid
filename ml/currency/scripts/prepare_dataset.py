#!/usr/bin/env python3
"""Stage B: builds processed/crops (one image per note side) from ml/currency/dataset WITHOUT touching originals.

  * front / back files           -> used as they are (trimmed of plain borders)       provenance "single"
  * stack pictures (front on top, back below, one specimen) -> split in two halves      provenance "stack"
  * all_currencies.jpeg (5 notes x 2 sides)                -> 10 crops                  provenance "grouped"
Every crop is written to processed/crops/<CLASS>/<name>.png and recorded in processed/manifest.json with
its source file, crop box and a *group id*. Crops that are near-duplicates of each other (same scan at a
different size/crop) get the SAME group id, and a group is never split across train/validation/test.
"""
import json, pathlib, sys
from common import *

OUT = ROOT / "processed"
Image.MAX_IMAGE_PIXELS = None


def trim(im: Image.Image, tol=28) -> Image.Image:
    """Cut away plain borders (white, black, grey backgrounds, drop shadows are kept small)."""
    a = np.asarray(im, dtype=np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.abs(a - bg).sum(axis=2) > tol
    rows, cols = np.where(diff.any(axis=1))[0], np.where(diff.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return im
    # ignore thin lines: require >=3% of pixels along row/col
    rr = np.where(diff.mean(axis=1) > 0.03)[0]; cc = np.where(diff.mean(axis=0) > 0.03)[0]
    if len(rr) and len(cc):
        rows, cols = rr, cc
    box = (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)
    return im.crop(box)


def split_stack(im: Image.Image):
    h = im.height
    top, bottom = im.crop((0, 0, im.width, h // 2)), im.crop((0, h // 2, im.width, h))
    return trim(top), trim(bottom)


def main():
    OUT.mkdir(exist_ok=True)
    manifest = []
    pairs = {"sle_1": "NEW_1", "sle_2": "NEW_2", "sle_5": "NEW_5", "sle_10": "NEW_10", "sle_20": "NEW_20",
             "sll_1000": "OLD_1000", "sll_2000": "OLD_2000", "sll_5000": "OLD_5000", "sll_10000": "OLD_10000"}
    def add(cls, name, im, src, view, prov, box=None):
        d = OUT / "crops" / cls; d.mkdir(parents=True, exist_ok=True)
        f = d / f"{name}.png"; im.save(f)
        manifest.append(dict(cls=cls, file=str(f.relative_to(ROOT)), source=src, view=view, provenance=prov,
                             crop_box=box, w=im.width, h=im.height, dhash=dhash(im)))
    for folder, cls in pairs.items():
        for p in list_images(ROOT / "dataset" / folder):
            v = view_of(p); im = load_rgb(p); src = str(p.relative_to(ROOT))
            if v in ("front", "back"):
                add(cls, f"{p.stem}", trim(im), src, v, "single")
            elif v == "stack":
                t, b = split_stack(im)
                add(cls, f"{p.stem}_front", t, src, "front", "stack"); add(cls, f"{p.stem}_back", b, src, "back", "stack")
    # grouped picture: 5 rows x 2 columns, order new 1,2,5,10,20; left = front, right = back
    g = ROOT / "dataset" / "sle_all_currencies" / "all_currencies.jpeg"
    if g.exists():
        im = load_rgb(g); gi = trim(im); W, H = gi.size
        for r, cls in enumerate(["NEW_1", "NEW_2", "NEW_5", "NEW_10", "NEW_20"]):
            for c, v in enumerate(["front", "back"]):
                box = (c * W // 2, r * H // 5, (c + 1) * W // 2, (r + 1) * H // 5)
                add(cls, f"all_currencies_r{r}_{v}", trim(gi.crop(box)), str(g.relative_to(ROOT)), v, "grouped", box)
    # --- group ids: near-duplicate crops of the same class+view share a group
    parent = list(range(len(manifest)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, a in enumerate(manifest):
        for j in range(i + 1, len(manifest)):
            b = manifest[j]
            if a["cls"] == b["cls"] and a["view"] == b["view"] and hamming(a["dhash"], b["dhash"]) <= 10:
                parent[find(j)] = find(i)
    for i, m in enumerate(manifest): m["group"] = find(i)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for m in manifest: print(m["cls"], m["view"], m["provenance"], f'{m["w"]}x{m["h"]}', "group", m["group"], m["file"].split("/")[-1])

if __name__ == "__main__":
    main()
