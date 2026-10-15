#!/usr/bin/env python3
"""Fills ml/currency/test/<folder>/ with example held-out images so the folders are not empty.
These are SYNTHETIC (simulated phone captures of the scans, plus collage-source crops) - NOT real phone
photos. Real photos you take yourself belong in the same folders and are evaluated by evaluate_real_photos.py.
Files are named synthetic_*.png / collage_synthetic_*.png so they can always be told apart."""
import json
from common import *

def main():
    d = ROOT / "processed" / "arrays"; folders = {v: k for k, v in LABELS["folder_to_class"].items()}; folders["NO_NOTE"] = "no_note"
    for name, prefix, per in (("test_sim", "synthetic", 8), ("test_collage", "collage_synthetic", 4), ("test_neg", "synthetic_negative", 10)):
        z = np.load(d / f"{name}.npz"); x, y = z["x"], z["y"]
        for c in sorted(set(y.tolist())):
            idx = np.where(y == c)[0]; pick = idx[np.linspace(0, len(idx) - 1, per).astype(int)]
            out = ROOT / "test" / folders[CLASS_IDS[c]]; out.mkdir(parents=True, exist_ok=True)
            for n, i in enumerate(pick): Image.fromarray(x[i]).save(out / f"{prefix}_{n:02d}.png")
    for f in sorted((ROOT / "test").iterdir()):
        if f.is_dir(): print(f.name, len([p for p in f.iterdir() if p.suffix.lower() in IMG_EXT]))

main()
