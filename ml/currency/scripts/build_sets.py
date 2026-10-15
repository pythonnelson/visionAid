#!/usr/bin/env python3
"""Builds the train / val / test arrays (uint8 224x224x3 + labels) with a FIXED seed from processed/crops.

Sources (see manifest.json):
  - "single" crops (18 unique note sides)           -> train, val, test-sim   (different seeds + different background pools)
  - "stack"  crops: byte-for-byte the same scan as the single crops -> EXCLUDED (duplicates)
  - "grouped" crops (collage, NEW notes only, different rendering) -> test-collage ONLY, never trained on
  - unsupported old Le 500 / Le 100 notes -> class NO_NOTE (train: 500#1, val: Le 100 sides, test: 500#2)
  - negatives: COCO photos split by image id into train/val/test pools (disjoint)
"""
import json, sys
from common import *
from synth import render, render_negative

SEED = 20261015
N_TRAIN, N_VAL, N_TEST_SIM, N_COLLAGE = 160, 30, 30, 20


def load_pool(files):
    return [load_rgb(f) for f in files]


def main():
    PR = ROOT / "processed"; man = json.loads((PR / "manifest.json").read_text())
    rng_split = np.random.default_rng(SEED)
    neg_files = sorted((ROOT / "negatives_source").glob("*.jpg")); rng_split.shuffle(neg_files)
    pools = {"train": neg_files[:200], "val": neg_files[200:260], "test": neg_files[260:320]}
    bg = {k: load_pool(v) for k, v in pools.items()}
    cls_index = {c: i for i, c in enumerate(CLASS_IDS)}
    singles = [m for m in man if m["provenance"] == "single"]
    collage = [m for m in man if m["provenance"] == "grouped"]

    # unsupported (old Le 500 / Le 100) notes, taken from the mis-filed no_note folder
    nn = ROOT / "dataset" / "no_note"
    unsup = {"train": [load_rgb(nn / "no-note-1.jpg")], "test": [load_rgb(nn / "no-note-2.jpeg")]}
    from prepare_dataset import trim, split_stack
    hundred_front, hundred_back = split_stack(load_rgb(nn / "no-note.png"))
    unsup["val"] = [hundred_front, hundred_back]

    sets = {k: {"x": [], "y": [], "meta": []} for k in ("train", "val", "test_sim", "test_collage", "test_neg", "test_unsup")}

    def add(split, img, y, meta):
        sets[split]["x"].append(np.asarray(img, dtype=np.uint8)); sets[split]["y"].append(y); sets[split]["meta"].append(meta)

    for split, n, seed_off, hard, bgkey in (("train", N_TRAIN, 1, False, "train"), ("val", N_VAL, 2, True, "val"),
                                             ("test_sim", N_TEST_SIM, 3, False, "test")):
        rng = np.random.default_rng(SEED + seed_off)
        for m in singles:
            note = load_rgb(ROOT / m["file"])
            for i in range(n):
                im, vis = render(note, rng, bg[bgkey], hard)
                add(split, im, cls_index[m["cls"]], dict(cls=m["cls"], view=m["view"], src=m["file"], visible=round(vis, 2)))
        # negatives: real photo crops + plain surfaces
        nneg = {"train": 1400, "val": 160, "test_sim": 160}[split]
        for i in range(nneg):
            add(split, render_negative(rng, bg[bgkey], hard), NO_NOTE_INDEX, dict(cls="NO_NOTE", view="-", src="negative"))
        # unsupported notes as NO_NOTE
        usplit = {"train": "train", "val": "val", "test_sim": None}[split]
        if usplit:
            for note in unsup[usplit]:
                for i in range({"train": 90, "val": 25}[usplit]):
                    im, vis = render(note, rng, bg[bgkey], hard)
                    add(split, im, NO_NOTE_INDEX, dict(cls="NO_NOTE", view="unsupported_note", src="old Le 500/100"))
    # test: collage crops (out-of-source), simulated + clean (clean = plain resize to 224 with white padding)
    rng = np.random.default_rng(SEED + 4)
    for m in collage:
        note = load_rgb(ROOT / m["file"]); w, h = note.size
        note = note.crop((0, 0, w, int(h * 0.92)))          # drop the sliver of the neighbouring note
        for i in range(N_COLLAGE):
            im, vis = render(note, rng, bg["test"], False)
            add("test_collage", im, cls_index[m["cls"]], dict(cls=m["cls"], view=m["view"], src=m["file"]))
    rng = np.random.default_rng(SEED + 5)
    for i in range(300):
        add("test_neg", render_negative(rng, bg["test"], False), NO_NOTE_INDEX, dict(cls="NO_NOTE", src="negative"))
    for note in unsup["test"]:
        for i in range(40):
            im, vis = render(note, rng, bg["test"], False)
            add("test_unsup", im, NO_NOTE_INDEX, dict(cls="NO_NOTE", src="old Le 500 specimen #2"))

    out = PR / "arrays"; out.mkdir(exist_ok=True)
    for k, v in sets.items():
        np.savez_compressed(out / f"{k}.npz", x=np.stack(v["x"]), y=np.array(v["y"], dtype=np.int64))
        (out / f"{k}_meta.json").write_text(json.dumps(v["meta"]))
        print(k, len(v["y"]), "samples; per-class:", np.bincount(v["y"], minlength=len(CLASS_IDS)).tolist())
    (PR / "split_info.json").write_text(json.dumps(dict(
        seed=SEED, neg_pool_sizes={k: len(v) for k, v in pools.items()},
        neg_pool_ids={k: [p.stem for p in v] for k, v in pools.items()},
        sizes={k: len(v["y"]) for k, v in sets.items()}), indent=1))
    # save a few preview images
    prev = PR / "previews"; prev.mkdir(exist_ok=True)
    for k in ("train", "val", "test_sim", "test_collage"):
        x = np.stack(sets[k]["x"]); idx = np.random.default_rng(0).choice(len(x), 24, replace=False)
        sheet = Image.new("RGB", (6 * 224, 4 * 224))
        for n, i in enumerate(idx): sheet.paste(Image.fromarray(x[i]), ((n % 6) * 224, (n // 6) * 224))
        sheet.resize((6 * 112, 4 * 112)).save(prev / f"{k}.jpg", quality=85)

if __name__ == "__main__":
    main()
