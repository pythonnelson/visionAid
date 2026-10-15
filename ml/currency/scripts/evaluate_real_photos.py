#!/usr/bin/env python3
"""Evaluates GENUINE phone photographs with the deployed TFLite model using the app's rules.

  python evaluate_real_photos.py              # real photos only  -> reports/real_photo_evaluation.md
  python evaluate_real_photos.py --list       # only show what is in the test folders and why (provenance), no scoring
  python evaluate_real_photos.py --selftest   # checks the metric arithmetic with made-up numbers (not an evaluation)
  python evaluate_real_photos.py --dump-inputs  # also save the exact 224x224 inputs (processed/real_inputs/) so the SAME bytes
                                                # can be pushed to the phone (see VISIONAID_CURRENCY_RECOGNITION.md)

PUT PHOTOS IN  ml/currency/test/<folder>/real_<anything>.jpg  (folder = TRUE note: sle_1 sle_2 sle_5 sle_10 sle_20 sll_1000
sll_2000 sll_5000 sll_10000, or no_note for photos with no banknote / unsupported notes).

Provenance (decided from the file name, never mixed in the numbers):
  real_*                       genuine phone photo  -> evaluated (after duplicate screening)
  synthetic_* collage_synthetic_*   simulated images made by make_test_samples.py -> NEVER counted as real
  anything else in test/       an ORIGINAL DOWNLOADED SCAN or other file -> NOT counted
Duplicate screening: every real photo is compared (perceptual hashes + correlation, 4 rotations, whole image and centre square)
with all scans, collage and negative photos used for training/testing; possible matches are EXCLUDED and listed for manual review.
Real photos must stay out of training and threshold tuning until this evaluation is documented."""
import argparse, collections, hashlib, json, os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import ImageOps
from common import *

ASSETS = ROOT.parent.parent / "VisionAidObjectDetector" / "android" / "src" / "main" / "assets"
FOLDERS = dict(LABELS["folder_to_class"]); FOLDERS["no_note"] = "NO_NOTE"
NOT_SURE = "NOT_SURE"
COLS = [c for c in CLASS_IDS if c != "NO_NOTE"] + [NOT_SURE]


# ---------- provenance ----------
def provenance(p: pathlib.Path) -> str:
    n = p.name.lower()
    if n.startswith("real_"): return "real"
    if n.startswith("synthetic") or n.startswith("collage_synthetic"): return "synthetic"
    return "scan_or_other"


# ---------- hashing / duplicate screening ----------
def ahash(im, s=8):
    a = np.asarray(im.convert("L").resize((s, s), Image.BILINEAR), dtype=np.float32); return (a > a.mean()).flatten()

def dhash_bits(im, s=8):
    a = np.asarray(im.convert("L").resize((s + 1, s), Image.BILINEAR), dtype=np.int16); return (a[:, 1:] > a[:, :-1]).flatten()

def phash_bits(im, s=8):
    n = 32; a = np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float64)
    k = np.arange(n); dct = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    d = (dct @ a @ dct.T)[:s, :s]; return (d > np.median(d[1:].flatten() if False else d)).flatten()

def signatures(im):
    """hash bundle for the whole image and its centre square, each at 4 rotations"""
    s = min(im.size); sq = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width - s) // 2 + s, (im.height - s) // 2 + s))
    out = []
    for view in (im, sq):
        for rot in (0, 90, 180, 270):
            v = view.rotate(rot, expand=True); g = np.asarray(v.convert("L").resize((64, 32), Image.BILINEAR), dtype=np.float32)
            g = (g - g.mean()) / (g.std() + 1e-6)
            out.append((np.concatenate([dhash_bits(v), phash_bits(v), ahash(v)]), g))
    return out

def reference_images():
    refs = []
    for sub in ("dataset", "test"):
        for p in list_images(ROOT / sub):
            if provenance(p) == "real": continue
            refs.append((str(p.relative_to(ROOT)), p))
    for p in sorted((ROOT / "negatives_source").glob("*.jpg")): refs.append((str(p.relative_to(ROOT)), p))
    return refs

def screen_duplicates(real_paths):
    refs = [(name, signatures(load_rgb(p))[0]) for name, p in reference_images()]   # rotation-0 signature of each reference
    flags = {}
    for rp in real_paths:
        sigs = signatures(ImageOps.exif_transpose(load_rgb(rp)))
        best = (999, 0.0, "")
        for name, (rb, rg) in refs:
            for sb, sg in sigs:
                ham = int((sb[:64] != rb[:64]).sum() + (sb[64:128] != rb[64:128]).sum() + (sb[128:] != rb[128:]).sum()) // 3
                corr = float((sg * rg).mean())
                if ham < best[0] or (ham == best[0] and corr > best[1]): best = (ham, corr, name)
                if corr > best[1] and ham <= best[0] + 2: best = (min(ham, best[0]), corr, name)
        ham, corr, name = best
        flags[rp] = dict(nearest=name, hamming_of_64=ham, correlation=round(corr, 3),
                         flagged=bool(ham <= 10 or corr >= 0.90))
    return flags


# ---------- metrics (pure, unit-checked by --selftest) ----------
def summarize(rows):
    """rows: list of dict(true=CLASS_ID, said=CLASS_ID|NOT_SURE, argmax=CLASS_ID). Returns metrics dict."""
    notes = [r for r in rows if r["true"] != "NO_NOTE"]; neg = [r for r in rows if r["true"] == "NO_NOTE"]
    per = {}
    for c in CLASS_IDS[:-1]:
        rs = [r for r in notes if r["true"] == c]
        if not rs: continue
        ok = sum(r["said"] == c for r in rs); ns = sum(r["said"] == NOT_SURE for r in rs); wr = len(rs) - ok - ns
        ann = ok + wr
        per[c] = dict(n=len(rs), correct=ok, wrong=wr, not_sure=ns, acc_all=ok / len(rs),
                      acc_announced=(ok / ann) if ann else None, coverage=ann / len(rs))
    n = len(notes); ok = sum(r["said"] == r["true"] for r in notes); ns = sum(r["said"] == NOT_SURE for r in notes); wr = n - ok - ns
    tot = dict(n=n, correct=ok, wrong=wr, not_sure=ns, acc_all=ok / n if n else None,
               acc_announced=ok / (ok + wr) if ok + wr else None, coverage=(ok + wr) / n if n else None)
    cm = collections.OrderedDict((t, collections.Counter()) for t in CLASS_IDS[:-1] + ["NO_NOTE"])
    for r in rows: cm[r["true"]][r["said"]] += 1
    fa = [r for r in neg if r["said"] != NOT_SURE]
    return dict(per_class=per, total=tot, confusion=cm, no_note=dict(n=len(neg), false_announcements=len(fa),
                false_detail=collections.Counter(r["said"] for r in fa)))

def selftest():
    mk = lambda t, s: dict(true=t, said=s, argmax=s)
    rows = [mk("NEW_2", "NEW_2"), mk("NEW_2", "OLD_2000"), mk("NEW_2", NOT_SURE), mk("OLD_2000", "OLD_2000"),
            mk("NO_NOTE", NOT_SURE), mk("NO_NOTE", "NEW_5")]
    m = summarize(rows); t = m["total"]; p = m["per_class"]["NEW_2"]
    assert (t["n"], t["correct"], t["wrong"], t["not_sure"]) == (4, 2, 1, 1), t
    assert abs(t["acc_all"] - 0.5) < 1e-9 and abs(t["acc_announced"] - 2 / 3) < 1e-9 and abs(t["coverage"] - 0.75) < 1e-9, t
    assert (p["n"], p["correct"], p["wrong"], p["not_sure"]) == (3, 1, 1, 1) and abs(p["coverage"] - 2 / 3) < 1e-9, p
    assert m["no_note"]["false_announcements"] == 1 and m["no_note"]["false_detail"]["NEW_5"] == 1
    assert m["confusion"]["NEW_2"]["OLD_2000"] == 1
    assert provenance(pathlib.Path("real_001.jpg")) == "real" and provenance(pathlib.Path("synthetic_00.png")) == "synthetic" \
        and provenance(pathlib.Path("collage_synthetic_01.png")) == "synthetic" and provenance(pathlib.Path("sle_1.jpg")) == "scan_or_other"
    print("selftest OK (metric arithmetic + provenance rules)")


# ---------- keep every run, so a later session can simply carry on ----------
def archive_run(text, model_path, tau, n_found, n_scored, n_excluded, t, nn):
    """Copies the report to reports/real_photo_runs/<date>_tau<tau>_<modelsha>.md and appends one row to reports/real_test_log.md."""
    import datetime
    sha = hashlib.sha256(pathlib.Path(model_path).read_bytes()).hexdigest()[:8]
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    runs = ROOT / "reports" / "real_photo_runs"; runs.mkdir(exist_ok=True)
    name = f"{stamp}_tau{tau}_{sha}.md"; (runs / name).write_text(text)
    log = ROOT / "reports" / "real_test_log.md"
    if not log.exists():
        log.write_text("# Real-photo test log\n\n| date | model sha | tau | real photos found | scored | excluded (dup?) | correct | WRONG denomination | not sure | no_note falsely announced | report |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    with log.open("a") as f:
        f.write(f"| {stamp} | {sha} | {tau} | {n_found} | {n_scored} | {n_excluded} | {t['correct']}/{t['n']} | {t['wrong']} | {t['not_sure']} | {nn['false_announcements']}/{nn['n']} | real_photo_runs/{name} |\n")
    print(f"archived: reports/real_photo_runs/{name}  and a row was added to reports/real_test_log.md")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-inputs", action="store_true"); ap.add_argument("--include-flagged", action="store_true",
                    help="score photos flagged as possible duplicates too (NOT recommended)")
    ap.add_argument("--model", default=str(ASSETS / "currency_classifier.tflite")); a = ap.parse_args()
    if a.selftest: return selftest()

    files = [(f, p) for f in FOLDERS for p in list_images(ROOT / "test" / f)]
    groups = collections.defaultdict(list)
    for f, p in files: groups[provenance(p)].append((f, p))
    print(f"test folders: real photos={len(groups['real'])}  synthetic (never counted as real)={len(groups['synthetic'])}  "
          f"scans/other (not counted)={len(groups['scan_or_other'])}")
    for f, p in groups["scan_or_other"]: print(f"  not counted (original scan / other): test/{f}/{p.name}")
    if a.list: return
    real = groups["real"]
    if not real:
        print("\nNO REAL PHOTOS FOUND -> nothing to evaluate. Add ml/currency/test/<folder>/real_001.jpg ... (see header). "
              "No real-world accuracy can be reported yet."); sys.exit(2)

    tau = json.loads((ROOT / "reports" / "thresholds.json").read_text())["accept_threshold"]
    assert tau == json.loads((ASSETS / "currency_thresholds.json").read_text())["accept_threshold"], "threshold files differ"
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=a.model); it.allocate_tensors(); i, o = it.get_input_details()[0], it.get_output_details()[0]

    print("screening for duplicates / near-duplicates of training material ...")
    flags = screen_duplicates([p for f, p in real])
    md5s = collections.defaultdict(list)
    for f, p in real: md5s[hashlib.md5(p.read_bytes()).hexdigest()].append(p.name)
    exact_dups = [v for v in md5s.values() if len(v) > 1]
    rows, excluded, noexif, perphoto = [], [], [], []
    dump = ROOT / "processed" / "real_inputs"
    if a.dump_inputs: dump.mkdir(parents=True, exist_ok=True)
    for f, p in real:
        fl = flags[p]
        raw = Image.open(p); exif = raw.getexif()
        if not (exif.get(271) or exif.get(272)): noexif.append(f"{f}/{p.name}")      # Make / Model tags
        if fl["flagged"] and not a.include_flagged:
            excluded.append((f"{f}/{p.name}", fl)); continue
        im = ImageOps.exif_transpose(load_rgb(p))                                       # phone photos carry EXIF rotation
        s = min(im.size); l, t = (im.width - s) // 2, (im.height - s) // 2
        sq = im.crop((l, t, l + s, t + s)).resize((224, 224), Image.BOX)                # centre square, area average (like the app)
        x = np.asarray(sq, dtype=np.float32)
        if a.dump_inputs: sq.save(dump / f"{f}__{p.stem}.png")
        it.set_tensor(i["index"], x[None]); it.invoke(); pr = it.get_tensor(o["index"])[0]
        top = int(pr.argmax()); said = CLASS_IDS[top] if (CLASS_IDS[top] != "NO_NOTE" and pr[top] >= tau) else NOT_SURE
        rows.append(dict(true=FOLDERS[f], said=said, argmax=CLASS_IDS[top], p=float(pr[top]), file=f"{f}/{p.name}"))
        perphoto.append(f"| {f}/{p.name} | {FOLDERS[f]} | {CLASS_IDS[top]} | {pr[top]:.3f} | {said} | {'OK' if said == FOLDERS[f] else ('not sure' if said == NOT_SURE and FOLDERS[f] != 'NO_NOTE' else ('correct reject' if said == NOT_SURE else 'WRONG'))} |")

    L = ["# Real-photo evaluation (generated by scripts/evaluate_real_photos.py)", "",
         f"Model: `{a.model}`; accept threshold tau = {tau}. Only genuine phone photos named `real_*` are scored.", "",
         f"- real photos found: {len(real)}; scored: {len(rows)}; excluded as possible duplicates of training material: {len(excluded)}",
         f"- synthetic images in test folders (NOT counted): {len(groups['synthetic'])}; original scans/other (NOT counted): {len(groups['scan_or_other'])}"]
    if exact_dups: L.append(f"- **exact duplicate real photos (same bytes):** {exact_dups}")
    if noexif: L.append(f"- real photos WITHOUT camera EXIF (Make/Model) - may be screenshots, downloads or edited; please confirm they are phone photos: {noexif}")
    if excluded:
        L += ["", "## Excluded: possible duplicates of training/test source images (manual review needed)", "", "| photo | nearest reference | hash distance (of 64) | correlation |", "|---|---|---|---|"]
        L += [f"| {n} | {d['nearest']} | {d['hamming_of_64']} | {d['correlation']} |" for n, d in excluded]
    if not rows:
        L += ["", "No photo could be scored."]; (ROOT / "reports" / "real_photo_evaluation.md").write_text("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(3)
    m = summarize(rows); t = m["total"]
    pct = lambda v: "-" if v is None else f"{100 * v:.1f} %"
    L += ["", "## Overall (banknote photos only)", "",
          f"n={t['n']}  correct={t['correct']}  **WRONG denomination={t['wrong']}**  not sure={t['not_sure']}",
          f"- accuracy over ALL images (correct / n): {pct(t['acc_all'])}",
          f"- accuracy among ANNOUNCED predictions: {pct(t['acc_announced'])}",
          f"- coverage (share of images where a denomination was announced): {pct(t['coverage'])}", "",
          "Accuracy among announced predictions alone hides wrong answers and silence; read all three numbers together.", "",
          "## Per denomination", "", "| denomination | n | correct | wrong denomination | not sure | accuracy (all) | accuracy (announced) | coverage |", "|---|---|---|---|---|---|---|---|"]
    for c, d in m["per_class"].items():
        flag = " (n<10: unreliable)" if d["n"] < 10 else ""
        L.append(f"| {c} | {d['n']}{flag} | {d['correct']} | {d['wrong']} | {d['not_sure']} | {pct(d['acc_all'])} | {pct(d['acc_announced'])} | {pct(d['coverage'])} |")
    L += ["", "## Confusion matrix (rows = true folder, columns = what the app SAID; NOT_SURE = no announcement)", "",
          "| true \\ said | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for tcls, cnt in m["confusion"].items():
        if sum(cnt.values()): L.append(f"| {tcls} | " + " | ".join(str(cnt.get(c, 0)) for c in COLS) + " |")
    nn = m["no_note"]
    L += ["", "## Non-currency photos (folder no_note)", "",
          f"n={nn['n']}  **falsely announced a note: {nn['false_announcements']}** {dict(nn['false_detail']) if nn['false_announcements'] else ''}", "",
          "## Every photo", "", "| photo | true | model top class | top prob | app said | result |", "|---|---|---|---|---|---|"] + perphoto
    L += ["", "Notes: probabilities above are the model's own scores, NOT demonstrated chances of being right (see safety_audit.md section 4). "
          "Keep these photos out of training/threshold tuning until this report has been reviewed."]
    text = "\n".join(L) + "\n"
    (ROOT / "reports" / "real_photo_evaluation.md").write_text(text); print(text)
    archive_run(text, a.model, tau, len(real), len(rows), len(excluded), t, nn)

if __name__ == "__main__":
    main()
