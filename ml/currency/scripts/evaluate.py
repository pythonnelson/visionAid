#!/usr/bin/env python3
"""Stage C/D: evaluation. Works on the .keras model OR the exported .tflite (--tflite path).
Thresholds are chosen on the VALIDATION set only (--calibrate), then applied unchanged to the test sets.

  python evaluate.py --keras runs/currency_classifier.keras --calibrate
  python evaluate.py --tflite ../../VisionAidObjectDetector/android/src/main/assets/currency_classifier.tflite
"""
import argparse, json, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
from common import *

ARR = ROOT / "processed" / "arrays"


def load(name):
    d = np.load(ARR / f"{name}.npz"); return d["x"], d["y"], json.loads((ARR / f"{name}_meta.json").read_text())


def predictor(args):
    if args.tflite:
        it = tf.lite.Interpreter(model_path=args.tflite); it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        def f(x):
            out = []
            for img in x:
                it.set_tensor(i["index"], img[None].astype(np.float32)); it.invoke(); out.append(it.get_tensor(o["index"])[0])
            return np.array(out)
        return f
    m = tf.keras.models.load_model(args.keras)
    return lambda x: m.predict(x.astype(np.float32), batch_size=64, verbose=0)


def decide(p, tau):
    """App rule: accept the top class only if it is a note class and its probability >= tau; else 'unsure'."""
    top = p.argmax(1); conf = p.max(1)
    accept = (top != NO_NOTE_INDEX) & (conf >= tau)
    return top, conf, accept


def calibrate(pv, yv, max_false_note=0.0, max_wrong_denom=0.0, margin=0.005):
    """Smallest tau (on VALIDATION data only) such that
         - NO_NOTE / unsupported-note samples are falsely announced as a note ZERO times  and
         - real notes are announced with the WRONG denomination ZERO times.
    A fixed safety MARGIN (+0.005) is added on top because the first zero-error value is an edge value. The margin
    was introduced after the audit saw one confident NEW_2 -> OLD_2000 announcement on the collage test set, so the
    collage set is no longer a clean test for this choice (disclosed in the documentation).
    (v1 used <= 1 % / <= 0.5 %, i.e. the very edge of the criterion with no safety margin; tightened after the
     safety audit found a confident NEW_2 -> OLD_2000 announcement just above that tau. See safety_audit.md.)
    Safety first: if no tau up to 1.0 satisfies both, 1.0 is returned (= never announce) and the report says so."""
    neg, pos = yv == NO_NOTE_INDEX, yv != NO_NOTE_INDEX
    for tau in np.round(np.arange(0.50, 1.0001, 0.005), 3):
        top, conf, acc = decide(pv, tau)
        false_note = (acc & neg).sum() / max(neg.sum(), 1)
        wrong = (acc & pos & (top != yv)).sum() / max(pos.sum(), 1)
        if false_note <= max_false_note and wrong <= max_wrong_denom:
            return float(min(1.0, round(tau + margin, 3)))   # +margin: the first zero-error value sits on the edge of the validation data
    return 1.0


def report(name, p, y, tau, lines):
    top, conf, acc = decide(p, tau)
    n = len(y); argmax_acc = (p.argmax(1) == y).mean()
    is_note = y != NO_NOTE_INDEX
    correct_ann = (acc & (top == y)).sum(); wrong_ann = (acc & (top != y)).sum()
    lines.append(f"### {name} (n={n})")
    lines.append(f"- argmax accuracy: {argmax_acc:.3f}")
    if is_note.any():
        lines.append(f"- note samples n={int(is_note.sum())}: announced correctly {correct_ann / is_note.sum():.3f}, "
                     f"announced WRONG denomination {int((acc & is_note & (top != y)).sum()) / is_note.sum():.3f}, "
                     f"said 'not sure' {int((~acc & is_note).sum()) / is_note.sum():.3f}")
    if (~is_note).any():
        lines.append(f"- NO_NOTE samples n={int((~is_note).sum())}: falsely announced a note {int((acc & ~is_note).sum()) / (~is_note).sum():.3f}")
    return top, conf, acc


def confusion(p, y, lines, title):
    k = len(CLASS_IDS); cm = np.zeros((k, k), int)
    for t, q in zip(y, p.argmax(1)): cm[t, q] += 1
    lines.append(f"#### {title}: confusion matrix (rows = true, cols = predicted, argmax)")
    lines.append("| true \\ pred | " + " | ".join(CLASS_IDS) + " | n | precision | recall | F1 |")
    lines.append("|" + "---|" * (k + 5))
    for i, c in enumerate(CLASS_IDS):
        tp = cm[i, i]; prec = tp / max(cm[:, i].sum(), 1); rec = tp / max(cm[i].sum(), 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        flag = " (n<30: unreliable)" if cm[i].sum() and cm[i].sum() < 30 else ""
        lines.append(f"| {c} | " + " | ".join(str(v) for v in cm[i]) + f" | {cm[i].sum()}{flag} | {prec:.2f} | {rec:.2f} | {f1:.2f} |")
    off = [(CLASS_IDS[i], CLASS_IDS[j], int(cm[i, j])) for i in range(k) for j in range(k) if i != j and cm[i, j]]
    lines.append("Confusions: " + (", ".join(f"{a}->{b} x{n}" for a, b, n in sorted(off, key=lambda t: -t[2])[:12]) or "none"))
    return cm


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--keras"); ap.add_argument("--tflite"); ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--tau", type=float); ap.add_argument("--out", default="reports/evaluation.md")
    a = ap.parse_args()
    f = predictor(a)
    cfg = ROOT / "reports" / "thresholds.json"
    xv, yv, _ = load("val"); pv = f(xv)
    if a.calibrate or not cfg.exists():
        tau = calibrate(pv, yv); cfg.write_text(json.dumps(dict(accept_threshold=tau, chosen_on="validation (hard augmentation) only",
                                                                  rule="validation: zero false notes on NO_NOTE and zero wrong denominations (v2) + safety margin 0.005 (v3 -> tau 0.98); v1 was <=1% / <=0.5% -> tau 0.965"), indent=1))
    tau = a.tau or json.loads(cfg.read_text())["accept_threshold"]
    L = [f"# Currency classifier evaluation", "", f"Model: `{a.tflite or a.keras}`", f"Accept threshold tau = {tau} (chosen on validation only)", ""]
    report("VALIDATION (hard simulated captures; used to choose tau)", pv, yv, tau, L)
    allp, ally, meta_all = [], [], []
    for name, title in (("test_sim", "TEST-SIM: simulated phone captures, held-out backgrounds/seeds (same scans as training!)"),
                        ("test_collage", "TEST-COLLAGE: NEW notes from a different image source (collage), simulated capture, never trained on"),
                        ("test_neg", "TEST-NEGATIVE: held-out real photos/surfaces without notes"),
                        ("test_unsup", "TEST-UNSUPPORTED: old Le 500 (unsupported note) specimen not used in training")):
        x, y, meta = load(name); p = f(x)
        top, conf, acc = report(title, p, y, tau, L)
        if name == "test_sim":
            cm = confusion(p, y, L, "TEST-SIM")
            for grp, key in (("front", lambda m: m["view"] == "front"), ("back", lambda m: m["view"] == "back"),
                             ("NEW series", lambda m: m["cls"].startswith("NEW")), ("OLD series", lambda m: m["cls"].startswith("OLD"))):
                idx = np.array([key(m) for m in meta]); L.append(f"- {grp}: argmax accuracy {(p[idx].argmax(1) == y[idx]).mean():.3f} (n={int(idx.sum())})")
        if name == "test_collage":
            confusion(p, y, L, "TEST-COLLAGE")
            srcs = {}
            for m in meta: srcs.setdefault(m["cls"], set()).add(m["src"] + m["view"])
            L.append("- DISTINCT source images per class (the n above are simulated variants of these): " + ", ".join(f"{k}={len(v)}" for k, v in srcs.items()) + " -> per-class results are NOT statistically reliable")
        L.append("")
    out = ROOT / a.out; out.write_text("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
