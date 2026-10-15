#!/usr/bin/env python3
"""
Reference (desktop) run of the SAME efficientdet_lite0.tflite that the Android app uses.

Purpose: check the model + label mapping on known images, outside the phone/camera pipeline,
using Google's own TensorFlow Lite Python interpreter.

Usage:
  python reference_detect.py [--mode stretch|letterbox|both] [--top N] image1.jpg image2.png ...

The image is resized to the model input size and the raw RGB bytes (0..255) are fed
directly (no normalisation) - this is the contract found in the model metadata.
Output order used (verified with the model's TFLite_Detection_PostProcess op):
  0 boxes [1,25,4] (ymin,xmin,ymax,xmax), 1 classes, 2 scores, 3 count.
"""
import argparse, os, pathlib
import numpy as np
from PIL import Image
import tensorflow as tf

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE.parent / "android/src/main/assets"
labels = [l.strip() for l in (ASSETS / "coco_labels.txt").read_text().splitlines() if l.strip()]

# Set VISIONAID_MODEL=/path/to/other.tflite to evaluate a different (compatible) model.
interp = tf.lite.Interpreter(model_path=os.environ.get("VISIONAID_MODEL", str(ASSETS / "efficientdet_lite0.tflite")))
interp.allocate_tensors()
inp = interp.get_input_details()[0]
outs = interp.get_output_details()
H, W = inp["shape"][1], inp["shape"][2]


def prep(img, mode):
    if mode == "stretch":
        return np.asarray(img.convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.uint8)
    # letterbox: keep aspect ratio, pad with black (zero)
    img = img.convert("RGB")
    s = min(W / img.width, H / img.height)
    nw, nh = round(img.width * s), round(img.height * s)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((W - nw) // 2, (H - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def run(arr):
    interp.set_tensor(inp["index"], arr[None])
    interp.invoke()
    boxes, classes, scores, count = [interp.get_tensor(o["index"]) for o in outs]
    return boxes[0], classes[0], scores[0], float(count[0])


def show(name, mode, arr, top):
    boxes, classes, scores, count = run(arr)
    print(f"\n== {name} [{mode}] count={count}")
    for i in np.argsort(-scores)[:top]:
        c = int(round(classes[i]))
        print(f"  {labels[c] if 0 <= c < len(labels) else '?':14s} class={c:2d} score={scores[i]:.3f} "
              f"box(yxyx)={np.round(boxes[i], 3).tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--mode", default="both", choices=["stretch", "letterbox", "both", "exact"])
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    print("input:", inp["shape"], inp["dtype"].__name__, inp["quantization"])
    for o in outs:
        print("output:", o["name"], o["shape"].tolist(), o["dtype"].__name__)
    for p in a.images:
        img = Image.open(p)
        if a.mode == "exact":   # feed a 320x320 PNG byte-for-byte (e.g. a phone snapshot)
            show(pathlib.Path(p).name, "exact", np.asarray(img.convert("RGB"), dtype=np.uint8), a.top)
            continue
        for m in (["stretch", "letterbox"] if a.mode == "both" else [a.mode]):
            show(pathlib.Path(p).name, m, prep(img, m), a.top)
