#!/usr/bin/env python3
"""Stage D: exports runs/currency_classifier.keras to TFLite (float32, no quantisation by default),
verifies it against the Keras model on the validation set, and copies model + labels into the Android assets.

  python export_tflite.py [--quantize dynamic]
"""
import argparse, json, os, shutil
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
from common import *

ASSETS = ROOT.parent.parent / "VisionAidObjectDetector" / "android" / "src" / "main" / "assets"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quantize", choices=["none", "dynamic"], default="none"); a = ap.parse_args()
    m = tf.keras.models.load_model(ROOT / "runs" / "currency_classifier.keras")
    # Keras-3 models convert reliably through a SavedModel export
    sm = ROOT / "runs" / "saved_model"; m.export(str(sm), format="tf_saved_model")
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    if a.quantize == "dynamic": conv.optimizations = [tf.lite.Optimize.DEFAULT]
    blob = conv.convert()
    out = ROOT / "runs" / "currency_classifier.tflite"; out.write_bytes(blob)
    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    info = dict(input_shape=i["shape"].tolist(), input_dtype=np.dtype(i["dtype"]).name, input_quant=[float(v) for v in i["quantization"]],
                output_shape=o["shape"].tolist(), output_dtype=np.dtype(o["dtype"]).name, size_bytes=len(blob), quantize=a.quantize,
                input_range="0..255 RGB (rescaling is inside the model)", class_order=CLASS_IDS)
    d = np.load(ROOT / "processed" / "arrays" / "val.npz"); x = d["x"][:300].astype(np.float32)
    ref = m.predict(x, verbose=0); got = []
    for img in x:
        it.set_tensor(i["index"], img[None]); it.invoke(); got.append(it.get_tensor(o["index"])[0])
    got = np.array(got); info["max_abs_diff_vs_keras"] = float(np.abs(ref - got).max()); info["argmax_agreement"] = float((ref.argmax(1) == got.argmax(1)).mean())
    print(json.dumps(info, indent=1)); (ROOT / "reports" / "model_export.json").write_text(json.dumps(info, indent=1))
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, ASSETS / "currency_classifier.tflite"); shutil.copy(ROOT / "labels.json", ASSETS / "currency_labels.json")
    th = ROOT / "reports" / "thresholds.json"
    if th.exists(): shutil.copy(th, ASSETS / "currency_thresholds.json")
    print("copied to", ASSETS)

if __name__ == "__main__":
    main()
