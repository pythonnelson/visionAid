#!/usr/bin/env python3
"""Stage C: transfer-learning classifier (MobileNetV3-Small, ImageNet weights) on processed/arrays.

Model input: float32 [1,224,224,3], RGB, values 0..255 (the model's own first layer rescales),
Model output: float32 [1,10] softmax in labels.json order.
Usage: python train.py [--epochs1 6 --epochs2 12]
"""
import argparse, json, os, random
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
from common import *

SEED = 20261015


def load(name):
    d = np.load(ROOT / "processed" / "arrays" / f"{name}.npz"); return d["x"], d["y"]


def build():
    base = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights="imagenet",
                                                  include_preprocessing=True, pooling="avg")
    inp = tf.keras.Input((224, 224, 3), name="image_rgb_0_255")
    x = base(inp, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(len(CLASS_IDS), activation="softmax", name="probabilities")(x)
    return tf.keras.Model(inp, out), base


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs1", type=int, default=6); ap.add_argument("--epochs2", type=int, default=12)
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    xt, yt = load("train"); xv, yv = load("val")
    model, base = build()
    counts = np.bincount(yt, minlength=len(CLASS_IDS)); w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    cw = {i: float(w[i]) for i in range(len(w))}
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    swt = np.array([cw[int(c)] for c in yt], dtype=np.float32)       # per-sample class weights (balanced)
    dtr = (tf.data.Dataset.from_tensor_slices((xt.astype(np.float32), yt, swt)).shuffle(len(xt), seed=SEED)
           .batch(32).prefetch(2))
    dva = tf.data.Dataset.from_tensor_slices((xv.astype(np.float32), yv)).batch(64)
    base.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(dtr, validation_data=dva, epochs=a.epochs1, callbacks=cb, verbose=2)
    base.trainable = True
    for l in base.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization): l.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(dtr, validation_data=dva, epochs=a.epochs2, callbacks=cb, verbose=2)
    out = ROOT / "runs"; out.mkdir(exist_ok=True)
    model.save(out / "currency_classifier.keras")
    (out / "training_config.json").write_text(json.dumps(dict(
        seed=SEED, backbone="MobileNetV3Small imagenet", input="224x224x3 float32 RGB 0..255",
        classes=CLASS_IDS, epochs1=a.epochs1, epochs2=a.epochs2, train_samples=int(len(yt)), val_samples=int(len(yv)),
        class_weights=cw, final_val_loss=float(hist.history["val_loss"][-1]),
        final_val_acc=float(hist.history["val_accuracy"][-1]), tf_version=tf.__version__), indent=1))
    print("saved", out)

if __name__ == "__main__":
    main()
