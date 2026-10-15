"""Shared helpers for the VisionAid currency pipeline (training side only, not used by the app)."""
import hashlib, json, pathlib
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent          # ml/currency
LABELS = json.loads((ROOT / "labels.json").read_text())
CLASS_IDS = [c["id"] for c in LABELS["classes"]]
NO_NOTE_INDEX = CLASS_IDS.index("NO_NOTE")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
INPUT_SIZE = 224


def md5(path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def load_rgb(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def dhash(img: Image.Image, size=8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def list_images(folder):
    return sorted(p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def view_of(path) -> str:
    """front / back / stack (front+back of one note in one picture) / grouped (many notes)."""
    n = pathlib.Path(path).stem.lower()
    if "all_currencies" in n:
        return "grouped"
    if n.endswith("_front"):
        return "front"
    if n.endswith("_back"):
        return "back"
    return "stack"
