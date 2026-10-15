"""Turns a flat banknote scan into realistic 'phone camera' training/test images.

Final image = 224x224 centre-square view of a scene, exactly what the app will feed the model.
Transformations (all deliberately MODERATE so the denomination stays readable):
  background (real photo crop or plain/gradient surface), size 60-120% of the view, tilt, 90-degree
  turns, perspective, soft shadow, optional finger-like occluder (<=12% of the note), brightness / gamma /
  colour cast / uneven lighting, defocus or motion blur, sensor noise, JPEG compression.
"""
import io, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

W = 448          # working resolution, downscaled to 224 at the end


def _rand_bg(rng, bg_pool):
    r = rng.random()
    if bg_pool and r < 0.65:
        im = bg_pool[rng.integers(len(bg_pool))]
        s = rng.uniform(0.35, 1.0); side = int(min(im.size) * s)
        x = rng.integers(0, im.width - side + 1); y = rng.integers(0, im.height - side + 1)
        return im.crop((x, y, x + side, y + side)).resize((W, W), Image.BILINEAR)
    base = rng.integers(20, 235, 3).astype(np.float32)
    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32) / W
    ang = rng.uniform(0, 2 * math.pi); grad = (np.cos(ang) * xx + np.sin(ang) * yy)[..., None]
    img = base[None, None] * (0.75 + 0.5 * grad)
    if r > 0.88:   # wood / cloth like stripes
        img = img * (0.9 + 0.1 * np.sin(yy[..., None] * rng.uniform(30, 120)))
    img += rng.normal(0, 4, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _perspective_coeffs(src, dst):
    A = []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(np.array(A, dtype=np.float64), b).tolist()


def _place_note(note: Image.Image, rng, hard):
    w, h = note.size
    long_side = rng.uniform(0.6, 1.3 if hard else 1.2) * W
    scale = long_side / max(w, h)
    note = note.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BICUBIC).convert("RGBA")
    # orientation: mostly landscape, sometimes held vertically, rarely upside down
    r = rng.random()
    angle = rng.uniform(-12, 12) + (0 if r < 0.55 else 90 if r < 0.80 else 270 if r < 0.93 else 180)
    # perspective jitter on the note before rotation
    j = (0.12 if hard else 0.07)
    nw, nh = note.size
    corners = [(0, 0), (nw, 0), (nw, nh), (0, nh)]
    moved = [(x + rng.uniform(-j, j) * nw, y + rng.uniform(-j, j) * nh) for x, y in corners]
    pad = int(0.2 * max(nw, nh)); canvas = Image.new("RGBA", (nw + 2 * pad, nh + 2 * pad), (0, 0, 0, 0))
    canvas.paste(note, (pad, pad))
    dst = [(x + pad, y + pad) for x, y in moved]; src = [(x + pad, y + pad) for x, y in corners]
    note = canvas.transform(canvas.size, Image.PERSPECTIVE, _perspective_coeffs(src, dst), Image.BICUBIC)
    return note.rotate(angle, expand=True, resample=Image.BICUBIC)


def _finger(rng, size):
    """semi-transparent skin coloured rounded bar entering from the border (simulates holding the note)."""
    skin = tuple(int(c) for c in rng.choice([[224, 172, 140], [198, 134, 98], [141, 85, 60], [92, 58, 40]]))
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    fw, fl = rng.uniform(0.07, 0.11) * W, rng.uniform(0.12, 0.2) * W
    side = rng.integers(4); cx, cy = rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * W
    if side == 0: box = (cx - fw / 2, -fl * 0.3, cx + fw / 2, fl)
    elif side == 1: box = (cx - fw / 2, W - fl, cx + fw / 2, W + fl * 0.3)
    elif side == 2: box = (-fl * 0.3, cy - fw / 2, fl, cy + fw / 2)
    else: box = (W - fl, cy - fw / 2, W + fl * 0.3, cy + fw / 2)
    d.rounded_rectangle(box, radius=fw / 2, fill=skin + (255,))
    return layer


def render(note: Image.Image, rng, bg_pool, hard=False):
    """Returns (PIL 224x224 RGB, visible_fraction_of_note)."""
    for _ in range(20):
        bg = _rand_bg(rng, bg_pool).convert("RGBA")
        n = _place_note(note, rng, hard)
        nw, nh = n.size
        cx, cy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * W
        x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
        mask_full = n.split()[3]
        # soft shadow
        sh = Image.new("RGBA", (W, W), (0, 0, 0, 0)); sh.paste((0, 0, 0, 110), (x0 + 6, y0 + 8), mask_full)
        bg = Image.alpha_composite(bg, sh.filter(ImageFilter.GaussianBlur(7)))
        layer = Image.new("RGBA", (W, W), (0, 0, 0, 0)); layer.paste(n, (x0, y0), n)
        a_all = np.asarray(mask_full, dtype=np.float32).sum()
        a_vis = np.asarray(layer.split()[3], dtype=np.float32).sum()
        vis = a_vis / max(a_all, 1)
        if vis < 0.55:
            continue
        img = Image.alpha_composite(bg, layer)
        if rng.random() < (0.35 if hard else 0.25):
            f = _finger(rng, W)
            covered = (np.asarray(f.split()[3], dtype=np.float32) > 0) & (np.asarray(layer.split()[3]) > 0)
            if covered.sum() / max(a_vis / 255.0, 1) <= 0.12:
                img = Image.alpha_composite(img, f)
        break
    img = img.convert("RGB")
    # ---- photometric
    gam = rng.uniform(0.6 if hard else 0.75, 1.5 if hard else 1.35)
    arr = (np.asarray(img, dtype=np.float32) / 255.0) ** gam
    arr *= rng.uniform(0.45 if hard else 0.65, 1.25)
    arr *= rng.uniform(0.9, 1.1, 3)[None, None]                      # colour cast
    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32) / W              # uneven light / shadow
    ang = rng.uniform(0, 2 * math.pi); g = np.cos(ang) * xx + np.sin(ang) * yy
    arr *= (1 - rng.uniform(0, 0.45 if hard else 0.3) * (g - g.min()) / (g.max() - g.min()))[..., None]
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.15))
    if rng.random() < 0.65:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 3.0 if hard else 1.8)))
    elif rng.random() < 0.3:                       # motion blur: average of shifted copies
        k = int(rng.integers(4, 14 if hard else 10)); ax = int(rng.integers(2))
        arr = np.asarray(img, dtype=np.float32)
        arr = sum(np.roll(arr, i - k // 2, axis=ax) for i in range(k)) / k
        img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((224, 224), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) + rng.normal(0, rng.uniform(0, 7), (224, 224, 3))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=int(rng.integers(40, 96))); buf.seek(0)
    return Image.open(buf).convert("RGB"), float(vis)


def render_negative(rng, bg_pool, hard=False):
    """A scene without a banknote: real photo crop or plain surface, optional finger, same photometrics."""
    bg = _rand_bg(rng, bg_pool).convert("RGBA")
    if rng.random() < 0.3:
        bg = Image.alpha_composite(bg, _finger(rng, W))
    img = bg.convert("RGB")
    arr = (np.asarray(img, dtype=np.float32) / 255.0) ** rng.uniform(0.7, 1.4) * rng.uniform(0.55, 1.2)
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 2.5)))
    return img.resize((224, 224), Image.LANCZOS)
