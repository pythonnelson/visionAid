#!/usr/bin/env python3
"""
Measures how the SAME model behaves on COCO val2017 images with ground truth, for different
ways of preparing the image. Used to decide stretch vs letterbox, and to check that the
label mapping (model class index == COCO category id - 1) matches real annotations.

  python evaluate_preprocessing.py instances_val2017.json /path/to/val2017_dir [N]

Images are downloaded on demand from images.cocodataset.org if missing.
Two views: "native" (image as is) and "portrait" (centre 9:16 crop, like a phone held upright).
"""
import json, sys, pathlib, urllib.request
import numpy as np
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from reference_detect import prep, run

import os
THRESH, IOU = float(os.environ.get("VISIONAID_THRESH", 0.4)), 0.5


def iou(a, b):  # boxes as x0,y0,x1,y1
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0


def main():
    ann = json.load(open(sys.argv[1])); d = pathlib.Path(sys.argv[2]); d.mkdir(exist_ok=True, parents=True)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    by_img = {}
    for a in ann["annotations"]:
        if not a["iscrowd"]:
            by_img.setdefault(a["image_id"], []).append(a)
    imgs = sorted(ann["images"], key=lambda i: i["id"])[:n]
    stats = {}
    for info in imgs:
        p = d / info["file_name"]
        if not p.exists():
            urllib.request.urlretrieve("http://images.cocodataset.org/val2017/" + info["file_name"], p)
        full = Image.open(p).convert("RGB"); W, H = full.size
        for view in ("native", "portrait"):
            if view == "portrait":
                cw = min(W, int(H * 9 / 16)); x0 = (W - cw) // 2
                img = full.crop((x0, 0, x0 + cw, H)); off = (x0, 0)
            else:
                img, off = full, (0, 0)
            w, h = img.size
            gts = []
            for a in by_img.get(info["id"], []):
                bx, by, bw, bh = a["bbox"]
                b = [max(0, bx - off[0]), max(0, by - off[1]), min(w, bx + bw - off[0]), min(h, by + bh - off[1])]
                if b[2] > b[0] and b[3] > b[1] and (b[2]-b[0])*(b[3]-b[1]) >= 0.02 * w * h:
                    gts.append((a["category_id"] - 1, b))          # model index = COCO id - 1
            for mode in ("stretch", "letterbox"):
                boxes, classes, scores, _ = run(prep(img, mode))
                dets = []
                for i in range(len(scores)):
                    if scores[i] < THRESH: continue
                    y0, x0_, y1, x1 = boxes[i]
                    if mode == "letterbox":   # undo padding
                        s = min(320 / w, 320 / h); pw, ph = (320 - w*s)/2/320, (320 - h*s)/2/320
                        x0_, x1 = (x0_ - pw)/(1-2*pw), (x1 - pw)/(1-2*pw); y0, y1 = (y0 - ph)/(1-2*ph), (y1 - ph)/(1-2*ph)
                    dets.append((int(round(classes[i])), [x0_*w, y0*h, x1*w, y1*h]))
                st = stats.setdefault((view, mode), [0, 0, 0, 0])  # gt, gt hit, det, det correct
                for c, b in gts:
                    st[0] += 1; st[1] += any(c == dc and iou(b, db) >= IOU for dc, db in dets)
                for dc, db in dets:
                    st[2] += 1; st[3] += any(c == dc and iou(b, db) >= IOU for c, b in gts)
    print(f"{len(imgs)} images, score>={THRESH}, IoU>={IOU}, GT objects >=2% of image")
    for (view, mode), (g, gh, dn, dc) in sorted(stats.items()):
        print(f"{view:9s} {mode:9s} recall={gh/max(g,1):.2f} ({gh}/{g})  precision={dc/max(dn,1):.2f} ({dc}/{dn})")


main()
