# VisionAid – My Notes on the Object Detection Pipeline

Branch where I did this: `fix/object-detection-pipeline` (based on `main` @ `18a9a6c`). Nothing committed yet.

> **Round 2 (why labels were still wrong) is in section 16.** Short version: I checked the camera-to-model pipeline against a desktop TensorFlow reference and it is correct; the wrong labels left are the model's own error rate on hard phone frames, not a decoding or preprocessing bug.
>
> **Round 1 status:** builds, 17 unit tests passed, and my first on-device run (2026-10-15, ~1 minute, a desk with a laptop) showed the pipeline works: the saved model-input image is upright and correctly coloured and detections match the scene. I have **not** finished the full list of controlled scenes (section 11).

## 1. Overview

**Problem:** the app ran the TensorFlow Lite model but results did not match the scene (e.g. "toilet", "keyboard" for a chair).
**Goal:** camera frame → correct image → model → correct class/score/box.
**What I ended up with:** the frame-to-tensor path rewritten (correct rotation, no JPEG step, strides and crop respected), the model contract checked from the `.tflite` file itself, validated and unit-tested output decoding, and diagnostics. The public React Native API (`detect(frame): void`) did not change.

## 2. The original problems

Confirmed by reading the old code and inspecting the model:

| # | Finding | Status |
|---|---------|--------|
| 1 | **No rotation was applied.** Phone cameras deliver landscape sensor pixels; with the phone upright (`rotationDegrees` = 90) the model saw a sideways image. Main cause of the nonsense classes. | Confirmed (code never read rotation/orientation) |
| 2 | YUV → NV21 → **JPEG (quality 90)** → Bitmap → resize: pointless compression, colour distortion and CPU cost every frame. | Confirmed |
| 3 | `Image.cropRect` ignored. | Confirmed (harmless if the crop is the full frame, wrong otherwise) |
| 4 | `isMirrored` ignored. | Confirmed (front camera only) |
| 5 | Error logs unlimited; output arrays re-allocated each frame; boxes printed raw (no clamping/validation); detection count `25` trusted blindly. | Confirmed |
| 6 | Input normalisation | **Not a bug.** The old code fed raw 0–255 bytes, which is what this model wants (section 4). |
| 7 | Label mapping | **Not a bug.** Zero-based lookup is right (section 6). The old log `class=73` really is `mouse`; it was just run on a sideways image. |
| 8 | Output order | **Not a bug.** Boxes, classes, scores, count – verified (section 4). |

## 3. Final architecture

```
VisionCamera frame (YUV_420_888, landscape sensor pixels)
   │  frameRouter.ts → objectDetector.detect(frame)  [Nitro]     (App.tsx routes frames here while currency mode is off)
   ▼
HybridVisionAidObjectDetector.detect()                             synchronous, @Synchronized
   ├─ unwrap NativeFrame → ImageProxy → android.media.Image
   ├─ rotation = imageProxy.imageInfo.rotationDegrees, mirrored = frame.isMirrored
   ├─ copyPlanes()          copy Y,U,V planes to reusable arrays (row/pixel strides kept)
   ├─ YuvPreprocessor       crop → rotate upright → resize 320×320 → RGB bytes (one pass)
   ├─ Interpreter.run       (created once, reused)
   ├─ DetectionDecoder      validate, clamp, label, threshold
   ├─ DetectionStabilizer   needs several consistent frames before a label is "stable"
   └─ rate-limited logs  (+ opt-in debug snapshots)
```

`App.tsx` uses `dropFramesWhileBusy: true`, so slow inference skips frames instead of queueing them. JS calls `frame.dispose()` after `detect` returns; the native code never keeps the frame, never closes or recycles anything VisionCamera owns, and copies the pixel data it needs before returning.

**Why I kept the raw interpreter and not the TFLite Task Library:** the Task Vision `ObjectDetector` would do decoding for me, but it needs an extra dependency set (`tensorflow-lite-task-vision` brings its own TFLite runtime versions that can clash with `tensorflow-lite:2.17.0`), I would still have to convert the camera `Image` to a `TensorImage`, and it hides things I need for debugging. The model's contract is simple and fully verified, so I fixed the existing interpreter code. Only one inference implementation exists in the project.

## 4. Model details

File: `VisionAidObjectDetector/android/src/main/assets/efficientdet_lite0.tflite` (4,563,519 bytes), "EfficientDet Lite0 V1", TFLite Metadata v1.2.0, Apache 2.0. Source (from `commands.md`): TensorFlow Task Library `lite-model_efficientdet_lite0_detection_metadata_1.tflite`. Not replaced.
**SHA-256 `2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b`** – byte-identical to the file Google publishes at `storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/{android,rpi}/lite-model_efficientdet_lite0_detection_metadata_1.tflite` (I downloaded and hashed both on 2026-10-15). `coco_labels.txt` SHA-256 `dc183f00…80fd` (90 lines).

Checked by parsing the flatbuffer with Python (`tflite` package) and reading the embedded metadata:

| Item | Value |
|------|-------|
| Input `serving_default_images:0` | `[1, 320, 320, 3]`, `UINT8`, scale `0.0078125`, zero point `127` |
| Input metadata text | "320 x 320, three channels (red, blue, green) per pixel. Each value is a single byte between 0 and 255." → **raw RGB bytes 0–255, no division by 255** |
| Final op | `TFLite_Detection_PostProcess` (NMS **inside the model**; `nms_iou_threshold` 0.5, `max_detections` 25, `num_classes` 90, `nms_score_threshold` −inf) |
| Output 0 | `[1,25,4]` float32 – boxes `[ymin, xmin, ymax, xmax]`, normalised 0–1 |
| Output 1 | `[1,25]` float32 – class index, zero-based |
| Output 2 | `[1,25]` float32 – score 0–1 |
| Output 3 | `[1]` float32 – detection count |

The output order is the fixed order of the post-process op (boxes, classes, scores, count). Outputs 1 and 2 have the same shape, so shape alone cannot tell them apart – the op order proves it. Because `nms_score_threshold` is −inf, the count is **always 25** (slots, not real detections), so the app has to filter by score, and it does. NMS is already done by the model, so I do not run it again.

`HybridVisionAidObjectDetector.init` re-checks input/output shapes and types at load and throws a clear error if a different model is dropped in.

## 5. Image preprocessing

All in `YuvPreprocessor.kt`, one pass, no Bitmap/JPEG:

1. **YUV layout:** planes are copied with their `rowStride`/`pixelStride`; pixel (x,y) is read at `y*rowStride + x*pixelStride` (Y) and `(y/2)*uvRowStride + (x/2)*uvPixelStride` (U,V). No assumption about tightly packed or interleaved chroma.
2. **Colour:** full-range BT.601: `R = Y + 1.402·(V−128)`, `G = Y − 0.344136·(U−128) − 0.714136·(V−128)`, `B = Y + 1.772·(U−128)`, clamped to 0–255.
3. **Channel order:** bytes are written **R, G, B** (the model's order, not BGR).
4. **Crop:** `ImageProxy.cropRect` offset/size is respected.
5. **Rotation:** `imageProxy.imageInfo.rotationDegrees` = clockwise degrees needed to make the picture upright (CameraX definition; VisionCamera turns the same number into `frame.orientation`; both are logged on the first frame). Applied exactly once, only here.
6. **Mirroring:** if `frame.isMirrored`, flipped left↔right after rotation (normally false on the back camera).
7. **Resize:** the upright image is **stretched** to 320×320 (no padding/letterbox), as the TFLite Task Library does for this model. Each output pixel averages 4 samples to limit aliasing.
8. **Quantisation:** the tensor is UINT8 with `real = (q − 127) × 0.0078125`. The metadata says the model wants raw 0–255 bytes, so **`q = pixel`** (the model's first layer does the scaling).

**Worked example for myself:** a pixel with Y=76, U=85, V=255 → R = 76 + 1.402·127 = 254, G = 76 − 0.344·(−43) − 0.714·127 = 0, B = 76 + 1.772·(−43) = 0 → bytes `FE 00 00` go into the tensor unchanged (q=254). The model reads that as (254−127)×0.0078125 = 0.992.

**Coordinates:** the whole upright image is stretched to the square, so normalised box coordinates map straight onto the upright image: `x_pixels = x_norm × uprightWidth`. Boxes are reported in that upright space (what I see holding the phone upright, before any preview "cover" cropping). No second transformation is needed.

## 6. Output decoding (`DetectionDecoder.kt`)

* **Count:** rounded and bounded to the number of slots (25). NaN → 0.
* **Class → name:** class ids are **zero-based indices into `coco_labels.txt` (90 lines)**: 0 `person`, 2 `car`, 43 `bottle`, 61 `chair`, 73 `mouse`. No background class, no +1 offset. Lines with `n/a` are COCO ids that were never used; they must stay in the file to keep the numbering, and a detection with such a class counts as *invalid*. The file's names/positions match the model's embedded `labelmap.txt` (which writes the placeholders as `???`).
* **Score:** must be finite and in 0–1, then compared with the threshold in `HybridVisionAidObjectDetector.kt` (started at 0.40, now **0.60 to start a stable label and 0.35 to keep one** – see 16.5, 16.13, 16.14).
* **Box:** `[ymin, xmin, ymax, xmax]` → clamped to 0–1 → min/max ordered → rejected if NaN or zero-sized. Output is `left, top, right, bottom`.
* Logs keep **raw slots**, **invalid**, **below threshold** and **accepted** separate.

## 7. Every file I changed

| File | Change |
|------|--------|
| `VisionAidObjectDetector/android/src/main/java/com/margelo/nitro/VisionAidObjectDetector/HybridVisionAidObjectDetector.kt` (**modified**, rewritten) | *Was wrong:* no rotation, JPEG round trip, crop/mirror ignored, per-frame output allocation, unvalidated outputs, unlimited error logs. *Now:* validates the model contract at load; reads `rotationDegrees`/`isMirrored`/`cropRect`; copies YUV planes to reused arrays; calls `YuvPreprocessor`; reuses interpreter, input buffer and output arrays; decodes through `DetectionDecoder`; `@Synchronized` so concurrent calls cannot corrupt buffers; rate-limited diagnostics (first-frame info, every 2 s: timings, raw/invalid/below-threshold/accepted counts, top 5 detections). Public `detect(frame)` signature unchanged, so no Nitro regeneration was needed then. (Later rounds: opt-in snapshots, stabilizer, threshold changes – see section 16.) |
| `.../YuvPreprocessor.kt` (**created**) | `YuvFrame`, `Orientation`, `YuvPreprocessor`: stride-aware YUV→RGB, crop, rotate, mirror, resize in one pass. No Android types, so it can be unit tested on the JVM. (Currency work later added `toFloatRgbSquare`.) |
| `.../DetectionDecoder.kt` (**created**, modified in round 2) | `Detection`, `LabelMap`, `DecodeResult`, `DetectionDecoder`: label lookup, validation, clamping, thresholding as in section 6. Round 2 added `rawSummary()` (first N raw slots before any filtering, for snapshots). |
| `VisionAidObjectDetector/android/src/test/java/com/margelo/nitro/VisionAidObjectDetector/DetectionDecoderTest.kt` (**created**) | 10 tests: 90 labels, zero-based mapping, placeholders, y/x order, clamp/order, threshold, invalid values, count handling, sorting, raw summary. |
| `.../YuvPreprocessorTest.kt` (**created**) | tests for gray, known colour, RGB order, rotation 90/180/270 on a hand-computed 4×2 image, mirror, crop offset, padded strides, upright size (+2 currency tests later). |
| `.../DetectionStabilizer.kt` (**created**, round 2) | Separate temporal-stability layer (rules in 16.6). Receives decoded detections and never alters the model output. |
| `.../DetectionStabilizerTest.kt` (**created**, round 2) | 10 tests: 3-of-5 rule, fast path, flicker, announce once, disappearance, brief gap, scene change, re-appearance, hysteresis. |
| `VisionAidObjectDetector/tools/reference_detect.py` (**created**, round 2) | Desktop reference: runs the same `.tflite` with Google's TensorFlow Lite Python interpreter on any image (modes stretch/letterbox/exact). |
| `VisionAidObjectDetector/tools/evaluate_preprocessing.py` (**created**, round 2) | Scores the model on COCO val2017 with ground truth (recall/precision) for preprocessing variants and thresholds. |
| `VisionAidObjectDetector/android/build.gradle` (**modified**) | Added `testImplementation "junit:junit:4.13.2"` so unit tests can run (later also test-only `org.json`). |
| `VISIONAID_OBJECT_DETECTION_IMPLEMENTATION.md` (**created**) | These notes. |

Not changed by this object-detection work: the model, the labels, the Nitro spec of the detector.

## 8. Dependencies and configuration

Added: `junit:junit:4.13.2` (test only). The Python tools need `tensorflow==2.16.2`, `keras==3.3.3`, `numpy<2`, `pillow` in a throw-away virtualenv (Python 3.12) – they are not part of the app. No runtime dependency changed (TFLite stays `2.17.0`); no SDK/NDK/manifest changes.

## 9. Tests and verification (round 1)

Ran and **passed**
* `cd android && ./gradlew :VisionAidObjectDetector:testDebugUnitTest` → 17 tests, 0 failures (more were added later; see 16.10 and the currency notes).
* `cd android && ./gradlew app:assembleDebug` → BUILD SUCCESSFUL.
* Model inspection (Python `tflite` + `flatbuffers` in a throw-away venv, outside the repo): tensors, quantisation, metadata, post-process options, output order, label comparison.

**My first on-device run (2026-10-15, one informal scene: a desk with a laptop, handheld camera)**
* Logs: model loaded, input `[1,320,320,3] UINT8`, `First frame: 1280x720 rotationDegrees=90 orientation=LEFT mirrored=false`, `invalid=0` on every frame.
* Saved `visionaid_preprocessed.png`: **upright, correct colours** (a laptop keyboard, motion-blurred) → rotation and RGB order work on real hardware.
* Timing: preprocess ~58 ms, inference ~47 ms per processed frame (first frame 262 ms).
* Detections were plausible: `laptop` (0.48–0.79), `keyboard` (0.46–0.50), `tv` (0.46–0.72, the laptop screen), `mouse` 0.72, `cell phone`, `person` (0.6–0.9). Doubtful ones too: `couch`, `banana`, `book`, `dining table`, `remote` at 0.41–0.67, and several frames with no detection (handheld motion blur).

**Not done yet**
* The controlled scenes: person alone, empty wall, chair, bottle, vehicle – no results claimed.

## 10. How I run VisionAid

From the repo root (scripts from `package.json`):
```
npm install
npm start                 # Metro dev server (terminal 1)
npm run android           # build + install on the connected device (terminal 2)
```
or build only: `cd android && ./gradlew app:assembleDebug`.
Check the device: `~/Library/Android/sdk/platform-tools/adb devices`.
Logs: `adb logcat -s VisionAidTFLite`.
(If the app cannot load JS: `adb reverse tcp:8081 tcp:8081`.)

## 11. How I test object detection

1. Connect the phone (USB debugging on), confirm `adb devices` shows it, run `npm start` and `npm run android`, grant camera permission.
2. `adb logcat -s VisionAidTFLite`. On startup I should see the model, `Input: shape=[1, 320, 320, 3] type=UINT8 scale=0.0078125 zeroPoint=127`, four `Output[...]` lines, and `First frame: 1280x720 rotationDegrees=90 ...` (portrait phone).
3. Every ~2 s: `frame … | preprocess=Xms inference=Yms | raw slots=25 invalid=0 belowThreshold(…)=N accepted(…)=M strong(…)=K`, then `DETECTED (this frame): person (class=0) score=0.80 box=[...]` and `STABLE now: …`.
4. **Check the image fed to the model** with snapshots (section 16.7): it must be **upright**, correctly coloured (a red object looks red) and slightly stretched. If it is sideways/upside-down the rotation is wrong – check the `First frame` log line.
5. Try: me standing (expect `person`), a blank wall (few or no detections; `strong=0` is fine), a chair, a bottle, a car/bus outside. Box meaning: `l,t,r,b` as fractions 0–1 of the upright image (`l=0.1` = 10 % from the left edge).

## 12. Known limitations

* Only a few informal device runs so far (see 9 and 16). Handheld motion blur clearly lowers scores – hold the phone steady when testing.
* EfficientDet-Lite0 is small: it misses small, distant, partly hidden or unusual objects and confuses similar ones; it only knows the 80 real COCO classes (no currency, nothing Sierra Leone–specific).
* `detect` returns nothing to JavaScript; results are only logged. Showing boxes or speaking them needs a Nitro API change (e.g. return an array of detections) – the natural next step.
* The preview may crop ("cover") differently from the full upright image, so boxes I draw later must account for it.
* Stretching (not letterboxing) distorts the aspect ratio a bit, same as Google's reference implementation.
* The CPU preprocessing loop is 320×320×4 samples per frame; speed is acceptable now but not optimised.
* Assets are read through the class loader (`assets/…` inside the APK), as in the earlier working code.

## 13. Troubleshooting

* **Model fails to load:** look for `VisionAidTFLite` errors / a Nitro creation error; the message names the failed contract check or missing asset. Rebuild so assets are packaged.
* **No detections:** check `raw slots` / `belowThreshold`; if scores are all low, look at a snapshot PNG (black/odd image = preprocessing problem).
* **Wrong labels:** compare the logged `class=` with line N+1 of `coco_labels.txt` (zero-based). Never delete the `n/a` lines.
* **Boxes in the wrong place:** they are in upright-image fractions; check the PNG orientation first.
* **Repeated false positives:** raise the threshold only after confirming the PNG looks right.
* **Rotation problems:** compare `rotationDegrees` and `orientation` in the `First frame` log, and the PNG.
* **Build errors:** `cd android && ./gradlew clean`, then `app:assembleDebug`; Java 17 required.
* **Native module errors:** after changing a `.nitro.ts` spec, run `npm run specs` in `VisionAidObjectDetector` and rebuild.

## 14. Rollback and maintenance

* Work is on branch `fix/object-detection-pipeline` (uncommitted until I commit). Review with `git status` and `git diff main`.
* To undo only the first round: `git checkout main -- VisionAidObjectDetector/android/build.gradle VisionAidObjectDetector/android/src/main/java/com/margelo/nitro/VisionAidObjectDetector/HybridVisionAidObjectDetector.kt` and delete the new files listed in section 7 (or just switch back to `main`).
* The untracked `.vscode/` folder was there before and I never touched it.

## 15. Final change summary (round 1)

- [x] Model contract verified from the file
- [x] Rotation / mirror / crop / strides handled; JPEG step removed
- [x] Output decoding validated, labels verified (zero-based)
- [x] Diagnostics added (rate-limited)
- [x] Unit tests: 17 passed
- [x] `app:assembleDebug`: succeeded
- [x] On-device run: rotation/colour verified, plausible detections (one informal scene)
- [ ] Controlled real-scene tests (wall, chair, bottle, vehicle): **not done**

---

## 16. Round 2 – why were the labels still wrong? (my investigation, with evidence)

On my Samsung A72 (SM-A725F) I saw: empty space → `tv`, bottle → `dining table`, chair → `book`, person sometimes wrong. I tested each possible cause in turn.

### 16.1 Hypotheses and verdicts

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| 1/2 | The image reaching the model is not the camera image / conversion wrong | **Rejected** | Snapshot pairs: `snap_N_frame.png` (upright camera frame) and `snap_N_input.png` (exact model bytes) show the same scene, upright, correct colours (laptop, orange sleeve, flag). Both written inside the same synchronized `detect()` call from the same buffer. |
| 3 | RGB order / pixel interpretation wrong | **Rejected** | The orange object is orange in the input PNG; unit test `outputIsRgbOrder`; model metadata says raw 0–255 RGB. |
| 4 | Output tensors decoded wrongly | **Rejected** | The same 320×320 phone bytes fed to desktop TensorFlow (`reference_detect.py --mode exact`) give the same class, box and score as the phone for snap 0, 4, 7 (e.g. laptop 0.6875 on the phone vs 0.688 on the desktop; boxes equal to 3 decimals; scores differ only by float16/quantisation rounding). If the output order were wrong, `invalid` would not be 0 on every frame (classes would be fractions). |
| 5 | Label mapping wrong | **Rejected** | `evaluate_preprocessing.py` uses COCO ground truth with `model index = COCO category id − 1`; 63 % of ground-truth objects are found with the right class and 67 % of confident detections are right. A wrong mapping would score ≈ 0 %. Reference run on sharp COCO images: living room → `tv`, `chair`, `chair`, `potted plant`; bedroom → `bed`, `potted plant`, `book`; stop sign → `stop sign 0.88`. |
| 6 | Wrong model file | **Rejected** | SHA-256 equals Google's published EfficientDet-Lite0 file (section 4). |
| 7 | Scores/boxes interpreted wrongly | **Rejected** | Boxes agree with the reference to 3 decimals; reference boxes sit where the objects are in the COCO images. |
| 8 | Model makes ordinary mistakes | **Confirmed** | On clean COCO images with the verified pipeline: at score ≥ 0.4 recall 61–63 %, precision 66–67 %, i.e. roughly **1 in 3 confident detections is wrong even on clean photos**. |
| – | Stretching a portrait frame into the square hurts | **Weak effect only** | Letterboxing instead of stretching: recall +2 pts, precision +4 pts in a portrait view (16.3). Not adopted: small gain, extra coordinate-remapping code. |
| – | Live frames are harder than COCO | **Plausible, partly evidenced** | Many saved phone frames are visibly motion-blurred or defocused (snap_0, snap_4, snap_5), and snapshots of empty/blank views give only low scores (snap_2, snap_3: best 0.03). Not proven to be the only cause: snap_6 is a *sharp* frame and the model still scored its real objects (laptop, seated person, chair, bag) only ≈ 0.24. |
| 9 | Model not suitable for the product | **Open** | See 16.4. |

**Conclusion:** I found no preprocessing, decoding, label or model-identity bug in this round. The earlier fixes (rotation, no JPEG, strides) stand. The wrong labels are the model's behaviour on difficult camera frames, made worse by reporting single frames.

### 16.2 Phone vs reference on the same bytes (my "trusted reference" check)

| Snapshot | Phone (Android TFLite 2.17) | Desktop TF 2.16.2 (same bytes) |
|----------|------------------------------|--------------------------------|
| snap_0 | laptop 0.6875, car 0.184, car 0.148 | laptop 0.688, car 0.184, car 0.148 |
| snap_4 | person 0.605, cell phone 0.434, hot dog 0.117 | person 0.605, cell phone 0.434, hot dog 0.117 |
| snap_7 | laptop 0.723, keyboard 0.543, keyboard 0.066 | laptop 0.723, keyboard 0.543, keyboard 0.066 |

The reference is the same `.tflite` file on Google's own interpreter, so this is a runtime check, not an accuracy check against a different model.

### 16.3 Known-image tests (desktop, same model)

Images: COCO val2017 (annotations © COCO Consortium, CC BY 4.0; images are Flickr photos under CC licences listed per image in the COCO annotation file `instances_val2017.json`), IDs 139 (living room), 632 (bedroom), 724 (stop sign). Results are in 16.1 (#5). **Not tested** because I did not find them among the downloaded images: a clear single-bottle close-up and an empty wall. COCO has no "empty wall" class and I did not call any image empty.

Numbers (first 200 val2017 images, GT objects ≥ 2 % of image, IoU ≥ 0.5, `evaluate_preprocessing.py`):

| View | Resize | Recall | Precision |
|------|--------|--------|-----------|
| native 4:3 | stretch (current) | 0.61 | 0.66 |
| native 4:3 | letterbox | 0.63 | 0.67 |
| portrait 9:16 crop | stretch (current) | 0.56 | 0.68 |
| portrait 9:16 crop | letterbox | 0.58 | 0.72 |

Threshold sweep, portrait + stretch (current pipeline): 0.4 → recall 0.56 / precision 0.68; 0.5 → 0.48 / 0.77; 0.6 → 0.33 / 0.84; 0.7 → 0.21 / 0.93.

### 16.4 Is the model suitable? Alternatives (checked, NOT adopted)

EfficientDet-Lite2 (Google Task Library file, 7.6 MB, 448×448, same post-process contract) on the same 200 images: portrait stretch recall 0.60 / precision 0.64, letterbox 0.65 / 0.66 (at 0.4). A few points better recall, ≈ 3× the compute, no precision gain – not enough to switch without on-device latency tests. Lite0 is the right *speed* class, but its accuracy ceiling (~1/3 wrong confident detections on clean photos) is a real limit for an assistive product. Options for later: a bigger model once I have on-device timing, a model fine-tuned on local data, and a separate model for currency (done, see the currency notes). Nothing guarantees any of these fixes the behaviour in the field.

### 16.5 Threshold

First changed 0.40 → 0.50 using the sweep above (precision 0.68 → 0.77 for a recall cost 0.56 → 0.48). This is a precision/recall trade-off, **not** a fix: it cannot correct a confident wrong label. Changed again to 0.60 in 16.14.

### 16.6 Temporal stability (`DetectionStabilizer`)

Added after the checks, separate from inference and decoding. Rules (in processed frames): stable after ≥ 3 hits in the last 5; fast path ≥ 2 hits when score ≥ 0.80; dropped after 3 consecutive misses; each stable period announced once (`STABLE appeared`) and re-announced only after it was dropped; flickering labels never become stable. Trade-offs: 2–3 frame delay, true objects that flicker get suppressed, tracking is per class (two chairs count as one), no box matching. Logs: `STABLE appeared/disappeared/now`. Not exposed to JavaScript yet. **Unit tested (passing). Installed on my A72 later the same day:** the `STABLE now:` log line works and the opt-in snapshot flag was not set (no new snapshots). In a ~42 s run (21 summary logs) the phone looked at a nearly featureless view: every frame had `accepted=0`, so the stabilizer had nothing to confirm; behaviour on real objects was at that point only covered by unit tests (16.13 and 16.15 add real runs).

### 16.7 Diagnostics and snapshots

The periodic log (every 2 s) separates raw slots, invalid, below-threshold, accepted, and stable. Snapshots (exact input PNG, upright frame, raw top-10 output) are saved to `cache/snapshots/` only on a debug build and only while `cache/ENABLE_SNAPSHOTS` exists:
```
adb shell run-as com.visionaid touch cache/ENABLE_SNAPSHOTS      # enable
adb shell run-as com.visionaid rm cache/ENABLE_SNAPSHOTS         # disable
adb shell run-as com.visionaid ls cache/snapshots
adb exec-out run-as com.visionaid cat cache/snapshots/snap_0_input.png > snap_0_input.png
```
Note: the build I used to collect the evidence above had snapshots on automatically in debug builds; the opt-in flag came afterwards and I have not run it on the phone.

### 16.8 Reproducing the desktop tests

```
python3.12 -m venv ref && ref/bin/pip install "tensorflow==2.16.2" "keras==3.3.3" "numpy<2" pillow
ref/bin/python VisionAidObjectDetector/tools/reference_detect.py --top 6 some.jpg          # stretch + letterbox
ref/bin/python VisionAidObjectDetector/tools/reference_detect.py --mode exact snap_0_input.png
ref/bin/python VisionAidObjectDetector/tools/evaluate_preprocessing.py instances_val2017.json ./val2017 200
```
(COCO annotations: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`.)

### 16.9 Performance (A72, debug build, round-2 logs)

Preprocessing ≈ 57–72 ms, inference ≈ 46–58 ms per processed frame. I did not measure the processed-frame rate directly (logs sample once every 2 s); the per-frame work is ≈ 105–130 ms, so at most roughly 8–9 frames per second. First frame 262 ms. Snapshot writing slows the frames where it happens (it is opt-in). Preprocessing is slower than inference, so it is the first thing to optimise (e.g. fewer samples per pixel). Not measured: memory, sustained thermal behaviour.

### 16.10 Tests I ran (round 2)

* `cd android && ./gradlew :VisionAidObjectDetector:testDebugUnitTest app:assembleDebug` → BUILD SUCCESSFUL; 28 unit tests for the detector side (10 decoder, 10 stabilizer, 8 preprocessor) after the hysteresis change (26 before), 0 failures. With the currency work the total is now 47.
* Desktop: `reference_detect.py` on 3 COCO images and 3 phone snapshots; `evaluate_preprocessing.py` on 200 images for Lite0 and Lite2 and four thresholds.

### 16.11 What I have and have not verified on the phone

* Done (A72, earlier build): saved frames/inputs inspected, raw outputs compared with the reference (16.2), live logs watched for several minutes while the phone mostly looked at a laptop/desk. These were **not** controlled scene tests.
* The build with threshold 0.50 + stabilizer, one ~42 s run on an essentially empty view: 21/21 summary logs had `accepted=0`, `invalid=0`, no false label (with the previous 0.40 threshold similar views gave `tv`/`person`-type guesses). One observation, not a test series. Preprocess 57–61 ms, inference 46–49 ms.
* **Not done:** the controlled scenes (empty wall, person at a known distance, chair, bottle, TV, multi-object, movement) with repeated frames. My own reports (empty → tv, bottle → dining table, chair → book) were therefore **not reproduced and not explained frame-by-frame**.
* To finish: reconnect, `npm run android`, run the snapshot commands in 16.7 while pointing at each scene for ~20 s, then run `reference_detect.py` on the pulled `snap_*_frame.png` (both modes). That separates "model weak on this frame" from anything phone-specific.

### 16.12 Remaining limitations / my next actions

1. Run the controlled device scenes above (most important).
2. Decide on a stronger model only after I have on-device latency numbers (Lite2 is the candidate, 16.4).
3. Optimise preprocessing (57–72 ms) if frame rate matters.
4. Return stable detections to JavaScript (Nitro API change) so the app can show or speak them.
5. Look at frame blur/exposure on the A72 (camera exposure/FPS/focus settings in VisionCamera) – a hypothesis, not a finding.

### 16.13 My device log of the 0.50 + stabilizer build (2026-10-15 18:32–18:35, ~3.5 min)

What I saw in the log (I did not note what was in front of the camera, so I only judge right/wrong where I said so):
* **First ~2.5 min, camera on a plain view: 0 detections ≥ 0.50 in every one of ~65 summary logs, 0 invalid.** Before the threshold/stabilizer change, similar views produced `tv`/`person` guesses – so the "empty space" case is now clean.
* Objects then appeared: `chair` 0.71–0.80, `laptop` 0.50–0.74, `person` 0.54–0.91, `tv` 0.52–0.65, `remote` 0.54–0.76 and a tall, narrow `bottle` 0.61–0.65 (box about 30 % wide, 60 % tall) held stable over ~8 s. I did not note whether `tv`/`remote` were really in view, so those stay unverified.
* Stabilizer events work (`STABLE appeared/disappeared/now`).
* **Defect I spotted:** labels near the threshold flapped: `person` appeared 18:35:33.7, disappeared 35.5, appeared 36.7, disappeared 39.1; `laptop` disappeared 36.4 and re-appeared 36.9. Each re-appearance would be announced again. Cause: a stable class was dropped after 3 frames below 0.50 even though scores were 0.4–0.5.
* **Fix (hysteresis):** detections are decoded at `KEEP_THRESHOLD` 0.35; a class needs ≥ `CONFIDENCE_THRESHOLD` (then 0.50) to start/build up a track, but a stable class is kept alive by detections down to 0.35. Changed `HybridVisionAidObjectDetector.kt` and `DetectionStabilizer.kt` (new `enterScore`), plus 2 unit tests (`weakDetectionsCannotStartATrack`, `weakDetectionsKeepAStableTrackAlive`). Log lines show `accepted(>=0.35)` and `strong(>=…)`; `DETECTED (this frame)` lists only strong detections. 28 detector-side unit tests pass; installed on the A72 and starts correctly (checked on an empty view only; the flapping fix was **not yet seen on real objects** then).
* **Timing drift:** preprocessing went from 57 ms to 71–72 ms and inference from 46 ms to 57 ms over the run, one 103 ms spike. Maybe CPU thermal throttling or frequency scaling – **not verified**.
* A stable label appeared quickly: `bottle` about 0.3 s after I first pointed at it.

### 16.14 Tightening: entry threshold 0.50 → 0.60 (my log 18:38–18:41)

I reported detections of things that were not there (`dining table`, `cell phone`; the log also had `hot dog` and `bed`). In that log (stabilizer build, enter 0.50) the objects that were not there scored 0.50–0.61 per frame (`dining table` 0.50–0.59, `cell phone` 0.50–0.57, `hot dog` 0.52–0.61, `bed` 0.50), while the objects that really were there (`person` 0.85–0.91, `chair` 0.71–0.76, `bottle` 0.63–0.74, `laptop` 0.65–0.74) scored higher. The COCO sweep (16.3) points the same way: precision 0.77 / 0.84 / 0.93 at 0.5 / 0.6 / 0.7 with recall 0.48 / 0.33 / 0.21.

Change: `CONFIDENCE_THRESHOLD` (stabilizer `enterScore`) 0.50 → **0.60**; keep-threshold stays 0.35. A label must now score ≥ 0.60 in at least 3 of 5 frames (or 2 frames at ≥ 0.80) to count as stable. On the logged data this would have blocked `dining table`, `cell phone` and `bed` completely, and `hot dog` too (only one frame ≥ 0.60).

Caveats (not verified): I tuned this on one short log plus a 200-image COCO sample, so it may overfit; recall drops (small or partly hidden real objects scoring 0.4–0.6 will not be announced any more, e.g. a `chair` that scored 0.50 at the frame edge); I did not use "full-frame box" as a rule because I have no evidence it generalises. Per-class thresholds would be the next tightening, but 200 images are too few per class to set them responsibly. 28 detector-side unit tests pass; the 0.60 build is installed on the A72.

### 16.15 My device log of the 0.60 build (18:47–18:50, ~3 min)

* ~2 min of a plain view: `accepted(>=0.35)=2` on every log line (weak guesses below 0.60) but `strong(>=0.6)=0` and `STABLE now: none` → no false announcements.
* Then `laptop` 0.72–0.83, `chair` 0.65–0.79, `bottle` 0.63–0.69, `person` 0.65–0.95 became stable. **`dining table`, `cell phone`, `hot dog` and `bed` did not appear at all** (they did with the 0.50 build in the previous log). One run, not a controlled test; I did not report any wrong label in this run.
* Remaining flapping: `person` appeared 18:50:14.8, disappeared 18:50:15.8, re-appeared 18:50:17.2 (the camera moved so that no detection ≥ 0.35 existed for 3 frames); a stable label is dropped after 3 empty frames by design.
* Currency recognition is **not** possible with this model: COCO has no banknote class. That is why I built a separate currency model (see `VISIONAID_CURRENCY_RECOGNITION.md`).

---
**See also:** `VISIONAID_CURRENCY_RECOGNITION.md` – the separate currency model and "currency mode" live on branch `feat/currency-recognition`. The object-detector code and model were not changed by that work (only `YuvPreprocessor.kt` got an extra, separate function, and `App.tsx` now routes frames through `frameRouter.ts`).
