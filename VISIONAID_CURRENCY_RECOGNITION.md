# VisionAid – My Notes on Sierra Leone Currency Recognition

Branch: `feat/currency-recognition` (nothing committed yet). Notes written 2026-10-15; real-world validation/safety audit added in section 14.

> ## Read this first (where things really stand)
> * **Built:** a separate banknote classifier (9 notes + "no note"), the training/evaluation/export pipeline, an Android native module, a "Currency mode" button, and spoken results with safety rules.
> * **Automatically tested:** 47 Android unit tests + 3 JS tests pass; I evaluated the exported TFLite model; the app builds.
> * **Tested on my Samsung A72:** the model loads, Currency mode switches on/off and back to object detection, per-frame time measured, and a note-like scene gave "not sure" instead of a guess. **I have never shown a real banknote to the camera yet.**
> * **NOT verified / NOT production-ready:** my dataset has **one scan per note side (18 images)**. There are no independent real-world photos, so **I do not know the real-world accuracy**. The numbers below are on *simulated* phone photos of those same scans, so they are optimistic. **Do not trust this with real money until I have tested real notes** (section 11 – one command prints the result).
> * It only *names a banknote design*. It does **not** detect fakes, does not say a note is genuine or still legal tender, and does not add up amounts.

## 1. My dataset and the classes I found

Lives in `ml/currency/dataset/` (my originals are never modified). `scripts/audit_dataset.py` writes `ml/currency/reports/dataset_audit.md`.

| Folder | Files | What is really inside (I looked at every image) |
|---|---|---|
| `sle_1`, `sle_2`, `sle_5`, `sle_10`, `sle_20` | 3 each | **New Leone** NLe 1, 2, 5, 10, 20: a front scan, a back scan, and one picture with front (top) + back (bottom) stacked |
| `sll_1000`, `sll_2000`, `sll_5000`, `sll_10000` | 3 each | **Old Leone** Le 1000, 2000, 5000, 10000 (same three kinds of file) |
| `sle_all_currencies` | 1 | collage of all 5 new notes, both sides (410×487, low resolution) |
| `no_note` | 3 | **NOT negatives:** old Le 500 (×2 specimens) and Le 100 banknotes (front+back) |
| `test/sle_1` | 2 | `sle_1.jpg` = byte-identical copy of `dataset/sle_1/sle_1_front.jpg`; `sle_1_1.jpg` = same scan as `sle_1_back.jpeg` at higher resolution |

**Classes (model output order = `ml/currency/labels.json`):**
`NEW_1, NEW_2, NEW_5, NEW_10, NEW_20, OLD_1000, OLD_2000, OLD_5000, OLD_10000, NO_NOTE`.
* No old Le 20 class – I have none in my data, so none was invented. Old and new are separate classes even though they are related by redenomination (e.g. NLe 2 and Le 2000 share a design).
* My **old Le 500 and Le 100 notes are sitting in `no_note`**. They are not supported denominations in this model, so they are used as "unsupported note → say not sure" examples (class `NO_NOTE`), not as blank negatives. If I want them recognised, move them to `sll_500` / `sll_100` folders – but one or two scans cannot make a reliable class.

**Data-quality findings (checked, not assumed):**
1. The "stack" pictures are the *same scans* as the front/back files (pixel correlation 1.000 at 128×64). They add nothing new and are **excluded as duplicates**. What I really have: **18 unique note-side scans** (9 notes × 2 sides) + 10 low-resolution collage crops (new notes only).
2. `test/sle_1/sle_1.jpg` and `sle_1_1.jpg` are not independent of the training data, so I do not count them as test evidence.
3. The collage crops were cut with a fixed grid; I trim the bottom 8 % to remove a sliver of the neighbouring note.
4. I have no photos of notes held in a hand, folded, in poor light, or at an angle.

## 2. How I prepare the dataset (`scripts/prepare_dataset.py`, `scripts/build_sets.py`)

* Originals untouched. Clean crops go to `ml/currency/processed/crops/` with `processed/manifest.json` (source file, crop box, view, provenance, near-duplicate group).
* With **one scan per side**, a normal 70/15/15 split of real images would leak. So I split by **conditions** and **source**, fixed seed `20261015` (saved in `processed/split_info.json`):

| Set | Samples | Content | Independence |
|---|---|---|---|
| train | 4370 | 18 scans → 160 simulated phone captures each (2880), negatives 1490 | – |
| validation | 750 | same 18 scans, **harsher** simulation, different seed and **different background photos**; Le 100 notes as unsupported | same scan pixels as training (unavoidable) |
| test-sim | 700 | same scans, third seed, third background pool | same scan pixels as training; **optimistic** |
| test-collage | 200 | NEW notes from the **collage** (different image source), simulated captures, never trained on | only **2 distinct source images per class** |
| test-negative | 300 | held-out real COCO photos (no notes) and plain surfaces | independent photos |
| test-unsupported | 40 | old Le 500 specimen #2 (not used in training) | 1 source image |

* **Simulated phone capture** (`scripts/synth.py`): note placed on real photo backgrounds or plain surfaces, size 60–120 %, tilt and 90°/180° turns, perspective, soft shadow, finger-like occluder (≤ 12 % of the note), brightness/gamma/colour cast/uneven light, defocus or motion blur, noise, JPEG compression. At least 55 % of the note stays visible and denominations stay readable.
* **Negatives:** 320 COCO val2017 photos (Flickr, Creative Commons; ids in `negatives_ids.txt`, downloaded by `scripts/fetch_negatives.py`, not committed), split by image into disjoint train 200 / val 60 / test 60, plus plain/gradient/stripe surfaces. COCO has few close-ups of hands/paper/cards, so my hard negatives are limited (see limitations).
* Leakage: no image is in more than one set; collage crops and held-out backgrounds never reach training. The same *scan content* does appear in train/val/test-sim – I note that wherever I quote those numbers.

## 3. The model

* **Architecture:** MobileNetV3-Small (ImageNet weights) + dropout 0.3 + Dense(10, softmax). Transfer learning: 5 epochs with a frozen backbone (lr 1e-3), then up to 10 epochs fine-tuning everything except BatchNorm (lr 1e-4), early stopping on validation loss with the best weights restored. Balanced class weights. TensorFlow 2.16.2, Keras 3.3.3. Config in `runs/training_config.json`.
* This is **not** EfficientDet-Lite0, and I did not touch the object detector.
* **Exported model:** `VisionAidObjectDetector/android/src/main/assets/currency_classifier.tflite`, **3,756,028 bytes**, float32 (no quantisation – see below).
  * Input: `[1, 224, 224, 3]` **FLOAT32**, RGB, **values 0–255** (the model rescales inside – not UINT8).
  * Output: `[1, 10]` FLOAT32 softmax in `labels.json` order.
  * Checked: TFLite vs Keras on 300 validation images: max |difference| 1.2e-5, argmax agreement 100 %; the full TFLite evaluation equals the Keras one (`reports/evaluation.md` vs `reports/evaluation_tflite.md`).
  * I did not quantise: 3.7 MB is fine and an unquantised model avoids an unchecked accuracy risk. `export_tflite.py --quantize dynamic` exists but I must re-evaluate before using it.
* **Preprocessing the app does** (`YuvPreprocessor.toFloatRgbSquare`): centre **square** of the upright camera frame → 224×224 (4-sample average) → float R,G,B 0–255. Training images are exactly such 224×224 square views.

## 4. Evaluation (held-out sets; threshold chosen on validation only)

Decision rule (same as the app): announce only if the top class is a banknote and its probability ≥ **τ = 0.98** (*changed from 0.965 in the round-2 audit, see section 14*). τ = the first value with **zero** false notes on NO_NOTE and **zero** wrong denominations on the **validation** set (0.975), **plus a 0.005 safety margin** (`reports/thresholds.json`). The margin was added after the audit saw a confident NEW_2→OLD_2000 announcement in the collage test, so the collage set is **no longer a clean test for this choice**.

| Set | Result at τ = 0.98 (TFLite model; the old τ = 0.965 results are kept in `reports/evaluation_tflite_v1_tau0.965.md`) |
|---|---|
| Validation (hard, n=750) | argmax accuracy 0.907; notes announced correctly 0.752, **wrong 0.000**, "not sure" 0.248; NO_NOTE falsely announced 0.000 |
| **Test-sim** (n=700, optimistic: same scans) | argmax accuracy 0.994 (front 0.985, back 1.000, NEW 0.997, OLD 0.988); notes announced correctly **0.880**, wrong **0.000**, not sure 0.120; NO_NOTE false announcements **0.000** |
| **Test-collage** (NEW only, n=200, only 2 source images per class → **unreliable**) | argmax accuracy 0.925; announced correctly 0.615, **announced wrong 0.000 at τ = 0.98 (it was 1 of 200 at τ = 0.965: a confident NEW_2→OLD_2000 swap, p = 0.975)**, not sure 0.385 |
| **Test-negative** (n=300 real photos/surfaces without notes) | argmax accuracy 1.000, **0 false announcements** |
| **Test-unsupported** (old Le 500, n=40, 1 source) | 0 false announcements (argmax said NO_NOTE 0.900) |

Test-sim per class (n=60 each, all F1 ≥ 0.98, 1–2 errors per class): NEW_10→NEW_5 ×1, OLD_2000→OLD_5000 ×1, OLD_10000→NEW_1 ×1, OLD_10000→NO_NOTE ×1. Full confusion matrices with precision/recall/F1 per class are in `ml/currency/reports/evaluation_tflite.md`.

Biggest weakness I saw (test-collage, a different rendering of the new notes): **NLe 2 was confused with old Le 2000** in 6 of 40 argmax predictions (same design family), plus 3 as NO_NOTE and 2 as NLe 5. The threshold turned most of those into "not sure" (one wrong announcement in total), but it tells me old/new look-alike pairs (1/1000, 2/2000, 5/5000, 10/10000) are the main risk, and that I do not know the *real* error rate on genuine phone photos.

Model size 3.76 MB. On the A72 (CPU, 2 threads, including preprocessing): ≈ 49–93 ms per analysed frame, first frame 207 ms (device logs). Memory and battery not measured.

## 5. How the app uses it

```
Camera frame ──► frameRouter.ts (JS worklet) ──► if currency mode ON : HybridVisionAidCurrencyRecognizer.analyze(frame)
                                              └► else                 : HybridVisionAidObjectDetector.detect(frame)   (unchanged)
```
* **Separate native object** `VisionAidCurrencyRecognizer` (`VisionAidObjectDetector/src/specs/VisionAidCurrencyRecognizer.nitro.ts`, Nitro files regenerated with `npm run specs`) inside the existing Nitro module; own model, labels, thresholds, interpreter, buffers and speech. Same camera session as before – no second camera.
* At most **~4 frames per second** are analysed; buffers are reused; a loading failure sets an error string instead of crashing (the app shows "Currency recognition is unavailable"); bad frames are logged (rate limited) and skipped.
* **Safety logic** (`CurrencyDecider.kt`, pure Kotlin, unit tested). A frame only counts if the top class is a note and p ≥ τ (0.98). A denomination is announced only when **≥ 4 of the last 6 frames** agree on the same class and **≤ 1** frame names a different note. It is not repeated until the note has been gone for 6 frames or a different note is confirmed. Invalid model output (NaN, wrong size, not a probability vector) counts as "no note". If something note-like stays in view for ~14 frames without a confirmed answer, it says **"I'm not sure. Please hold the note steady and closer."** (at most once per 8 s) and forgets any earlier result. Nothing is spoken when nothing note-like is in view.
* **Speech:** Android `TextToSpeech` (English), e.g. "New Leone, five" / "Old Leone, five thousand" (strings in `labels.json`, `spoken_en`; a Krio field can be added later). The app had no speech before; I needed no new dependency.
* **UI:** a large "Currency mode" button (TalkBack labelled), a status panel with the last spoken text and a disclaimer; tapping again returns to object detection. Leaving the app turns currency mode and speech off.
* Android only. The Nitro spec lists iOS/Swift but there is no Swift implementation, so iOS is **not** supported.

## 6. Files I created / modified

Created: `VISIONAID_CURRENCY_RECOGNITION.md`; `ml/currency/` (`README.md`, `.gitignore`, `labels.json`, `negatives_ids.txt`, `scripts/{common,audit_dataset,prepare_dataset,synth,build_sets,train,evaluate,export_tflite,make_test_samples,evaluate_real_photos,fetch_negatives}.py`, `reports/{dataset_audit.md,dataset_audit.json,evaluation.md,evaluation_tflite.md,thresholds.json,model_export.json}`, `test/*/synthetic_*.png`); assets `currency_classifier.tflite`, `currency_labels.json`, `currency_thresholds.json`; Kotlin `CurrencyLabels.kt`, `CurrencyDecider.kt`, `YuvPlaneReader.kt`, `HybridVisionAidCurrencyRecognizer.kt`; Nitro spec `VisionAidCurrencyRecognizer.nitro.ts`; regenerated `nitrogen/generated/**`; `frameRouter.ts`; tests `CurrencyLabelsTest.kt`, `CurrencyDeciderTest.kt`, `__tests__/frameRouter.test.ts`.
Modified: `App.tsx` (button, status panel, frame routing), `VisionAidObjectDetector/nitro.json` (registers the new hybrid object), `VisionAidObjectDetector/src/index.ts`, `VisionAidObjectDetector/android/build.gradle` (test-only `org.json` dependency), `YuvPreprocessor.kt` (added `toFloatRgbSquare`; existing functions unchanged), `YuvPreprocessorTest.kt` (2 tests added), `VISIONAID_OBJECT_DETECTION_IMPLEMENTATION.md` (cross reference). Object-detection behaviour and model are unchanged.

## 7. Commands to remember

Python commands run from `ml/currency/scripts` in a virtualenv (Python 3.12):
```bash
python3.12 -m venv ~/venvs/visionaid-ml && source ~/venvs/visionaid-ml/bin/activate
pip install "tensorflow==2.16.2" "keras==3.3.3" "numpy<2" pillow       # needs internet; CPU is enough (full run ≈ 40 min on my Intel MacBook)
cd ml/currency/scripts
python audit_dataset.py            # 1. audit -> reports/dataset_audit.md
python fetch_negatives.py          # 2. downloads the 320 negative photos (internet)
python prepare_dataset.py          # 3. crops -> processed/
python build_sets.py               # 4. train/val/test arrays (≈ 12 min)
python train.py                    # 5. training (downloads ImageNet weights on first run)
python evaluate.py --keras ../runs/currency_classifier.keras --calibrate      # 6. picks tau on validation, evaluates
python export_tflite.py            # 7. writes the TFLite file + labels + thresholds into the Android assets
python evaluate.py --tflite ../runs/currency_classifier.tflite --out reports/evaluation_tflite.md
python make_test_samples.py        # fills ml/currency/test with synthetic examples
python evaluate_real_photos.py     # evaluates MY real photos (section 11)
```
App and tests (repository root):
```bash
cd android && ./gradlew :VisionAidObjectDetector:testDebugUnitTest app:assembleDebug     # unit tests + build
cd .. && npx jest __tests__/frameRouter.test.ts                                           # JS test
cd VisionAidObjectDetector && npm run specs                                               # only after I change a .nitro.ts file
npm start    # terminal 1;  npm run android   # terminal 2;  adb logcat -s VisionAidCurrency
```
After adding images: re-run steps 1, 3, 4, 5, 6, 7. No GPU needed; internet is needed for step 2, the first run of step 5, and `pip install`.

## 8. Tests I ran

| Check | Result |
|---|---|
| `./gradlew :VisionAidObjectDetector:testDebugUnitTest` | **passed: 47 tests** (CurrencyLabels 7, CurrencyDecider 10, YuvPreprocessor 10, DetectionDecoder 10, DetectionStabilizer 10) – includes the existing object-detection tests |
| `./gradlew app:assembleDebug` | **BUILD SUCCESSFUL** |
| `npx jest __tests__/frameRouter.test.ts` | **passed: 3 tests** (mode routing, switching back) |
| `npx jest __tests__/App.test.tsx` | **fails, and it failed before this work too** (I checked by stashing my `App.tsx` change): Jest cannot parse `react-native-vision-camera`'s ES modules. Not fixed. |
| Python: audit, build, train, evaluate (Keras and TFLite), export | run; results in section 4 |

The tests caught a real bug that I fixed: the "I'm not sure" counter was being reset every frame once the note-gone counter passed 6 (`CurrencyDecider`). Covered: class-index→denomination mapping (and that the Android copy of the labels equals `labels.json`), old/new never merged, invalid/missing labels rejected, threshold handling, unknown-input rejection, consistency window, no repeated announcements, invalid model output, input preprocessing (centre square, rotation, 0–255 floats), mode routing. **Not unit-tested:** native model-load failure and TTS (they need a device); the failure path only logs and shows "unavailable".

## 9. On my phone (Samsung A72, SM-A725F, over adb)

Seen in the logs (2026-10-15): the currency model loaded (`10 classes, accept threshold 0.965, input [1,224,224,3] FLOAT32`); tapping the button toggled mode ON/OFF; with OFF the object detector's logs came back, with ON the currency analysis ran (`top=NO_NOTE p=0.99–1.00`, 49–93 ms); on a scene that looked slightly note-like (top class NEW_1/NEW_10/OLD_10000 at p = 0.44–0.57, below τ) the app logged `UNSURE` and made **no denomination announcement**. The TTS engine reported ready (no "speech not ready" warnings), but nobody listened to the audio.
**Not done yet:** any real banknote (new or old, front/back, angles, light, folded, hands/cards, spoken output heard by me). Those results are **unknown**.

## 10. Where things live

* Class order + spoken labels: `ml/currency/labels.json` (master) and its copy `…/assets/currency_labels.json` (a test fails if they differ; the model refuses to load if its output size ≠ number of labels).
* Threshold: `…/assets/currency_thresholds.json` (written by the export from `reports/thresholds.json`).
* Model: `…/assets/currency_classifier.tflite`.

## 11. How I test with real notes (the step that matters)

1. Take 10–20 phone photos per note, both sides, different light/angle/distance, some partly covered, plus 20+ photos of hands, paper, cards, tables. Name each `real_<anything>.jpg` and put it in `ml/currency/test/<folder>/` (`sle_1` … `sll_10000`, or `no_note`).
2. Run `python evaluate_real_photos.py`; it prints each prediction and the totals: correct / **wrong denomination** / not sure, plus false announcements on no-note photos. A wrong denomination is a serious finding.
3. To train on those photos too, copy them into `dataset/<folder>/` and re-run the pipeline – but keep a separate set of *new* photos for testing.
4. On the phone: install, tap **Currency mode**, hold one note close (fill most of the screen), listen. Check: correct amount, old vs new wording, no repetition, "I'm not sure" with a hand/paper, back to object detection with the button. Log: `adb logcat -s VisionAidCurrency` (`ANNOUNCE`, `UNSURE`, `top=… p=…`).

## 12. Limitations and warnings

* **18 unique scans; no real photos.** The model has basically memorised how these scans look under simulated conditions. Real notes (worn, folded, other print runs, glare, fingers) may be recognised much worse; the true error rate is unknown.
* Old/new look-alike pairs (NLe 1/Le 1000, 2/2000, 5/5000, 10/10000) are the highest risk; the threshold reduces but does not remove wrong announcements. NEW_2 was the weakest on the out-of-source test.
* Only Le 1000–10000 (old) and NLe 1–20 (new) are supported. Other notes (old Le 500/100/20 etc., other currencies) should give "not sure" or silence, but I only trained/tested 3 such examples.
* A wrong or missing announcement is possible; never rely on this alone for payments. It does not detect fakes, validity, or totals.
* Negatives: few hard negatives (hands holding paper, coloured cards, receipts); false announcements on those are unmeasured.
* Android only; English only (no Krio yet); the existing `App.test.tsx` Jest test is broken independently of this work.
* Memory/battery not measured; I limit analysis to ~4 frames/s to save battery.

## 13. My next actions

1. Run the real-photo test (section 11).
2. Add real photos to the dataset and retrain; add Le 500/100 only if I want them supported and have more than one scan each.
3. Add hard negatives (hands with paper, cards, receipts, other currencies).
4. Krio/English spoken text, haptic confirmation, and a "describe amount" step kept *separate* from recognition.

## 14. Round 2 – real-world validation and safety audit

**Short answer for myself: I still do not know whether it recognises genuine banknotes in real camera conditions, because there are no real photos yet.** No file named `real_*` exists anywhere in the project. Everything below is audit, tooling and device-pipeline verification – **not real-world accuracy**. Four kinds of evidence are kept apart:

| Kind | What it is | Status |
|---|---|---|
| Synthetic | simulated phone captures of my scans (`synthetic_*`, `collage_synthetic_*`, `processed/arrays`) | numbers in section 4 and `reports/safety_audit.md`; **optimistic, never real-world** |
| Offline evaluation | Python/TFLite on those arrays | done |
| Android build verification | Gradle build + 47 Kotlin unit tests + 3 JS tests | passed (this round, after my changes) |
| Real-device testing | A72 running the real model and speech | partly done (below); **no real banknote shown** |
| Genuine real photos | my own phone photos named `real_*` | **0 so far – waiting for me** |

### 14.1 What I verified in the actual files (`python safety_audit.py` → `reports/safety_audit.md`)
* Model in the Android assets = `ml/currency/runs/` model (same SHA-256); labels in assets = `labels.json`; thresholds in assets = `reports/thresholds.json`; the training class order equals `labels.json`. Model input `[1,224,224,3]` float32, output `[1,10]`.
* Old and new are separate classes (`NEW_*` / `OLD_*`) and the class names, spoken labels and indices come from one file; a unit test and the model loader reject any mismatch.
* The object detector is unaffected: its source files were last modified before the currency work (detector 18:43 and 15:58, currency 20:30+), its model SHA-256 `2e04c53b…` is unchanged, its 30 unit tests still pass, and on the phone the detector's logs came straight back after switching currency mode OFF.
* **Leakage:** `test/sle_1/sle_1.jpg` is a byte-identical copy of a training image and `sle_1_1.jpg` is the same scan at a higher resolution – both are **not counted** (the new script labels them "original scan / other"). The test-sim set uses the *same scans* as training (only backgrounds/seeds differ). The collage set is a different image source but has only 2 source images per class.

### 14.2 Findings and what I changed
1. **Confident old/new swap found.** The only wrong announcement in the collage test was a true **NLe 2 announced as old Le 2000 with p = 0.975** (at τ = 0.965). Argmax confusion NEW_2→OLD_2000 was 6 of 40 collage samples (0 of 60 on the synthetic sets). The four hardest NEW_2 collage crops give OLD_2000 with p = 0.73–0.975 – on the phone (see 14.4) they were all "not sure" at τ = 0.98. **This pair is the highest-risk confusion and I have no real photos to measure it.**
2. **Confidence is not calibrated.** On validation, the 0.965–0.99 bin has 96.3 % accuracy (confidence ≈ 97.9 %) and on the collage set 92.3 %; the 0.90–0.965 bin on validation is 89 % right. Expected calibration error 0.040 (validation), 0.029 (collage). The model is trained to near-zero loss, so a score of 0.98 must **not** be read as "98 % likely to be right".
3. **τ had no safety margin.** The first rule (≤ 1 % false notes, ≤ 0.5 % wrong) picked the very edge of the validation data. I changed it to zero errors on validation (→ 0.975) and added a **0.005 safety margin → τ = 0.98**. This lowers coverage (validation: 81.5 % → 75.2 % of notes announced). I did **not** lower the threshold; I raised it. Because the collage finding motivated the margin, the collage set can no longer count as a clean test of τ. Whether 0.98 is safe on real photos is **unknown**; a sensitivity table (τ 0.90–0.995) is in `safety_audit.md`.
4. **Evaluation tool bugs fixed:** the old `evaluate_real_photos.py` ignored EXIF rotation (phone photos would have been fed sideways), had no duplicate screening, reported only totals and counted anything in the folders. Replaced (14.3).
5. Unknown input: `NO_NOTE` is a real class; all 300 held-out non-note photos and the old Le 500 test image were never announced as a note. Not measurable here: hands holding paper, coloured cards, receipts, other currencies, a screen showing a note.
6. Resizing sensitivity: changing the downscale filter (box/bilinear/Lanczos/nearest) changed the argmax for ≤ 1 of 100 test images, so the Python tools and the app's own 4-sample averaging should agree closely on real photos.
7. The app never claims a note is genuine or legal tender (only names the design; the disclaimer is on screen).

### 14.3 The real-photo evaluation workflow (`ml/currency/scripts/evaluate_real_photos.py`)
Put **my own phone photos** in `ml/currency/test/<folder>/` named `real_001.jpg`, `real_002.jpg` … (folder = the TRUE note; `no_note` for photos with no note or an unsupported note).
```bash
cd ml/currency/scripts && source ~/venvs/visionaid-ml/bin/activate
python evaluate_real_photos.py --list        # shows what is real / synthetic / original scan (only real_* are ever scored)
python evaluate_real_photos.py               # -> reports/real_photo_evaluation.md
python evaluate_real_photos.py --selftest    # checks the tool's own arithmetic (done: OK)
```
It reports per denomination: n, correct, wrong denomination, not sure, accuracy over ALL images, accuracy among announced, coverage; plus overall numbers, a confusion matrix (true folder × what the app said, with NOT_SURE), every photo on one line, false announcements on `no_note` photos, and warnings (no camera EXIF → maybe a screenshot/download; exact duplicate photos). **Duplicate screening:** each photo is compared (perceptual hashes + correlation, 4 rotations, whole image and centre square) against all scans, the collage and the negative photos; possible matches are **excluded and listed for manual review**. I tested this with temporary fixtures (a rotated/rescaled/recompressed copy of a scan was caught at hash distance 0, correlation 1.0; an EXIF-rotated image was scored) and deleted the fixtures and the throw-away report afterwards. With no real photos it refuses to report anything. Photos are rotated by EXIF, centre-square cropped and area-resized to 224×224 like the app.
**Keep the real photos out of training and threshold tuning until I have read the report.**

### 14.4 The A72, the same bytes through the real Android model (physically tested)
Debug builds now have a self-test (`HybridVisionAidCurrencyRecognizer.runSelfTest`, only when `BuildConfig.DEBUG` and the folder exists). I pushed 27 exact 224×224 PNGs (2 per denomination from the synthetic test set, 3 + 1 + 1 negatives, and the 4 hardest NEW_2 collage crops) into the app cache and turned Currency mode on:
```bash
adb push <folder-with-224px-pngs> /data/local/tmp/currency_selftest
adb shell "run-as com.visionaid sh -c 'mkdir -p cache/currency_selftest; cp /data/local/tmp/currency_selftest/* cache/currency_selftest/'"
# restart the app, tap "Currency mode", then:
adb logcat -s VisionAidCurrency | grep SELFTEST
```
**Result: all 27 images gave the same top-3 classes in the same order and the same probabilities (to the 4 decimals logged) on the phone as in Python/TFLite** → class-index mapping, float 0–255 RGB input, class order and threshold are identical on device. Decisions at τ = 0.98: confident notes announced correctly (e.g. NEW_1, NEW_2, NEW_10, OLD_1000, OLD_2000, OLD_5000, OLD_10000), borderline ones "not sure" (e.g. NEW_5 p = 0.969, OLD_10000 p = 0.911), all negatives and the old Le 500 not announced, and **all four hard NEW_2 collage crops (top class OLD_2000, p = 0.73–0.975) → not sure, none announced**.
**Not covered by this test:** the camera YUV → RGB preprocessing path (covered by unit tests only) and real photos.

### 14.5 Speech on the A72
`TextToSpeech.speak()` returned SUCCESS and the engine fired **`onStart` and `onDone`** (new listener logging) for "New Leone, one" and for "I'm not sure. Please hold the note steady and closer."; Android reported a speech playback stream; media volume on the speaker was at maximum. **Nobody listened to the audio, so audibility by a person is still unconfirmed** – I must listen for it myself when I test.
Also found and fixed (debug self-test only): the first speak attempt came before the asynchronous TTS init finished and was dropped; real use is not affected because speech is requested only after a confirmed result.

### 14.6 Mode switching on the A72
Tapping the button: currency analysis (`top=… p=…`, `UNSURE`) ran while ON; after OFF the object detector's `frame … STABLE now:` logs resumed immediately. Verified twice on the device.

### 14.7 Test results this round
`./gradlew :VisionAidObjectDetector:testDebugUnitTest app:assembleDebug` → BUILD SUCCESSFUL, 47 tests passed (labels 7, decider 10, preprocessor 10, decoder 10, stabilizer 10); `npx jest __tests__/frameRouter.test.ts` → 3 passed; `evaluate_real_photos.py --selftest` → OK; `safety_audit.py`, `evaluate.py` (Keras and TFLite, τ = 0.98) re-run. `App.test.tsx` is still broken for the old, unrelated reason.

### 14.8 Remaining risks and what I must still do
* **No real-photo result exists.** Take the photos (section 11 / 14.3) and run the evaluation. Any wrong denomination – especially NEW_2 vs Le 2000 – is a blocker.
* More data needed, in this order: **NLe 2 and Le 2000** (both sides, many conditions), then the other look-alike pairs (NLe 1/Le 1000, 5/5000, 10/10000), then NLe 20; for every note: bright and dim light, flat and folded, held in a hand, tilted, near/far, partly covered, and with worn/dirty notes if I have them. Also 20+ photos of hands, paper, cards, receipts, other banknotes and a screen showing a note (all `no_note`).
* Re-train only **after** the first real-photo report is saved; keep a separate untouched set of new real photos for the second test; compare before/after on that independent set.
* Still true: the model has only seen 18 scans; confidence scores are over-confident; it does not detect fakes or legal-tender status; audio audibility unconfirmed; the camera YUV path on real notes untested.
