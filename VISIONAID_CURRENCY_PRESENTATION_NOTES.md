# VisionAid – Currency Recognition: Reference Notes for My Presentation

*Written for: me, and my professors who will assess the project.*
*Status date: 2026-10-15. Branch `feat/currency-recognition`. Companion files: `VISIONAID_CURRENCY_RECOGNITION.md` (full engineering notes), `ml/currency/REAL_PHOTO_TESTING_GUIDE.md` (how I finish the real-note test), `ml/currency/reports/` (raw evidence).*

---

## 1. One-minute summary

VisionAid is an Android assistive app for blind and low-vision users in Sierra Leone. Besides recognising everyday objects, it has a **Currency mode**: the user holds one banknote to the phone camera and the app **speaks which Sierra Leone Leone note it is** (for example "New Leone, five" or "Old Leone, five thousand").

A wrong amount can cost a blind person real money, so I designed the system to **prefer silence to a wrong answer**: it announces a note only when the model is very sure **and** several consecutive camera frames agree; otherwise it says "I'm not sure. Please hold the note steady and closer."

**Honest status:** the system is built, tested offline and checked on a real phone, but I have **not yet tested it with real banknote photographs**. My training data is small (18 unique scans). Every accuracy number in this document comes from simulated phone images and is therefore an **optimistic** estimate. The real-note test is prepared and scheduled (section 12).

---

## 2. The problem and why a separate model

* The app already used a general object detector (EfficientDet-Lite0, trained on the COCO dataset). COCO has **no banknote category**, so that model cannot recognise currency.
* Currency needs a model trained on the actual notes. Because the user holds **one note close to the camera**, this is an **image classification** problem ("which note is this?"), not object detection ("where are the objects?"). A classifier is simpler, smaller and more reliable for this case.
* The old Leone (before redenomination) and the new Leone (NLe, since 2022) use **related designs**, and a note and its old counterpart can look almost identical (e.g. NLe 2 and Le 2000). The system must tell them apart and say "old" or "new" correctly.

---

## 3. The model I used

| Item | Value |
|---|---|
| Task | image classification, 10 classes |
| Architecture | **MobileNetV3-Small** (Howard et al., 2019, *Searching for MobileNetV3*), pretrained on **ImageNet**, plus Dropout(0.3) and a Dense(10) softmax layer |
| Why this model | small and fast enough for ordinary phones (CPU, a few frames per second); strong pretrained image features, so a small dataset can be enough for a first prototype |
| Method | **transfer learning** in two phases: (1) backbone frozen, train the new head (5 epochs, learning rate 1e-3); (2) fine-tune the whole network except BatchNorm layers (up to 10 epochs, lr 1e-4) with **early stopping** on validation loss (best weights restored) |
| Optimiser / loss | Adam / sparse categorical cross-entropy; batch size 32; balanced per-sample class weights; **fixed random seed 20261015** |
| Framework | TensorFlow 2.16.2, Keras 3.3.3, trained on CPU (Intel MacBook Pro, roughly 20–25 minutes for the training itself) |
| Deployed format | **TensorFlow Lite**, float32, **3.76 MB** (3,756,028 bytes), no quantisation |
| Input | one RGB image, 224 × 224 pixels, values 0–255 (rescaling is inside the model) |
| Output | 10 probabilities (softmax) that add up to 1 |
| Runtime | TensorFlow Lite interpreter on the phone CPU (2 threads) |

**The 10 classes** (output order fixed in one file, `ml/currency/labels.json`):
`NEW_1, NEW_2, NEW_5, NEW_10, NEW_20` (new Leone), `OLD_1000, OLD_2000, OLD_5000, OLD_10000` (old Leone) and `NO_NOTE` ("not one of my supported notes": includes empty scenes, hands, paper, other objects and unsupported notes such as old Le 500/100).
Old and new notes are **separate classes** and never merged. There is no old Le 20 because my data has none.

**Preprocessing in the app:** the camera frame is rotated upright, the **centre square** is cut out, shrunk to 224 × 224 and given to the model. The training images were generated in exactly this form.

The existing object detector (EfficientDet-Lite0) is a different model and was **not changed**.

---

## 4. The data I had (and its weakness)

My dataset (`ml/currency/dataset/`): 31 image files. Checking every image showed:

* **18 unique scans** = 9 notes × 2 sides (front and back). The "stack" pictures that show front and back together are the *same scans* as the separate files (image correlation 1.000), so they were excluded as duplicates.
* One collage image with all five new notes, both sides (low resolution) → 10 small crops. These came from a **different image source**, so I kept them **only for testing**.
* The folder I had named `no_note` actually contained old Le 500 and Le 100 banknotes – not empty negatives. I used them as examples of "unsupported note → say not sure".
* No photographs of notes held in a hand, folded, tilted or in poor light.

**Consequence:** the model has seen only one picture per note side. This is the biggest limitation of the whole project and it is why real photographs are required before I can make any real-world claim.

### How I made training images
Because real photos were missing, I created **simulated phone photos** from the scans (`ml/currency/scripts/synth.py`): the flat note is placed on real photo backgrounds or plain surfaces, resized (60–120 % of the view), tilted, turned 90°/180°, given perspective distortion and a soft shadow, partly covered by a finger-like shape (≤ 12 %), and then brightness, gamma, colour cast, uneven light, defocus or motion blur, noise and JPEG compression are applied. At least 55 % of the note always stays visible and the denominations stay readable.
**Negative images** ("no note"): 320 photographs from the public COCO dataset (Lin et al., 2014; Creative Commons Flickr photos) plus plain/gradient surfaces, split by image into separate train / validation / test groups.

---

## 5. How I tested (six test sets, and what each one proves)

Because I had only one scan per note side, a normal 70/15/15 random split would have leaked identical pixels between training and testing. I therefore separated the sets by **conditions and source**, with a fixed seed and no image used in more than one set:

| Set | Size | Content | What it can prove | Weakness |
|---|---|---|---|---|
| Train | 4,370 | simulated captures of the 18 scans + negatives | – | – |
| Validation | 750 | same scans, **harder** simulation, different seed and backgrounds | used to choose the confidence threshold | same scan pixels as training |
| Test-sim | 700 | same scans, third seed and third background set | robustness to capture conditions | **optimistic**: same scans |
| Test-collage | 200 | the new notes from the **collage** (different image source), simulated capture | does it generalise to a different image of the same note? | only 2 source images per note → unreliable per class |
| Test-negative | 300 | held-out real photos and surfaces without notes | does it wrongly announce a note? | few hard negatives (hands with paper, cards…) |
| Test-unsupported | 40 | an old Le 500 specimen not used in training | refuses unsupported notes? | one source image |

**Audits added later** (`ml/currency/scripts/safety_audit.py`, `reports/safety_audit.md`): confirmation that the deployed model, labels and threshold files are identical to the evaluated ones; old/new mix-ups; look-alike pairs; calibration of the confidence; sensitivity of the threshold; sensitivity to the image-resize method.

**Real-photo test (not done yet):** `evaluate_real_photos.py` scores only genuine phone photos named `real_*`; keeps synthetic images and downloaded scans out of the numbers; screens every photo for duplicates of training material (perceptual hashes + correlation, 4 rotations); applies phone EXIF rotation; reports per denomination the number of photos, correct, wrong, "not sure", accuracy over all images, accuracy among announced, coverage, a confusion matrix and false announcements on non-note photos.

---

## 6. Confidence levels – what they mean and what they do not

* The model outputs **10 probabilities**. The **confidence** is the largest one (the "top probability").
* A frame counts as a recognition only if (a) the top class is a banknote (not `NO_NOTE`) **and** (b) the top probability is **at least τ = 0.98**.
* **How τ was chosen:** on the **validation set only**, I took the first value with **zero** false note announcements on non-note images and **zero** wrong denominations (0.975), and added a **0.005 safety margin** → **τ = 0.98**. (The first version used ≤ 1 % / ≤ 0.5 % and gave 0.965; the audit found that this sat exactly on the edge of the data and that a confident NLe 2 → Le 2000 announcement (p = 0.975) occurred just above it, so I raised it. I **raised** the threshold for safety; I never lowered it to get more announcements.)
* **A score of 0.98 does NOT mean "98 % chance of being right."** I measured calibration (how often the top class is right for a given confidence):

| Top-probability bin | Validation accuracy | Collage-test accuracy |
|---|---|---|
| 0.90–0.965 | 89.1 % | 95.5 % |
| 0.965–0.99 | 96.3 % | 92.3 % |
| ≥ 0.99 | 100 % | 100 % |

  The expected calibration error is 0.040 (validation) and 0.029 (collage). The network is trained to near-zero loss, so it is **over-confident**; the number is a *score*, not a proven probability (Guo et al., 2017, *On Calibration of Modern Neural Networks*).

**Threshold sensitivity** (same model; the τ = 0.98 row is the deployed one):

| τ | Validation: non-notes falsely announced | Validation: wrong denomination | Validation: notes announced correctly | Collage: wrong announcements | Collage: announced correctly |
|---|---|---|---|---|---|
| 0.90 | 3.3 % | 0 % | 89.1 % | 2 / 200 | 77.5 % |
| 0.95 | 1.9 % | 0 % | 85.0 % | 1 / 200 | 70.0 % |
| 0.965 (old) | 1.0 % | 0 % | 81.5 % | 1 / 200 | 67.0 % |
| **0.98 (deployed)** | **0 %** | **0 %** | **75.2 %** | **0 / 200** | **61.5 %** |
| 0.99 | 0 % | 0 % | 68.0 % | 0 / 200 | 55.0 % |
| 0.995 | 0 % | 0 % | 59.1 % | 0 / 200 | 47.0 % |

Trade-off: a higher τ means fewer wrong announcements but more "I'm not sure". The collage set influenced the +0.005 margin, so it is **no longer a clean test of τ**; only real photos can judge that.

**Second safety layer – consistency over time** (`CurrencyDecider.kt`): a denomination is spoken only if **at least 4 of the last 6** analysed frames agree on the same note (each frame ≥ τ) and **at most 1** frame names a different note. It is not repeated until the note is gone for 6 frames or a different note is confirmed. If something note-like stays in view for about 14 frames without a confirmed answer, the app says "I'm not sure. Please hold the note steady and closer." (at most once per 8 seconds). Invalid model output is treated as "no note".

---

## 7. Results (all on simulated or scan-derived images – NOT real-world accuracy)

Deployed TensorFlow Lite model, τ = 0.98 (`reports/evaluation_tflite.md`; the Keras and TFLite models give identical results – maximum probability difference 1.2 × 10⁻⁵):

| Set | Result |
|---|---|
| Validation (n = 750) | top-class accuracy 90.7 %; notes announced correctly 75.2 %, **wrong 0 %**, not sure 24.8 %; non-notes falsely announced 0 % |
| **Test-sim** (n = 700; optimistic) | top-class accuracy 99.4 % (front 98.5 %, back 100 %, new 99.7 %, old 98.8 %); notes announced correctly 88.0 %, **wrong 0 %**, not sure 12.0 %; non-notes falsely announced 0 % |
| **Test-collage** (n = 200; unreliable per class) | top-class accuracy 92.5 %; announced correctly 61.5 %, **wrong 0** (at τ = 0.965 it was 1 wrong), not sure 38.5 % |
| Test-negative (n = 300) | top-class accuracy 100 %; **0** false announcements |
| Test-unsupported old Le 500 (n = 40) | **0** false announcements (90 % classified as NO_NOTE) |

Per-class results on test-sim (60 images each): every class has F1 ≥ 0.98; only 4 errors in 540 note images (NLe 10 → NLe 5, Le 2000 → Le 5000, Le 10000 → NLe 1, Le 10000 → NO_NOTE). Full confusion matrices with precision / recall / F1 are in `reports/evaluation_tflite.md`.

**Main weakness found:** on the collage (a different image of the new notes), **NLe 2 was confused with old Le 2000** in 6 of 40 predictions, and the four hardest NLe 2 crops gave "Le 2000" with probabilities 0.73–0.975. This is exactly the dangerous old/new mix-up, and it is why the system must say "not sure" in such cases. Whether it also happens on real notes is **unknown** until the real-photo test.

---

## 8. Safety design (why this is suitable to put in front of users – and what it never claims)

1. **Allowed to say "I'm not sure"** – unknown, conflicting or weak input never produces an amount.
2. **Confidence threshold + consistency over frames** (section 6).
3. **Old/new always spoken** ("New Leone, …", "Old Leone, …"), taken from one label file so an indexing error cannot swap them (a unit test checks the Android copy equals the master file and the model refuses to load if its output size does not match the number of labels).
4. **No claims about authenticity:** the app **only names the design of a note**. It does **not** say a note is genuine, counterfeit or currently legal tender, and it does not add up amounts. The screen shows this disclaimer.
5. **Battery/performance:** the camera is analysed at most about 4 times per second, with reused buffers.
6. **Failure handling:** if the model cannot load, the app shows "Currency recognition is unavailable" instead of crashing.

---

## 9. Verification on a real phone (Samsung Galaxy A72, SM-A725F, Android 14)

| Check | Result |
|---|---|
| Model loads; mode switching currency ↔ object detection | works in both directions; the object detector's output resumes immediately |
| Speed | roughly 49–93 ms per analysed frame (preprocessing + inference), first frame ≈ 207 ms; analysis limited to about 4 per second |
| **Same bytes, same answer** | I pushed 27 exact 224 × 224 test images through the real on-device model: **all 27 gave the same top-3 classes in the same order and the same probabilities (4 decimals) as my Python evaluation** → class mapping, input format and threshold are identical on the device |
| Hard NLe 2 crops on the device | all four ended as "not sure"; none announced |
| Speech | the Android text-to-speech engine accepted the text and reported start and finish for "New Leone, one" and "I'm not sure. Please hold the note steady and closer."; speaker volume was at maximum. **I have not yet confirmed by listening that the voice is audible.** |
| Real banknotes shown to the camera | **none yet** |

Automated tests: 47 Android unit tests (labels, decision logic, preprocessing, object-detector logic) and 3 JavaScript tests (mode routing) pass; the app builds.

---

## 10. What I can claim, and what I must not claim

| I CAN say | I must NOT say |
|---|---|
| A MobileNetV3-Small classifier was trained to recognise 9 Sierra Leone notes and "no note", exported to TFLite, and runs on an Android phone in under 100 ms per frame. | "It recognises Sierra Leone banknotes with 99 % accuracy." (that number is from simulated images of the same scans) |
| On simulated phone images and held-out negatives it announced no wrong denominations at τ = 0.98 and refused to announce non-notes. | "It is safe to use for payments." |
| The deployed model gives identical outputs on the phone and in Python on the same inputs. | "A confidence of 98 % means 98 % correct." (not calibrated) |
| It is designed to prefer "I'm not sure" over a wrong answer, and never claims a note is genuine or legal tender. | "It detects counterfeit notes." |
| Real-world accuracy is **unknown** and a real-photo test is prepared. | Any real-world accuracy figure before the real test exists. |

---

## 11. Limitations (stated openly)

* **Only 18 unique scans**, no real photos: real notes (worn, folded, glare, fingers, other print runs) may be recognised much worse.
* Old/new look-alike pairs (NLe 1/Le 1000, 2/2000, 5/5000, 10/10000) are the highest risk; NLe 2 vs Le 2000 was the weakest case.
* Supported notes: NLe 1, 2, 5, 10, 20 and Le 1000, 2000, 5000, 10000 only. Others (e.g. Le 500, Le 100, other countries) should give "not sure", but only 3 such examples were used.
* Few hard negatives (hands holding paper, coloured cards, receipts, screens showing notes) – false announcements on these are unmeasured.
* The confidence score is over-confident (section 6).
* Android only; English speech only (Krio planned); speech audibility to a person not yet confirmed.
* Memory use and battery drain were not measured.
* The existing JavaScript test `App.test.tsx` was already failing before this work (a test-environment problem with the camera library).

---

## 12. What happens next (plan)

1. **Real-note test** (postponed until I have the physical notes): follow `ml/currency/REAL_PHOTO_TESTING_GUIDE.md`; results are saved automatically to `ml/currency/reports/real_photo_evaluation.md`, a permanent copy in `reports/real_photo_runs/`, and one summary row per run in `reports/real_test_log.md`.
2. **My acceptance gates (set before looking):** 0 wrong denominations; 0 false announcements on non-note photos; coverage ≥ 50 %; at least 10 photos per note for a reliable row. (0 errors in *n* photos only shows the true error rate is probably below about 3/*n* at 95 % confidence – 15 % for 20 photos, 5 % for 60, 3 % for 100.)
3. If a gate fails: add real photos of the failing cases (priority NLe 2 / Le 2000) to training, retrain, and test again on a **new** set of real photos; compare before and after on independent data.
4. Then: Krio/English speech options, a second set of hard negatives, a confirmed listening test, and a field trial with a blind or low-vision user (with supervision).

---

## 13. Likely questions from the professors – and my honest answers

**Q: Why not just use the object detector (EfficientDet) for money?**
A: Its training data (COCO) has no banknote class, so it cannot know a Le 5000 note. A dedicated classifier trained on the notes is required.

**Q: Why MobileNetV3-Small and transfer learning?**
A: It is small (3.8 MB) and fast on a phone CPU, and its ImageNet-pretrained features let a small dataset reach a useful first prototype. Training a network from scratch on 18 images would not work.

**Q: Your accuracy is 99 % – is that real?**
A: No, and I do not claim it. That number is on simulated phone images made from the *same 18 scans* used for training, so it only shows robustness to simulated capture conditions. The more independent test (collage, a different image source) gave 92.5 % top-class accuracy and only 61.5 % announced correctly at the strict threshold. Real-world accuracy is unknown until I test real notes.

**Q: Why so few images? Is that enough?**
A: It is not enough for a production system. It is enough for a working prototype and for an honest pipeline. The pipeline lets me retrain as soon as I add real photos, and I documented exactly which photos are needed.

**Q: How did you avoid data leakage?**
A: Duplicate "stack" images were removed; collage crops were kept only for testing; negative photos were split by image into disjoint groups; the validation set uses different seeds and backgrounds; and the evaluation script rejects real test photos that look like training images (perceptual hash and correlation). I state clearly that the simulated sets still share the same scan content, which is why they are optimistic.

**Q: How did you choose the confidence threshold?**
A: On the validation set only: the first value with zero false announcements and zero wrong denominations, plus a safety margin of 0.005 (0.98). I later raised it after an audit found a confident wrong old/new announcement; I never lowered it. I also show the table of how the results change with the threshold.

**Q: Does 98 % confidence mean it is right 98 % of the time?**
A: No. I measured calibration: in the 0.965–0.99 bin the model was right 96.3 % of the time on validation and 92.3 % on the collage. The network is over-confident. That is why I add the 4-of-6 frame consistency rule and the "I'm not sure" behaviour.

**Q: What if it is wrong?**
A: That is the main risk. A wrong amount is worse than no answer, so the system is built to stay silent when unsure. Also, the real-note test has a gate of **zero** wrong denominations, and the app says it only names the design and cannot verify authenticity.

**Q: Why are old and new notes separate classes?**
A: They are different notes with different values (1 new Leone is worth 1,000 old Leones) and look alike. Merging them could cause a real mistake, so the app must say "New" or "Old" explicitly.

**Q: What does the app do with objects that are not banknotes?**
A: A `NO_NOTE` class was trained on empty scenes, hands, everyday photos and unsupported notes; and the app requires a confident banknote prediction in several frames. On 300 held-out non-note photos and 40 old Le 500 images there were 0 false announcements. Hard cases (a hand holding paper, receipts, other currencies) are not yet measured.

**Q: How do you know the phone gives the same result as your Python test?**
A: I ran 27 identical images through both; all 27 produced the same top-3 classes and the same probabilities to four decimals.

**Q: Does it work with the camera at the speed needed?**
A: About 49–93 ms per analysed frame on a Galaxy A72, limited to about 4 analyses per second to save battery.

**Q: Is it ready for blind users?**
A: No. It is a validated prototype pipeline. It needs the real-note test, a confirmed listening test and a supervised trial with the target users before any real use.

**Q: What would you do with more time or data?**
A: Photograph every note in many real conditions (priority: NLe 2 / Le 2000), include hard negatives and other currencies, retrain, calibrate the confidence on real validation photos, add Krio speech, and test with users.

---

## 14. Short glossary (for the slides)

* **Classification / softmax / confidence:** the model assigns a probability to each class; the largest is its "confidence".
* **Transfer learning:** starting from a network already trained on ImageNet and adapting it to my notes.
* **Argmax accuracy:** how often the most probable class is the correct one (before applying the threshold).
* **Announced / coverage / not sure:** the app announces only when confident; coverage = share of notes it announced; "not sure" = it stayed silent or said "I'm not sure".
* **False announcement:** announcing a banknote for something that is not a supported note.
* **Calibration / ECE:** whether a confidence of 0.98 really means 98 % correct; ECE = average gap between confidence and accuracy.
* **Data leakage:** accidentally testing on images that the model has already seen (gives falsely high results).
* **Synthetic / simulated image:** an image I generated from a scan to imitate a phone photo.
* **Precision / recall / F1:** correctness of the predictions for a class / share of that class found / their combination.

## 15. Suggested slide order (10 slides)

1. Problem & users → 2. System overview (two models, one app) → 3. Why a separate currency model → 4. Data and its weakness → 5. Model (MobileNetV3-Small, transfer learning) → 6. Test design (six sets, leakage) → 7. Confidence and the safety rules → 8. Results (with the "simulated, optimistic" label) → 9. Phone verification and what is still unproven → 10. Limitations and next steps (real-note test).

## 16. Where every piece of evidence lives

| Evidence | File |
|---|---|
| Labels / class order (master) | `ml/currency/labels.json` |
| Dataset audit | `ml/currency/reports/dataset_audit.md` |
| Evaluation (deployed TFLite, τ = 0.98) | `ml/currency/reports/evaluation_tflite.md` (the earlier τ = 0.965 version: `evaluation_tflite_v1_tau0.965.md`) |
| Safety audit (calibration, pairs, τ table) | `ml/currency/reports/safety_audit.md` |
| Training settings | `ml/currency/runs/training_config.json` |
| Threshold and its rule | `ml/currency/reports/thresholds.json` |
| Real-photo results (after the real test) | `ml/currency/reports/real_photo_evaluation.md`, `real_photo_runs/`, `real_test_log.md` |
| How to finish the real test | `ml/currency/REAL_PHOTO_TESTING_GUIDE.md` |
| Full engineering notes | `VISIONAID_CURRENCY_RECOGNITION.md` |

**References:** Howard, A. et al. (2019) *Searching for MobileNetV3*. Lin, T.-Y. et al. (2014) *Microsoft COCO: Common Objects in Context*. Guo, C. et al. (2017) *On Calibration of Modern Neural Networks*. TensorFlow Lite documentation (tensorflow.org/lite).
