# Real banknote test – my step-by-step guide (to resume later)

Written for myself. The real-currency test is **postponed** because I do not have all the physical notes with me. This file is everything I need to pick it up again without asking anybody. Presentation material is in `VISIONAID_CURRENCY_PRESENTATION_NOTES.md` (repo root).

## 0. Where I stopped (state on 2026-10-15)

| Item | State |
|---|---|
| Branch | `feat/currency-recognition` (nothing committed yet) |
| Model | `VisionAidObjectDetector/android/src/main/assets/currency_classifier.tflite`, SHA-256 starts `019a3e77`, 3.76 MB |
| Accept threshold τ | **0.98** (`currency_thresholds.json`) |
| Real photos in the project | **0** (no `real_*` file exists) |
| Real-world accuracy | **unknown** – every number so far is on simulated images |
| Phone | Samsung A72 – model, mode switching and speech events verified; **no real note was ever shown** |
| Not yet confirmed | that I can actually **hear** the spoken result |

## 1. Get ready (10 minutes, once)

```bash
cd /Users/mac/Desktop/dev/mobile/visionAid
git branch --show-current                      # must say: feat/currency-recognition
source ~/venvs/visionaid-ml/bin/activate       # my Python 3.12 environment (TensorFlow 2.16.2) – already created
cd ml/currency/scripts
python evaluate_real_photos.py --selftest      # must print: selftest OK
python evaluate_real_photos.py --list          # shows what is real / synthetic / original scan
```
If `~/venvs/visionaid-ml` is ever missing, recreate it (needs internet):
```bash
pip install uv && uv venv --python 3.12 ~/venvs/visionaid-ml
uv pip install --python ~/venvs/visionaid-ml/bin/python "tensorflow==2.16.2" "keras==3.3.3" "numpy<2" pillow
```

## 2. What to photograph (the plan)

I have 9 notes: **NLe 1, 2, 5, 10, 20** (new) and **Le 1000, 2000, 5000, 10000** (old). Take **real physical notes only** (not a picture on a screen or a printout).

| What | Minimum (quick test) | Good test | Why |
|---|---|---|---|
| each note, **front** | 5 photos | 10–20 | the model must work on both sides |
| each note, **back** | 5 photos | 10–20 | |
| **no_note** photos | 20 | 40+ | to prove it does NOT announce an amount for non-notes |

Total for the good test: 9 notes × 20 + 40 = about **220 photos**. If time is short, do **NLe 2 and Le 2000 first** (the pair the model confuses most), then NLe 1/Le 1000, 5/5000, 10/10000, then NLe 20.

**Conditions to spread across each note's photos** (tick them off):
- [ ] bright light  [ ] dim light  [ ] shadow across the note  [ ] near a window / glare
- [ ] note flat on a table  [ ] held in one hand  [ ] held in two hands  [ ] slightly folded or curled
- [ ] tilted a little  [ ] landscape and portrait phone  [ ] close (note fills the screen) and a bit further away
- [ ] partly covered by a finger  [ ] worn or dirty note (if I have one)

**no_note photos** (put these in the `no_note` folder): my hand, a blank sheet, a notebook page with writing, a coloured card, a receipt, a phone screen showing a banknote, a table or cloth, **a banknote of another country**, and **an old Le 500 / Le 100 note** (the model should NOT announce these as a supported note).

## 3. How to take the photos (rules that keep the test honest)

1. Use the phone's normal **Camera app**, JPEG. No filters, no beauty mode, no editing, no cropping, no screenshots.
2. **One note per photo**, the note clearly visible (about 60–100 % of the frame width).
3. Do **not** send photos through WhatsApp/Telegram/Messenger – they strip the camera information (EXIF) and rotation. Copy them with a cable (next step).
4. Take **two separate sets** if possible: **Set A** for testing now, **Set B** kept untouched for the test **after** any retraining. Put Set B in a different folder and do not use it yet.
5. Never use the same photo twice and never use a photo from the internet.

## 4. Copy the photos from the phone to the Mac

Phone connected by USB, USB debugging on:
```bash
~/Library/Android/sdk/platform-tools/adb devices                      # phone must say "device"
~/Library/Android/sdk/platform-tools/adb shell ls -t /sdcard/DCIM/Camera | head -30   # newest photos first
mkdir -p ~/Desktop/currency_photos
~/Library/Android/sdk/platform-tools/adb pull /sdcard/DCIM/Camera ~/Desktop/currency_photos/all_from_phone
```
Then, in Finder, **sort the photos into one folder per kind**, e.g. `~/Desktop/currency_photos/nle2_front`, `nle2_back`, `le2000_front`, `hands_and_paper`, ... (any names I like; one kind of note per folder; front and back can be in the same folder).

## 5. Put them in the project (names and folders are done by a script)

Folder name = the TRUE note: `sle_1 sle_2 sle_5 sle_10 sle_20` (new Leone 1, 2, 5, 10, 20), `sll_1000 sll_2000 sll_5000 sll_10000` (old Leone), `no_note`.
```bash
cd /Users/mac/Desktop/dev/mobile/visionAid/ml/currency/scripts
python import_photos.py ~/Desktop/currency_photos/nle2_front   sle_2
python import_photos.py ~/Desktop/currency_photos/nle2_back    sle_2
python import_photos.py ~/Desktop/currency_photos/le2000_front sll_2000
python import_photos.py ~/Desktop/currency_photos/hands_and_paper no_note
```
The script copies each photo to `ml/currency/test/<folder>/real_001.jpg, real_002.jpg …` (numbers continue, nothing is overwritten, importing the same folder twice does nothing, my originals stay untouched). It warns if a photo has no camera EXIF. The photos are git-ignored (they are large) – **I back them up myself** (e.g. copy `~/Desktop/currency_photos` to a drive).
**A photo in the wrong folder gives a wrong result, so double-check each folder name.**

## 6. Run the evaluation

```bash
cd /Users/mac/Desktop/dev/mobile/visionAid/ml/currency/scripts && source ~/venvs/visionaid-ml/bin/activate
python evaluate_real_photos.py --list     # check: "real photos=N" and that N is what I imported
python evaluate_real_photos.py            # scores them
```

### Where the result goes (all created automatically)

| File | What it is |
|---|---|
| `ml/currency/reports/real_photo_evaluation.md` | the **latest** full report (overwritten each run) |
| `ml/currency/reports/real_photo_runs/<date>_tau<τ>_<modelsha>.md` | a **permanent copy of every run** (never overwritten) |
| `ml/currency/reports/real_test_log.md` | **one table row per run** (date, model, τ, how many photos, correct, WRONG, not sure, false announcements) – the quick history |

(The name is `real_photo_evaluation.md`, with "photo".) These files are the proof I show my professors.

## 7. What to expect and how to read it

The report shows, **for each denomination**: number of photos, correct, **wrong denomination**, not sure, accuracy over all photos, accuracy among announced, and coverage; then a confusion matrix (true note × what the app said, with `NOT_SURE`), every photo on its own line, and the false announcements on `no_note` photos.

* **Wrong denomination** = the app announced the wrong amount. This is the dangerous one.
* **Not sure** = the app stayed silent / said "I'm not sure". Safe, but annoying if it is too frequent.
* **Coverage** = share of photos where it announced an amount.
* **Read all three of accuracy-over-all, accuracy-among-announced and coverage together** – accuracy among announced alone hides the silent ones.
* Photos listed under **"Excluded: possible duplicates"** matched an original scan too closely; I look at them by eye and fix the cause (a downloaded or screenshot picture is not a real photo).

**My own honest expectation** (so I am not surprised): on the synthetic images coverage was 75–88 %; on real photos I expect **more "not sure" (lower coverage)** because the model saw only 18 scans. A **wrong denomination, especially NLe 2 announced as Le 2000 (or the other new/old pairs)** is possible. Both are findings about data, not a reason to hide the result.

**Pass / fail rules I set before looking (my acceptance gates):**

| Gate | Pass |
|---|---|
| Wrong denominations | **0** among all banknote photos |
| False announcements on `no_note` photos | **0** |
| Coverage | at least 50 % to be practical (otherwise: usable but needs more data) |
| Every denomination has at least 10 photos | otherwise its row is marked "unreliable" |

**What the numbers can and cannot prove:** 0 wrong in *n* photos only shows the true error rate is probably below about **3 / n** (95 % confidence): n = 20 → below ~15 %, n = 60 → ~5 %, n = 100 → ~3 %. So 20 photos per note is a first check, not proof of safety.

## 8. Test on the phone (live, with the spoken output)

```bash
cd /Users/mac/Desktop/dev/mobile/visionAid
~/Library/Android/sdk/platform-tools/adb devices
npm start                                                      # terminal 1 (leave running)
```
```bash
# terminal 2
cd /Users/mac/Desktop/dev/mobile/visionAid
~/Library/Android/sdk/platform-tools/adb reverse tcp:8081 tcp:8081
npm run android
```
```bash
# terminal 3: save the log of this session to a file (create the folder once)
mkdir -p /Users/mac/Desktop/dev/mobile/visionAid/ml/currency/reports/device_logs
~/Library/Android/sdk/platform-tools/adb logcat -c
~/Library/Android/sdk/platform-tools/adb logcat -s VisionAidCurrency > /Users/mac/Desktop/dev/mobile/visionAid/ml/currency/reports/device_logs/$(date +%Y-%m-%d_%H%M).txt
```
On the phone: turn the **volume up**, tap **Currency mode**, hold one note close with the note filling most of the screen, wait 1–3 seconds. For each try, write one line in the table below (copy it into `ml/currency/reports/real_test_log.md` under a heading "Phone session <date>"):

| Date | Note (true) | Side | Light | Distance | What I HEARD | Correct? | Heard clearly? | Log line (`ANNOUNCE`/`UNSURE`, p=) |
|---|---|---|---|---|---|---|---|---|

Things to check: correct amount and the word **"New"/"Old"**; it **does not repeat** while I keep holding the note; it says **"I'm not sure…"** for a hand, paper or a card; after removing the note and showing another note it announces the new one; **Stop currency mode** returns to object detection; and – never yet confirmed – **I can really hear the voice** (volume, speaker, not in silent mode).
When finished: press Ctrl-C in the log terminal, then close everything:
```bash
lsof -ti tcp:8081 | xargs kill
~/Library/Android/sdk/platform-tools/adb reverse --remove-all
cd /Users/mac/Desktop/dev/mobile/visionAid/android && ./gradlew --stop
```

## 9. Optional extra: prove the phone gives the same answer as Python on my real photos

```bash
cd /Users/mac/Desktop/dev/mobile/visionAid/ml/currency/scripts
python evaluate_real_photos.py --dump-inputs        # also saves the exact 224x224 inputs in ml/currency/processed/real_inputs/
```
Copy a few of those PNGs to the phone and run the debug self-test exactly as written in section 14.4 of `VISIONAID_CURRENCY_RECOGNITION.md` (`adb push …` then `run-as com.visionaid` copy into `cache/currency_selftest`, tap Currency mode, `adb logcat -s VisionAidCurrency | grep SELFTEST`) and compare the probabilities with the Python lines.

## 10. After I have the results – what happens next

1. **Do not retrain yet.** First save the report (it is already archived), read it, and note the result in `real_test_log.md`.
2. If a gate fails, send the report to the assistant (or look at the "Every photo" table): find the pattern (one denomination? one side? dim light? the NLe 2 / Le 2000 pair?).
3. The fix is usually **more real photos of the failing cases** in the training data. **The current training scripts only use the original scans.** To train on real photos I need a small extension of `build_sets.py` (real photos → training, a separate held-out set → test). I ask the assistant to do that *after* the first report exists.
4. Re-test only on **new** real photos (Set B), so the before/after comparison is honest. Record both in `real_test_log.md`.
5. Do not change the threshold τ to make the numbers look better; τ may only be **raised** for safety, and any change is written in the notes with the reason.

## 11. Quick checklist (copy-paste when I come back)

- [ ] `cd /Users/mac/Desktop/dev/mobile/visionAid && git branch --show-current` → `feat/currency-recognition`
- [ ] `source ~/venvs/visionaid-ml/bin/activate && cd ml/currency/scripts && python evaluate_real_photos.py --selftest`
- [ ] take photos (section 2–3) → `adb pull` (section 4) → sort into folders
- [ ] `python import_photos.py <folder> <class_folder>` for every folder (section 5)
- [ ] `python evaluate_real_photos.py --list` then `python evaluate_real_photos.py`
- [ ] read `reports/real_photo_evaluation.md`, check gates (section 7), look at `reports/real_test_log.md`
- [ ] phone test with sound on (section 8), fill the phone table
- [ ] back up `~/Desktop/currency_photos`, close ports, update my notes
