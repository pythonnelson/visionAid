# ml/currency – my notes on the Sierra Leone banknote classifier (training side)

Everything in this folder is **training / evaluation tooling**. None of it is packaged into the app. The app only uses the exported files in `VisionAidObjectDetector/android/src/main/assets/` (`currency_classifier.tflite`, `currency_labels.json`, `currency_thresholds.json`).

```
dataset/<folder>/        MY ORIGINAL IMAGES (no script ever changes them): sle_1 sle_2 sle_5 sle_10 sle_20 (new Leone),
                         sll_1000 sll_2000 sll_5000 sll_10000 (old Leone), sle_all_currencies (collage), no_note
test/<folder>/           held-out images (synthetic_*.png are generated; my own phone photos go here as real_*.jpg)
negatives_ids.txt        COCO image ids used as "not a banknote" photos (scripts/fetch_negatives.py downloads them)
labels.json              class order + spoken labels = SINGLE SOURCE OF TRUTH
scripts/                 audit_dataset.py prepare_dataset.py build_sets.py train.py evaluate.py export_tflite.py ...
reports/                 dataset_audit.md  evaluation_tflite.md  evaluation.md  safety_audit.md  thresholds.json  model_export.json
                         (real_photo_evaluation.md appears only after I run evaluate_real_photos.py on my real_*.jpg photos)
processed/ runs/ negatives_source/   generated, git-ignored
```

Scripts for the audit: `safety_audit.py` (deployed-file checks, calibration, look-alike pairs) and `evaluate_real_photos.py` (my real phone photos only; refuses to report when there are none).

**To finish the real-note test later: `REAL_PHOTO_TESTING_GUIDE.md` (this folder). For my presentation: `VISIONAID_CURRENCY_PRESENTATION_NOTES.md` (repo root).** Photo import helper: `scripts/import_photos.py`.

Reminder: `no_note/` currently holds old Le 500 / Le 100 notes, not blank negatives (details in the main notes).
Full notes and commands: `VISIONAID_CURRENCY_RECOGNITION.md` in the repository root.
