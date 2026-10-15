# Currency classifier evaluation

Model: `../runs/currency_classifier.tflite`
Accept threshold tau = 0.98 (chosen on validation only)

### VALIDATION (hard simulated captures; used to choose tau) (n=750)
- argmax accuracy: 0.907
- note samples n=540: announced correctly 0.752, announced WRONG denomination 0.000, said 'not sure' 0.248
- NO_NOTE samples n=210: falsely announced a note 0.000
### TEST-SIM: simulated phone captures, held-out backgrounds/seeds (same scans as training!) (n=700)
- argmax accuracy: 0.994
- note samples n=540: announced correctly 0.880, announced WRONG denomination 0.000, said 'not sure' 0.120
- NO_NOTE samples n=160: falsely announced a note 0.000
#### TEST-SIM: confusion matrix (rows = true, cols = predicted, argmax)
| true \ pred | NEW_1 | NEW_2 | NEW_5 | NEW_10 | NEW_20 | OLD_1000 | OLD_2000 | OLD_5000 | OLD_10000 | NO_NOTE | n | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NEW_1 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 0.98 | 1.00 | 0.99 |
| NEW_2 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 1.00 | 1.00 | 1.00 |
| NEW_5 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 0.98 | 1.00 | 0.99 |
| NEW_10 | 0 | 0 | 1 | 59 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 1.00 | 0.98 | 0.99 |
| NEW_20 | 0 | 0 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 60 | 1.00 | 1.00 | 1.00 |
| OLD_1000 | 0 | 0 | 0 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 60 | 1.00 | 1.00 | 1.00 |
| OLD_2000 | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 1 | 0 | 0 | 60 | 1.00 | 0.98 | 0.99 |
| OLD_5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 0 | 0 | 60 | 0.98 | 1.00 | 0.99 |
| OLD_10000 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 58 | 1 | 60 | 1.00 | 0.97 | 0.98 |
| NO_NOTE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 160 | 160 | 0.99 | 1.00 | 1.00 |
Confusions: NEW_10->NEW_5 x1, OLD_2000->OLD_5000 x1, OLD_10000->NEW_1 x1, OLD_10000->NO_NOTE x1
- front: argmax accuracy 0.985 (n=270)
- back: argmax accuracy 1.000 (n=270)
- NEW series: argmax accuracy 0.997 (n=300)
- OLD series: argmax accuracy 0.988 (n=240)

### TEST-COLLAGE: NEW notes from a different image source (collage), simulated capture, never trained on (n=200)
- argmax accuracy: 0.925
- note samples n=200: announced correctly 0.615, announced WRONG denomination 0.000, said 'not sure' 0.385
#### TEST-COLLAGE: confusion matrix (rows = true, cols = predicted, argmax)
| true \ pred | NEW_1 | NEW_2 | NEW_5 | NEW_10 | NEW_20 | OLD_1000 | OLD_2000 | OLD_5000 | OLD_10000 | NO_NOTE | n | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NEW_1 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40 | 1.00 | 1.00 | 1.00 |
| NEW_2 | 0 | 28 | 2 | 0 | 1 | 0 | 6 | 0 | 0 | 3 | 40 | 1.00 | 0.70 | 0.82 |
| NEW_5 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40 | 0.95 | 1.00 | 0.98 |
| NEW_10 | 0 | 0 | 0 | 39 | 0 | 0 | 0 | 0 | 0 | 1 | 40 | 1.00 | 0.97 | 0.99 |
| NEW_20 | 0 | 0 | 0 | 0 | 38 | 0 | 0 | 0 | 0 | 2 | 40 | 0.97 | 0.95 | 0.96 |
| OLD_1000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| OLD_2000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| OLD_5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| OLD_10000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| NO_NOTE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
Confusions: NEW_2->OLD_2000 x6, NEW_2->NO_NOTE x3, NEW_2->NEW_5 x2, NEW_20->NO_NOTE x2, NEW_2->NEW_20 x1, NEW_10->NO_NOTE x1
- DISTINCT source images per class (the n above are simulated variants of these): NEW_1=2, NEW_2=2, NEW_5=2, NEW_10=2, NEW_20=2 -> per-class results are NOT statistically reliable

### TEST-NEGATIVE: held-out real photos/surfaces without notes (n=300)
- argmax accuracy: 1.000
- NO_NOTE samples n=300: falsely announced a note 0.000

### TEST-UNSUPPORTED: old Le 500 (unsupported note) specimen not used in training (n=40)
- argmax accuracy: 0.900
- NO_NOTE samples n=40: falsely announced a note 0.000

