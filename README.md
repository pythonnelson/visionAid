# VisionAid – my project notes

VisionAid is an assistive app I am building for blind and low-vision users, especially in Sierra Leone. It is a React Native (0.87.1) app with VisionCamera 5.2.3 and native Kotlin/TensorFlow Lite code (through a Nitro module) on Android. These notes are for me, so I remember how everything fits and how to run it.

## What it does so far

- **Object detection** – EfficientDet-Lite0 finds everyday objects; stable detections are drawn on screen and spoken ("Person ahead") with a Voice on/off button. Notes: `VISIONAID_OBJECT_DETECTION_IMPLEMENTATION.md`.
- **Currency mode** – a separate model that recognises Sierra Leone banknotes (new and old Leone) and says the amount. Not tested with real notes yet. Notes: `VISIONAID_CURRENCY_RECOGNITION.md`.
- Later: Krio spoken guidance, obstacle awareness.
- **Coming back to currency later?** Start with `ml/currency/REAL_PHOTO_TESTING_GUIDE.md` (how to finish the real-note test). Presenting it? `VISIONAID_CURRENCY_PRESENTATION_NOTES.md`.

## Where things are

| Path | What is in it |
|---|---|
| `App.tsx` | camera screen, "Currency mode" button, status panel |
| `frameRouter.ts` | sends each frame to the currency recogniser (mode on) or the object detector (mode off) |
| `VisionAidObjectDetector/` | the Nitro native module (Kotlin, models, tests) – see its own README |
| `ml/currency/` | currency dataset, training/evaluation/export scripts – see its README |
| `android/` | the Android app project |
| `commands.md` | my older command notes |

## Run it (what I do every time)

Phone connected with USB debugging on, then from the repo root:

```bash
~/Library/Android/sdk/platform-tools/adb devices          # my phone must show as "device"
npm start                                                  # terminal 1: Metro (port 8081), leave it running
```
```bash
# terminal 2
~/Library/Android/sdk/platform-tools/adb reverse tcp:8081 tcp:8081
npm run android                                            # builds, installs, opens the app
```
```bash
# terminal 3: logs
~/Library/Android/sdk/platform-tools/adb logcat -s VisionAidTFLite VisionAidCurrency
```

Build only: `cd android && ./gradlew app:assembleDebug`

## Tests

```bash
cd android && ./gradlew :VisionAidObjectDetector:testDebugUnitTest   # Kotlin unit tests (53 pass)
npx jest __tests__/frameRouter.test.ts __tests__/objectAnnouncer.test.ts   # JS tests (15 pass)
```
`__tests__/App.test.tsx` fails already (Jest cannot read the vision-camera ES modules) – not caused by my changes, still to fix.

## Close everything when I'm done

```bash
lsof -ti tcp:8081 | xargs kill                      # stops Metro
~/Library/Android/sdk/platform-tools/adb reverse --remove-all
cd android && ./gradlew --stop                      # stops Gradle daemons
```

## If something goes wrong

- App cannot load the JavaScript: run `adb reverse tcp:8081 tcp:8081` again and check `npm start` is running.
- Phone not listed: unplug/replug, `adb devices` again.
- Build errors: `cd android && ./gradlew clean`, then build again (Java 17).
- I changed a `.nitro.ts` spec: `cd VisionAidObjectDetector && npm run specs`, then rebuild.

## Setup facts I should not forget

Node 22.20.0, npm 11.19.0, Java/OpenJDK 17, Android SDK at `/Users/mac/Library/Android/sdk`, minSdk 24, test phone Samsung A72 (SM-A725F, Android 14).
