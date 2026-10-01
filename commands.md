find android/app/build/intermediates -type f \( -name "*NitroVisionaidobjectdetector*" -o -name "libNitroVisionaidobjectdetector.so" \) 2>/dev/null

<!-- Custom Model Implementation (TensorFlow) -->
# Prepare the model location (from the root of the project)
mkdir -p VisionAidObjectDetector/android/src/main/assets

# The verify if the folder exists (it should return nothing)
ls VisionAidObjectDetector/android/src/main/assets


# VisionAidObjectDetector/android/build.gradle
# Remove 
implementation "com.google.mlkit:object-detection-custom:17.0.2" 
# Replace with
implementation "org.tensorflow:tensorflow-lite:2.17.0"

# Remove the detector imports in the file
VisionAidObjectDetector/android/src/main/java/com/margelo/nitro/VisionAidObjectDetector/HybridVisionAidObjectDetector.kt

# imports to remove
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.custom.ObjectDetectorOptions

# Download the actual model (tensorFlow) -> from the root project (run this command)
curl -L \
'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/lite-model_efficientdet_lite0_detection_metadata_1.tflite' \
-o VisionAidObjectDetector/android/src/main/assets/efficientdet_lite0.tflite

# Verify the file if it's available
ls -lh VisionAidObjectDetector/android/src/main/assets/

# Add the COCO labels (from the root of the project)
curl -L \
'https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt' \
-o VisionAidObjectDetector/android/src/main/assets/coco_labels.txt

# Verify the file if it's available
ls -lh VisionAidObjectDetector/android/src/main/assets/

# Expected result
coco_labels.txt
efficientdet_lite0.tflite

# Replace the Kotlin detector
# open (VisionAidObjectDetector/android/src/main/java/com/margelo/nitro/VisionAidObjectDetector/HybridVisionAidObjectDetector.kt)

Replace everything in the old <HybridVisionAidObjectDetector.kt> file with the recent updated code

# Install the updated native module (from the root of the project)
npm install ./VisionAidObjectDetector

# Then
cd android
./gradlew clean
./gradlew app:assembleDebug

# If it's successful, then Install and launch the newly built APK (from the root of the project => cd ..)
adb install -r android/app/build/outputs/apk/debug/app-debug.apk

# Watch the detector logs (different terminals)
adb logcat -c
adb logcat | grep -E "VisionAidTFLite|FATAL EXCEPTION"
adb logcat -s VisionAidMLKit

# Watch for similar result
VisionAidTFLite: TFLite model loaded successfully
VisionAidTFLite: Input shape: [...]
VisionAidTFLite: Input type: [...]
VisionAidTFLite: Output[0] shape=...
VisionAidTFLite: Output[1] shape=...
VisionAidTFLite: Output[2] shape=...
VisionAidTFLite: Output[3] shape=...
