package com.margelo.nitro.VisionAidObjectDetector

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import android.os.SystemClock
import android.util.Log

import com.margelo.nitro.NitroModules
import com.margelo.nitro.camera.HybridFrameSpec
import com.margelo.nitro.camera.public.NativeFrame

import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Real-time object detector (EfficientDet-Lite0, TensorFlow Lite).
 *
 * Pipeline for every camera frame (see VISIONAID_OBJECT_DETECTION_IMPLEMENTATION.md):
 *   YUV_420_888 frame -> crop / rotate upright / resize to 320x320 / RGB bytes
 *   -> interpreter -> decode + validate outputs -> log detections
 *
 * The frame is only read while detect() runs; VisionCamera keeps ownership of it.
 */
class HybridVisionAidObjectDetector :
    HybridVisionAidObjectDetectorSpec() {

    companion object {
        private const val TAG = "VisionAidTFLite"
        private const val MODEL_FILE = "efficientdet_lite0.tflite"
        private const val LABEL_FILE = "coco_labels.txt"

        /** A detection needs this score to be reported / to start a stable label. */
        private const val CONFIDENCE_THRESHOLD = 0.60f

        /** An already stable label is kept alive by weaker detections down to this score. */
        private const val KEEP_THRESHOLD = 0.35f

        /** How many top detections to print in each log line. */
        private const val MAX_LOGGED_DETECTIONS = 5

        /** Summary log at most once per this many milliseconds. */
        private const val LOG_INTERVAL_MS = 2000L

        /**
         * Development only: periodic snapshots, see maybeSaveSnapshot(). Debug builds AND opt-in:
         * enable with  adb shell run-as com.visionaid touch cache/ENABLE_SNAPSHOTS
         * disable with adb shell run-as com.visionaid rm cache/ENABLE_SNAPSHOTS
         */
        private val SNAPSHOTS_ALLOWED = BuildConfig.DEBUG
        private const val SNAPSHOT_FLAG_FILE = "ENABLE_SNAPSHOTS"
        private const val SAVE_INTERVAL_MS = 4000L
        private const val SNAPSHOT_RING = 12
    }

    private val interpreter: Interpreter
    private val labels: LabelMap

    private val inputWidth: Int
    private val inputHeight: Int
    private val inputBuffer: ByteBuffer

    // Output buffers are allocated once and reused for every frame.
    private val maxDetections: Int
    private val outputBoxes: Array<Array<FloatArray>>
    private val outputClasses: Array<FloatArray>
    private val outputScores: Array<FloatArray>
    private val outputCount = FloatArray(1)
    private val outputs: Map<Int, Any>

    // Plane copies, reused while the frame size stays the same.
    private var yBytes = ByteArray(0)
    private var uBytes = ByteArray(0)
    private var vBytes = ByteArray(0)

    private var lastLogTime = 0L
    private var lastSaveTime = 0L
    private var lastErrorLogTime = 0L
    private var loggedFirstFrame = false
    private var snapshotCounter = 0
    private val stabilizer = DetectionStabilizer(enterScore = CONFIDENCE_THRESHOLD)
    private val store = StableDetectionStore()
    private val speech = SpeechOutput(TAG)

    init {
        interpreter = Interpreter(
            loadAsset(MODEL_FILE),
            Interpreter.Options().apply { setNumThreads(4) }
        )
        labels = LabelMap.fromLines(
            String(loadAssetBytes(LABEL_FILE), Charsets.UTF_8).lines().filter { it.isNotBlank() }
        )

        // ---- Verify the model really has the contract this code relies on ----
        val input = interpreter.getInputTensor(0)
        val inShape = input.shape()
        check(
            inShape.size == 4 && inShape[0] == 1 && inShape[3] == 3 &&
                input.dataType() == DataType.UINT8
        ) {
            "Unexpected input tensor: shape=${inShape.contentToString()} type=${input.dataType()} " +
                "(expected [1,H,W,3] UINT8 raw RGB 0..255)"
        }
        inputHeight = inShape[1]
        inputWidth = inShape[2]
        inputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3)
            .order(ByteOrder.nativeOrder())

        check(interpreter.outputTensorCount == 4) {
            "Expected 4 output tensors, model has ${interpreter.outputTensorCount}"
        }
        val boxShape = interpreter.getOutputTensor(0).shape()
        check(boxShape.size == 3 && boxShape[0] == 1 && boxShape[2] == 4) {
            "Output 0 must be boxes [1,N,4] but is ${boxShape.contentToString()}"
        }
        maxDetections = boxShape[1]
        for (i in 0..3) {
            check(interpreter.getOutputTensor(i).dataType() == DataType.FLOAT32) {
                "Output $i must be FLOAT32 but is ${interpreter.getOutputTensor(i).dataType()}"
            }
        }
        check(interpreter.getOutputTensor(1).shape().contentEquals(intArrayOf(1, maxDetections))) { "Output 1 must be classes [1,N]" }
        check(interpreter.getOutputTensor(2).shape().contentEquals(intArrayOf(1, maxDetections))) { "Output 2 must be scores [1,N]" }
        check(interpreter.getOutputTensor(3).shape().contentEquals(intArrayOf(1))) { "Output 3 must be count [1]" }

        outputBoxes = Array(1) { Array(maxDetections) { FloatArray(4) } }
        outputClasses = Array(1) { FloatArray(maxDetections) }
        outputScores = Array(1) { FloatArray(maxDetections) }
        outputs = hashMapOf(0 to outputBoxes, 1 to outputClasses, 2 to outputScores, 3 to outputCount)

        speech.ensureStarted()   // start text-to-speech now so the first spoken phrase is not lost
        Log.d(TAG, "Model $MODEL_FILE loaded, ${labels.size} labels")
        Log.d(
            TAG,
            "Input: shape=${inShape.contentToString()} type=${input.dataType()} " +
                "scale=${input.quantizationParams().scale} zeroPoint=${input.quantizationParams().zeroPoint} " +
                "(raw RGB 0..255 bytes are fed directly)"
        )
        for (i in 0 until interpreter.outputTensorCount) {
            val t = interpreter.getOutputTensor(i)
            Log.d(TAG, "Output[$i] name=${t.name()} shape=${t.shape().contentToString()} type=${t.dataType()}")
        }
    }

    /**
     * Called by VisionCamera for every frame. Runs synchronously on the frame thread;
     * VisionCamera is configured with dropFramesWhileBusy so slow inference simply
     * skips frames instead of building a backlog. @Synchronized protects the
     * interpreter and the shared buffers if two calls ever overlap.
     */
    @Synchronized
    override fun detect(frame: HybridFrameSpec) {
        try {
            val nativeFrame = frame as? NativeFrame
                ?: throw IllegalStateException("Frame is not a VisionCamera NativeFrame")
            val imageProxy = nativeFrame.image
            val image = imageProxy.image
                ?: throw IllegalStateException("Frame has no android.media.Image")
            check(image.format == ImageFormat.YUV_420_888) {
                "Expected YUV_420_888 but got format ${image.format}. Use pixelFormat: 'yuv'."
            }

            // Clockwise degrees needed to make the pixels upright (CameraX definition).
            val rotation = ((imageProxy.imageInfo.rotationDegrees % 360) + 360) % 360
            val mirrored = frame.isMirrored

            val t0 = SystemClock.elapsedRealtime()
            val yuv = copyPlanes(image, imageProxy.cropRect)
            YuvPreprocessor.toRgbBuffer(yuv, rotation, mirrored, inputWidth, inputHeight, inputBuffer)
            val t1 = SystemClock.elapsedRealtime()

            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
            val t2 = SystemClock.elapsedRealtime()

            val result = DetectionDecoder.decode(
                outputBoxes[0], outputClasses[0], outputScores[0], outputCount[0],
                labels, KEEP_THRESHOLD
            )

            val stability = stabilizer.update(result.detections) // enter 0.60, keep 0.35
            for (d in stability.appeared) Log.i(TAG, "STABLE appeared: $d")
            for (id in stability.disappeared) Log.i(TAG, "STABLE disappeared: class=$id (${labels.labelFor(id)})")
            store.update(
                stability,
                Orientation.uprightWidth(yuv.cropWidth, yuv.cropHeight, rotation),
                Orientation.uprightHeight(yuv.cropWidth, yuv.cropHeight, rotation),
                SystemClock.elapsedRealtime(), labels
            )

            logDiagnostics(frame, image, rotation, mirrored, result, stability, t1 - t0, t2 - t1)
            maybeSaveSnapshot(yuv, rotation, mirrored)
        } catch (error: Exception) {
            // Rate limited so a persistent problem does not flood the log.
            val now = SystemClock.elapsedRealtime()
            if (now - lastErrorLogTime > LOG_INTERVAL_MS) {
                lastErrorLogTime = now
                Log.e(TAG, "Detection failed for this frame", error)
            }
        }
    }

    // ---- API for JavaScript (see VisionAidObjectDetector.nitro.ts). These never touch the camera thread's buffers.

    override fun getStableDetections(): StableDetections {
        val snap = store.snapshot()
        val age = if (snap.atMs == 0L) -1.0 else (SystemClock.elapsedRealtime() - snap.atMs).toDouble()
        return StableDetections(
            snap.frameWidth.toDouble(), snap.frameHeight.toDouble(), age,
            snap.detections.map { it.toDetectedObject() }.toTypedArray()
        )
    }

    override fun drainEvents(): Array<DetectionEvent> =
        store.drain().map {
            DetectionEvent(if (it.appeared) DetectionEventType.APPEARED else DetectionEventType.DISAPPEARED, it.detection.toDetectedObject())
        }.toTypedArray()

    @Synchronized
    override fun reset() {
        store.clear()
        stabilizer.reset()
    }

    override fun speak(text: String, interrupt: Boolean) { speech.speak(text, interrupt) }

    override fun stopSpeaking() { speech.stop() }

    private fun Detection.toDetectedObject() = DetectedObject(
        label, classId.toDouble(), score.toDouble(), left.toDouble(), top.toDouble(), right.toDouble(), bottom.toDouble()
    )

    /** Copies the three YUV planes into reusable arrays (honouring row/pixel strides). */
    private fun copyPlanes(image: Image, crop: android.graphics.Rect): YuvFrame {
        val planes = image.planes
        require(planes.size == 3) { "YUV image must have 3 planes, has ${planes.size}" }
        val (yp, up, vp) = Triple(planes[0], planes[1], planes[2])
        require(up.rowStride == vp.rowStride && up.pixelStride == vp.pixelStride) {
            "U and V planes have different strides"
        }
        yBytes = readPlane(yp.buffer, yBytes)
        uBytes = readPlane(up.buffer, uBytes)
        vBytes = readPlane(vp.buffer, vBytes)

        val c = if (crop.isEmpty) android.graphics.Rect(0, 0, image.width, image.height) else crop
        return YuvFrame(
            yBytes, uBytes, vBytes,
            yp.rowStride, yp.pixelStride, up.rowStride, up.pixelStride,
            c.left, c.top, c.width(), c.height()
        )
    }

    private fun readPlane(buffer: ByteBuffer, reuse: ByteArray): ByteArray {
        val dup = buffer.duplicate() // do not disturb VisionCamera's buffer position
        dup.rewind()
        val array = if (reuse.size == dup.remaining()) reuse else ByteArray(dup.remaining())
        dup.get(array)
        return array
    }

    private fun logDiagnostics(
        frame: HybridFrameSpec, image: Image, rotation: Int, mirrored: Boolean,
        result: DecodeResult, stability: DetectionStabilizer.Update, preprocessMs: Long, inferenceMs: Long
    ) {
        val now = SystemClock.elapsedRealtime()
        if (!loggedFirstFrame) {
            loggedFirstFrame = true
            Log.d(
                TAG,
                "First frame: ${image.width}x${image.height} rotationDegrees=$rotation " +
                    "orientation=${frame.orientation} mirrored=$mirrored -> " +
                    "model input ${inputWidth}x$inputHeight"
            )
        }
        if (now - lastLogTime < LOG_INTERVAL_MS) return
        lastLogTime = now

        Log.d(
            TAG,
            "frame ${image.width}x${image.height} rot=$rotation mirrored=$mirrored | " +
                "preprocess=${preprocessMs}ms inference=${inferenceMs}ms | " +
                "raw slots=${result.rawCount} invalid=${result.invalidCount} " +
                "belowThreshold(<$KEEP_THRESHOLD)=${result.belowThresholdCount} " +
                "accepted(>=$KEEP_THRESHOLD)=${result.detections.size} strong(>=$CONFIDENCE_THRESHOLD)=${result.detections.count { it.score >= CONFIDENCE_THRESHOLD }}"
        )
        val strong = result.detections.filter { it.score >= CONFIDENCE_THRESHOLD }
        if (strong.isEmpty()) {
            Log.d(TAG, "No detections at or above $CONFIDENCE_THRESHOLD")
        }
        for (d in strong.take(MAX_LOGGED_DETECTIONS)) {
            Log.d(TAG, "DETECTED (this frame): $d")
        }
        Log.d(TAG, "STABLE now: ${stability.stable.joinToString { it.label }.ifEmpty { "none" }}")
    }

    /**
     * Debug builds only. Every few seconds saves a SNAPSHOT to cacheDir/snapshots so the
     * exact data of one inference can be examined offline:
     *   snap_N_input.png  the exact 320x320 bytes that were given to the interpreter
     *   snap_N_frame.png  the same upright camera frame at half resolution (360x640 etc.)
     *   snap_N.txt        the raw model output (first 10 slots, before any filtering)
     * Everything is written inside the synchronized detect() call, so the buffers
     * cannot change while the snapshot is written.
     */
    private fun maybeSaveSnapshot(yuv: YuvFrame, rotation: Int, mirrored: Boolean) {
        if (!SNAPSHOTS_ALLOWED) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastSaveTime < SAVE_INTERVAL_MS) return
        lastSaveTime = now
        try {
            val cache = NitroModules.applicationContext?.cacheDir ?: return
            if (!File(cache, SNAPSHOT_FLAG_FILE).exists()) return
            val dir = File(cache, "snapshots")
            dir.mkdirs()
            val n = snapshotCounter++ % SNAPSHOT_RING
            val view = inputBuffer.duplicate().order(ByteOrder.nativeOrder())
            savePng(File(dir, "snap_${n}_input.png"), rgbBytesToBitmap(view, inputWidth, inputHeight))

            val fw = Orientation.uprightWidth(yuv.cropWidth, yuv.cropHeight, rotation) / 2
            val fh = Orientation.uprightHeight(yuv.cropWidth, yuv.cropHeight, rotation) / 2
            val frameBuf = ByteBuffer.allocate(fw * fh * 3)
            YuvPreprocessor.toRgbBuffer(yuv, rotation, mirrored, fw, fh, frameBuf)
            savePng(File(dir, "snap_${n}_frame.png"), rgbBytesToBitmap(frameBuf, fw, fh))

            val lines = DetectionDecoder.rawSummary(
                outputBoxes[0], outputClasses[0], outputScores[0], outputCount[0], labels, 10
            )
            File(dir, "snap_$n.txt").writeText(
                "rotation=$rotation mirrored=$mirrored crop=${yuv.cropWidth}x${yuv.cropHeight}\n" +
                    lines.joinToString("\n") + "\n"
            )
        } catch (e: Exception) {
            Log.w(TAG, "Could not save debug snapshot", e)
        }
    }

    private fun rgbBytesToBitmap(buf: ByteBuffer, w: Int, h: Int): Bitmap {
        val pixels = IntArray(w * h)
        for (i in pixels.indices) {
            val r = buf.get(i * 3).toInt() and 0xFF
            val g = buf.get(i * 3 + 1).toInt() and 0xFF
            val b = buf.get(i * 3 + 2).toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun savePng(file: File, bitmap: Bitmap) {
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
    }

    private fun loadAssetBytes(fileName: String): ByteArray =
        (javaClass.classLoader!!.getResourceAsStream("assets/$fileName")
            ?: throw IllegalStateException("Could not find asset: $fileName"))
            .use { it.readBytes() }

    private fun loadAsset(fileName: String): ByteBuffer {
        val bytes = loadAssetBytes(fileName)
        return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
            put(bytes)
            rewind()
        }
    }
}
