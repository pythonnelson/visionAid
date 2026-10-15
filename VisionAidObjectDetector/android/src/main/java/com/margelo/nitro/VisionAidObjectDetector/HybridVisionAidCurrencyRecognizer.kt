package com.margelo.nitro.VisionAidObjectDetector

import android.graphics.ImageFormat
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log

import com.margelo.nitro.NitroModules
import com.margelo.nitro.camera.HybridFrameSpec
import com.margelo.nitro.camera.public.NativeFrame

import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

/**
 * Sierra Leone banknote recogniser. Completely separate from the object detector:
 * own model (currency_classifier.tflite), labels, thresholds, interpreter and speech.
 *
 * Per frame: centre square of the upright frame -> 224x224 float32 RGB 0..255 -> classifier ->
 * [CurrencyDecider] (confidence + consistency + "I'm not sure") -> speech.
 * It only NAMES a classified denomination: it never says a note is genuine, valid or counts money.
 */
class HybridVisionAidCurrencyRecognizer : HybridVisionAidCurrencyRecognizerSpec() {

    companion object {
        private const val TAG = "VisionAidCurrency"
        private const val MODEL_FILE = "currency_classifier.tflite"
        private const val LABEL_FILE = "currency_labels.json"
        private const val THRESHOLD_FILE = "currency_thresholds.json"
        private const val INPUT_SIZE = 224
        /** Currency mode analyses at most ~4 frames per second to save battery. */
        private const val MIN_INTERVAL_MS = 250L
        private const val HINT = "Looking for a banknote. Hold one note close to the camera."
        private const val LOG_INTERVAL_MS = 2000L
        /** Debug builds only: folder (inside the app cache) with exact 224x224 PNGs to run through the real on-device model. */
        private const val SELFTEST_DIR = "currency_selftest"
    }

    private var interpreter: Interpreter? = null
    private var labels: CurrencyLabels? = null
    private var decider: CurrencyDecider? = null
    private var loadError = ""
    private var threshold = 0f

    private val reader = YuvPlaneReader()
    private val inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4).order(ByteOrder.nativeOrder())
    private var output = Array(1) { FloatArray(0) }

    @Volatile private var active = false
    @Volatile private var status = ""
    private var lastAnalyzedAt = 0L
    private var lastLogAt = 0L
    private var lastErrorLogAt = 0L
    private var statusSetAt = 0L

    private var tts: TextToSpeech? = null
    @Volatile private var ttsReady = false

    init {
        try {
            val l = CurrencyLabels.fromJson(String(asset(LABEL_FILE), Charsets.UTF_8))
            val tau = JSONObject(String(asset(THRESHOLD_FILE), Charsets.UTF_8)).getDouble("accept_threshold").toFloat()
            require(tau in 0.5f..1.0f) { "accept_threshold $tau is outside 0.5..1.0" }
            val bytes = asset(MODEL_FILE)
            val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }
            val it = Interpreter(model, Interpreter.Options().apply { setNumThreads(2) })

            val inT = it.getInputTensor(0); val outT = it.getOutputTensor(0)
            check(inT.dataType() == DataType.FLOAT32 && inT.shape().contentEquals(intArrayOf(1, INPUT_SIZE, INPUT_SIZE, 3))) {
                "Unexpected currency input tensor ${inT.shape().contentToString()} ${inT.dataType()}"
            }
            check(outT.dataType() == DataType.FLOAT32 && outT.shape().contentEquals(intArrayOf(1, l.size))) {
                "Currency model output ${outT.shape().contentToString()} does not match ${l.size} labels"
            }
            output = Array(1) { FloatArray(l.size) }
            interpreter = it; labels = l; decider = CurrencyDecider(l, tau); threshold = tau
            Log.d(TAG, "Currency model loaded: ${l.size} classes, accept threshold $tau, input ${inT.shape().contentToString()} ${inT.dataType()}")
        } catch (e: Throwable) {
            loadError = e.message ?: e.javaClass.simpleName
            Log.e(TAG, "Currency recognition unavailable", e)
        }
    }

    @Synchronized
    override fun setActive(active: Boolean) {
        if (active && loadError.isNotEmpty()) { this.active = false; return }
        this.active = active
        decider?.reset()
        if (active) {
            ensureSpeech(); setStatus(HINT)
            if (BuildConfig.DEBUG) runSelfTest()
        } else {
            tts?.stop(); setStatus("")
        }
        Log.d(TAG, "Currency mode ${if (active) "ON" else "OFF"}")
    }

    override fun isActive(): Boolean = active

    override fun getStatus(): String = if (loadError.isNotEmpty()) "Currency recognition is unavailable." else status

    override fun getError(): String = loadError

    @Synchronized
    override fun analyze(frame: HybridFrameSpec) {
        val it = interpreter ?: return
        val d = decider ?: return
        if (!active) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastAnalyzedAt < MIN_INTERVAL_MS) return
        lastAnalyzedAt = now
        try {
            val proxy = (frame as? NativeFrame)?.image ?: throw IllegalStateException("Frame is not a VisionCamera NativeFrame")
            val image = proxy.image ?: throw IllegalStateException("Frame has no android.media.Image")
            check(image.format == ImageFormat.YUV_420_888) { "Expected YUV_420_888, got ${image.format}" }
            val rotation = ((proxy.imageInfo.rotationDegrees % 360) + 360) % 360

            val t0 = SystemClock.elapsedRealtime()
            YuvPreprocessor.toFloatRgbSquare(reader.read(image, proxy.cropRect), rotation, frame.isMirrored, INPUT_SIZE, inputBuffer)
            it.run(inputBuffer, output)
            val ms = SystemClock.elapsedRealtime() - t0

            val probs = output[0]
            when (val ev = d.update(probs, now)) {
                is CurrencyEvent.Announce -> { setStatus(ev.text); speak(ev.text); Log.i(TAG, "ANNOUNCE ${ev.cls.id} p=${"%.2f".format(ev.confidence)}") }
                is CurrencyEvent.Unsure -> { setStatus(ev.text); speak(ev.text); Log.i(TAG, "UNSURE") }
                null -> if (d.currentIndex == -1 && status != HINT && now - statusSetAt > 4000) setStatus(HINT)
            }
            if (now - lastLogAt > LOG_INTERVAL_MS) {
                lastLogAt = now
                val top = probs.indices.maxByOrNull { i -> probs[i] } ?: 0
                Log.d(TAG, "top=${labels?.classAt(top)?.id} p=${"%.2f".format(probs[top])} ${ms}ms stable=${d.currentIndex.let { i -> labels?.classAt(i)?.id }}")
            }
        } catch (e: Exception) {
            if (now - lastErrorLogAt > LOG_INTERVAL_MS) { lastErrorLogAt = now; Log.e(TAG, "Currency analysis failed for a frame", e) }
        }
    }

    private fun setStatus(text: String) { status = text; statusSetAt = SystemClock.elapsedRealtime() }

    private fun ensureSpeech() {
        if (tts != null) return
        val ctx = NitroModules.applicationContext ?: run { Log.w(TAG, "No application context for speech"); return }
        tts = TextToSpeech(ctx) { code ->
            ttsReady = code == TextToSpeech.SUCCESS
            if (ttsReady) tts?.language = Locale.ENGLISH else Log.e(TAG, "Text-to-speech init failed: $code")
        }
        // Proves the engine really started/finished producing audio (a "ready" flag alone does not).
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) { Log.i(TAG, "TTS onStart: $utteranceId") }
            override fun onDone(utteranceId: String?) { Log.i(TAG, "TTS onDone: $utteranceId") }
            @Deprecated("deprecated in Java") override fun onError(utteranceId: String?) { Log.e(TAG, "TTS onError: $utteranceId") }
            override fun onError(utteranceId: String?, errorCode: Int) { Log.e(TAG, "TTS onError: $utteranceId code=$errorCode") }
        })
    }

    private fun speak(text: String) {
        if (text.isBlank()) return
        if (!ttsReady) { Log.w(TAG, "Speech not ready; would say: $text"); return }
        val r = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "currency-${SystemClock.elapsedRealtime()}")
        Log.i(TAG, "TTS speak(\"$text\") returned ${if (r == TextToSpeech.SUCCESS) "SUCCESS" else r.toString()}")
    }

    /**
     * DEBUG BUILDS ONLY. Runs every PNG/JPG in <cache>/currency_selftest through the deployed model exactly as the
     * camera path would after preprocessing (pixels used as they are, float R,G,B 0..255), logs the top-3 probabilities
     * and the decision, and speaks the first announced result through the real text-to-speech engine.
     * Used to compare the device against the Python evaluation on identical bytes. Not used in release builds.
     */
    private fun runSelfTest() {
        val dir = File(NitroModules.applicationContext?.cacheDir ?: return, SELFTEST_DIR)
        val files = dir.listFiles { f -> f.extension.lowercase() in setOf("png", "jpg", "jpeg") }?.sortedBy { it.name } ?: return
        val it = interpreter ?: return
        val l = labels ?: return
        Log.i(TAG, "SELFTEST start: ${files.size} files, tau=$threshold")
        var firstSpoken: CurrencyEvent.Announce? = null
        for (f in files) {
            val bmp = BitmapFactory.decodeFile(f.path) ?: run { Log.e(TAG, "SELFTEST cannot decode ${f.name}"); null } ?: continue
            if (bmp.width != INPUT_SIZE || bmp.height != INPUT_SIZE) { Log.e(TAG, "SELFTEST ${f.name} must be 224x224"); continue }
            val px = IntArray(INPUT_SIZE * INPUT_SIZE); bmp.getPixels(px, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
            inputBuffer.clear()
            for (p in px) { inputBuffer.putFloat(((p shr 16) and 0xFF).toFloat()); inputBuffer.putFloat(((p shr 8) and 0xFF).toFloat()); inputBuffer.putFloat((p and 0xFF).toFloat()) }
            inputBuffer.rewind()
            it.run(inputBuffer, output)
            val pr = output[0]
            val order = pr.indices.sortedByDescending { i -> pr[i] }.take(3)
            val top = order.joinToString(" ") { i -> "${l.classAt(i)?.id}=${"%.4f".format(pr[i])}" }
            val d = CurrencyDecider(l, threshold)
            var event: CurrencyEvent? = null
            repeat(6) { n -> d.update(pr.copyOf(), (n + 1) * 1000L)?.let { e -> event = e } }   // 6 identical frames, like a steady note
            val said = (event as? CurrencyEvent.Announce)?.cls?.id ?: "NOT_SURE"
            Log.i(TAG, "SELFTEST ${f.name} | $top | frame-rule says: $said")
            val ev = event
            if (firstSpoken == null && ev is CurrencyEvent.Announce) firstSpoken = ev
        }
        // text-to-speech initialises asynchronously: wait (debug only) until it is ready, then speak the first announced result
        val spoken = firstSpoken
        if (spoken != null) {
            var waited = 0
            while (!ttsReady && waited < 5000) { Thread.sleep(100); waited += 100 }
            Log.i(TAG, "SELFTEST speaking \"${spoken.text}\" (tts ready=$ttsReady after ${waited} ms)")
            setStatus(spoken.text); speak(spoken.text)
        }
        Log.i(TAG, "SELFTEST done")
    }

    private fun asset(name: String): ByteArray =
        (javaClass.classLoader!!.getResourceAsStream("assets/$name") ?: throw IllegalStateException("Could not find asset: $name"))
            .use { it.readBytes() }
}
