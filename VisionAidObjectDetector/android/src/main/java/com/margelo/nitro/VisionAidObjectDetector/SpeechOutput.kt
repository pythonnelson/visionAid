package com.margelo.nitro.VisionAidObjectDetector

import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.margelo.nitro.NitroModules
import java.util.Locale

/**
 * Thin wrapper around Android TextToSpeech used by JavaScript (`speak(text)`).
 * JavaScript decides WHAT to say (and in which language); this class only produces the audio.
 * Engine start/done/error are logged so "speech worked" can be proven in logcat, not just assumed.
 */
class SpeechOutput(private val tag: String, private val locale: Locale = Locale.ENGLISH) {
    private var tts: TextToSpeech? = null
    @Volatile private var ready = false
    /** Phrases requested before the engine finished starting; spoken as soon as it is ready (bounded, newest kept). */
    private val pending = ArrayDeque<String>()
    private val pendingMax = 3

    @Synchronized
    fun ensureStarted() {
        if (tts != null) return
        val ctx = NitroModules.applicationContext ?: run { Log.w(tag, "No application context for speech"); return }
        val engine = TextToSpeech(ctx) { code ->
            ready = code == TextToSpeech.SUCCESS
            if (ready) {
                tts?.language = locale
                val queued = synchronized(pending) { val l = pending.toList(); pending.clear(); l }
                for (t in queued) speakNow(t, false)
            } else Log.e(tag, "Text-to-speech init failed: $code")
        }
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) { Log.i(tag, "TTS onStart: $utteranceId") }
            override fun onDone(utteranceId: String?) { Log.i(tag, "TTS onDone: $utteranceId") }
            @Deprecated("deprecated in Java") override fun onError(utteranceId: String?) { Log.e(tag, "TTS onError: $utteranceId") }
            override fun onError(utteranceId: String?, errorCode: Int) { Log.e(tag, "TTS onError: $utteranceId code=$errorCode") }
        })
        tts = engine
    }

    /** @param interrupt true = cut off whatever is being spoken. Returns false if nothing could be spoken. */
    fun speak(text: String, interrupt: Boolean = false): Boolean {
        if (text.isBlank()) return false
        ensureStarted()
        if (!ready) {
            synchronized(pending) { pending.addLast(text); while (pending.size > pendingMax) pending.removeFirst() }
            Log.w(tag, "Speech engine still starting; queued: $text")
            return false
        }
        return speakNow(text, interrupt)
    }

    private fun speakNow(text: String, interrupt: Boolean): Boolean {
        val mode = if (interrupt) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        val r = tts?.speak(text, mode, null, "obj-${SystemClock.elapsedRealtime()}")
        Log.i(tag, "TTS speak(\"$text\") returned ${if (r == TextToSpeech.SUCCESS) "SUCCESS" else r.toString()}")
        return r == TextToSpeech.SUCCESS
    }

    fun stop() { synchronized(pending) { pending.clear() }; tts?.stop() }
}
