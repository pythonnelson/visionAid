package com.margelo.nitro.VisionAidObjectDetector

/** What the app should do after looking at one more frame. */
sealed class CurrencyEvent {
    /** A denomination passed all safety checks: speak [text] once. */
    data class Announce(val cls: CurrencyClass, val text: String, val confidence: Float) : CurrencyEvent()
    /** The user keeps showing something note-like but predictions conflict / are weak. */
    data class Unsure(val text: String) : CurrencyEvent()
}

/**
 * Safety layer between the model and the speaker. Pure logic (no Android), fully unit tested.
 *
 * A single frame is "accepted" only when the model's top class is a banknote AND its probability >= [threshold]
 * (threshold calibrated on validation data, see ml/currency/reports/thresholds.json).
 * A denomination is announced only when
 *   - at least [needed] of the last [window] frames were accepted for the SAME class, and
 *   - no other class was accepted in that window more than [maxConflicts] times.
 * After an announcement the same class is not announced again until the note is gone for
 * [clearAfterEmpty] consecutive frames (no accepted note), or a DIFFERENT class is confirmed.
 * If the user keeps presenting something that never becomes a confirmed answer, "I'm not sure" is said
 * at most once per [unsureCooldownMs].
 *
 * It NEVER claims authenticity, legal-tender status or a sum; it only names a classified denomination.
 */
class CurrencyDecider(
    private val labels: CurrencyLabels,
    private val threshold: Float,
    private val window: Int = 6,
    private val needed: Int = 4,
    private val maxConflicts: Int = 1,
    private val clearAfterEmpty: Int = 6,
    private val unsureAfterFrames: Int = 14,
    private val unsureCooldownMs: Long = 8000,
    private val unsureText: String = "I'm not sure. Please hold the note steady and closer."
) {
    private val recent = ArrayDeque<Int>()          // accepted class index per frame, -1 = none
    private var announced = -1
    private var emptyStreak = 0
    private var unsureStreak = 0
    private var lastUnsureAt: Long? = null

    /** Class currently confirmed and announced (or -1). Used for the on-screen text. */
    val currentIndex: Int get() = announced

    fun reset() { recent.clear(); announced = -1; emptyStreak = 0; unsureStreak = 0 }

    /**
     * @param probabilities model output (softmax) in labels.json order; invalid output -> treated as "no note".
     */
    fun update(probabilities: FloatArray?, nowMs: Long): CurrencyEvent? {
        val accepted = acceptedClass(probabilities)
        recent.addLast(accepted); while (recent.size > window) recent.removeFirst()

        if (accepted == -1) { emptyStreak++ } else { emptyStreak = 0 }
        val noteLike = probabilities != null && validOutput(probabilities) &&
            probabilities.argmax() != labels.noNoteIndex && probabilities.max() >= 0.40f
        if (emptyStreak >= clearAfterEmpty) announced = -1
        if (!noteLike) unsureStreak = 0

        val counts = recent.filter { it >= 0 }.groupingBy { it }.eachCount()
        val best = counts.maxByOrNull { it.value }
        if (best != null && best.value >= needed) {
            val conflicts = counts.filterKeys { it != best.key }.values.sum()
            if (conflicts <= maxConflicts) {
                unsureStreak = 0
                if (best.key != announced) {
                    announced = best.key
                    val cls = labels.classAt(best.key)!!
                    return CurrencyEvent.Announce(cls, cls.spokenEn, probabilities!![best.key])
                }
                return null
            }
        }
        // Something note-like is in view but no confirmed answer yet.
        if (noteLike) {
            unsureStreak++
            if (unsureStreak >= unsureAfterFrames && (lastUnsureAt == null || nowMs - lastUnsureAt!! >= unsureCooldownMs)) {
                lastUnsureAt = nowMs; unsureStreak = 0
                announced = -1                                  // an earlier answer must not stay "valid"
                return CurrencyEvent.Unsure(unsureText)
            }
        }
        return null
    }

    private fun acceptedClass(p: FloatArray?): Int {
        if (p == null || !validOutput(p)) return -1
        val top = p.argmax()
        val cls = labels.classAt(top) ?: return -1
        return if (cls.isNote && p[top] >= threshold) top else -1
    }

    private fun validOutput(p: FloatArray): Boolean =
        p.size == labels.size && p.all { it.isFinite() && it >= -1e-4f && it <= 1f + 1e-4f } &&
            kotlin.math.abs(p.sum() - 1f) < 0.05f

    private fun FloatArray.argmax(): Int { var b = 0; for (i in indices) if (this[i] > this[b]) b = i; return b }
}
