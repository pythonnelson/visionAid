package com.margelo.nitro.VisionAidObjectDetector

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CurrencyDeciderTest {

    private val labels = CurrencyLabels.fromJson(File("../../ml/currency/labels.json").readText())
    private val n = labels.size

    /** softmax-like vector with [p] on [idx] and the rest spread over the others. */
    private fun probs(idx: Int, p: Float): FloatArray {
        val rest = (1f - p) / (n - 1); return FloatArray(n) { if (it == idx) p else rest }
    }
    private val noNote get() = probs(9, 0.97f)
    private fun decider() = CurrencyDecider(labels, threshold = 0.90f)
    private var t = 0L
    private fun CurrencyDecider.feed(p: FloatArray?): CurrencyEvent? { t += 250; return update(p, t) }

    @Test fun announcesOnlyAfterConsistentConfidentFrames() {
        val d = decider()
        repeat(3) { assertNull(d.feed(probs(2, 0.97f))) }
        val e = d.feed(probs(2, 0.97f)) as CurrencyEvent.Announce
        assertEquals("NEW_5", e.cls.id); assertEquals("New Leone, five", e.text)
    }

    @Test fun belowThresholdNeverAnnounces() {
        val d = decider()
        repeat(30) { assertTrue(d.feed(probs(2, 0.85f)) !is CurrencyEvent.Announce) }
    }

    @Test fun noNoteAndInvalidOutputNeverAnnounce() {
        val d = decider()
        repeat(20) { assertNull(d.feed(noNote)) }
        repeat(20) { assertNull(d.feed(null)) }
        repeat(20) { assertNull(d.feed(FloatArray(n) { Float.NaN })) }
        repeat(20) { assertNull(d.feed(FloatArray(3))) }               // wrong size
        repeat(20) { assertNull(d.feed(FloatArray(n) { 5f })) }          // not a probability vector
    }

    @Test fun sameDenominationIsNotRepeated() {
        val d = decider()
        repeat(4) { d.feed(probs(7, 0.98f)) }
        repeat(40) { assertNull(d.feed(probs(7, 0.98f))) }
    }

    @Test fun announcedAgainOnlyAfterNoteWasRemoved() {
        val d = decider()
        repeat(4) { d.feed(probs(7, 0.98f)) }
        repeat(7) { d.feed(noNote) }
        var again: CurrencyEvent? = null
        repeat(4) { again = d.feed(probs(7, 0.98f)) ?: again }
        assertEquals("OLD_5000", (again as CurrencyEvent.Announce).cls.id)
    }

    @Test fun differentNoteIsAnnouncedAndKeepsOldNewApart() {
        val d = decider()
        repeat(4) { d.feed(probs(2, 0.98f)) }                            // NEW_5
        var e: CurrencyEvent? = null
        repeat(8) { e = d.feed(probs(7, 0.98f)) ?: e }                  // OLD_5000
        assertEquals("Old Leone, five thousand", (e as CurrencyEvent.Announce).text)
    }

    @Test fun conflictingPredictionsAreNotAnnouncedAndEventuallyUnsure() {
        val d = decider()
        var unsure: CurrencyEvent? = null
        for (i in 0 until 40) {
            val ev = d.feed(if (i % 2 == 0) probs(2, 0.95f) else probs(7, 0.95f))
            assertTrue(ev !is CurrencyEvent.Announce)
            if (ev is CurrencyEvent.Unsure) unsure = ev
        }
        assertEquals("I'm not sure. Please hold the note steady and closer.", (unsure as CurrencyEvent.Unsure).text)
    }

    @Test fun unsureIsRateLimited() {
        val d = decider(); var count = 0
        for (i in 0 until 60) if (d.feed(if (i % 2 == 0) probs(2, 0.95f) else probs(7, 0.95f)) is CurrencyEvent.Unsure) count++
        assertTrue("at most 2 'not sure' in 15 s of frames, got $count", count in 1..2)
    }

    @Test fun lowConfidenceNoteLikeFramesGiveUnsureNotAnAmount() {
        val d = decider(); var sawUnsure = false
        repeat(40) { val e = d.feed(probs(3, 0.60f)); assertTrue(e !is CurrencyEvent.Announce); if (e is CurrencyEvent.Unsure) sawUnsure = true }
        assertTrue(sawUnsure)
    }

    @Test fun resetForgetsPreviousResult() {
        val d = decider()
        repeat(4) { d.feed(probs(2, 0.98f)) }
        d.reset()
        var e: CurrencyEvent? = null
        repeat(4) { e = d.feed(probs(2, 0.98f)) ?: e }
        assertTrue(e is CurrencyEvent.Announce)
    }
}
