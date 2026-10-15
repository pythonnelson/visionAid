package com.margelo.nitro.VisionAidObjectDetector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DetectionStabilizerTest {

    private fun det(id: Int, score: Float = 0.6f) =
        Detection(id, "c$id", score, 0f, 0f, 1f, 1f)

    private val none = emptyList<Detection>()

    @Test fun becomesStableAfterThreeHits() {
        val s = DetectionStabilizer()
        assertTrue(s.update(listOf(det(61))).stable.isEmpty())
        assertTrue(s.update(listOf(det(61))).stable.isEmpty())
        val u = s.update(listOf(det(61)))
        assertEquals(listOf(61), u.stable.map { it.classId })
        assertEquals(listOf(61), u.appeared.map { it.classId })
    }

    @Test fun confidentObjectIsFast() {
        val s = DetectionStabilizer()
        s.update(listOf(det(0, 0.9f)))
        assertEquals(listOf(0), s.update(listOf(det(0, 0.85f))).appeared.map { it.classId })
    }

    @Test fun flickeringLabelsNeverBecomeStable() {
        val s = DetectionStabilizer()
        val labels = listOf(71, 83, 66, 61, 72, 76, 51, 75)   // a different label every frame
        for (l in labels) assertTrue(s.update(listOf(det(l))).stable.isEmpty())
    }

    @Test fun announcedOnlyOnce() {
        val s = DetectionStabilizer()
        repeat(3) { s.update(listOf(det(61))) }
        repeat(5) { assertTrue(s.update(listOf(det(61))).appeared.isEmpty()) }
    }

    @Test fun disappearingObjectIsDropped() {
        val s = DetectionStabilizer()
        repeat(3) { s.update(listOf(det(61))) }
        assertEquals(1, s.update(none).stable.size)
        assertEquals(1, s.update(none).stable.size)
        val u = s.update(none)
        assertTrue(u.stable.isEmpty())
        assertEquals(listOf(61), u.disappeared)
    }

    @Test fun briefGapDoesNotDropObject() {
        val s = DetectionStabilizer()
        repeat(3) { s.update(listOf(det(61))) }
        s.update(none)
        assertEquals(1, s.update(listOf(det(61))).stable.size)
    }

    @Test fun sceneChangeReplacesLabels() {
        val s = DetectionStabilizer()
        repeat(3) { s.update(listOf(det(61))) }              // chair
        var last = s.update(none)
        repeat(4) { last = s.update(listOf(det(39))) }       // bottle
        assertEquals(listOf(39), last.stable.map { it.classId })
    }

    @Test fun reappearingObjectIsAnnouncedAgain() {
        val s = DetectionStabilizer()
        repeat(3) { s.update(listOf(det(61))) }
        repeat(3) { s.update(none) }
        val appeared = (1..3).map { s.update(listOf(det(61))) }.flatMap { it.appeared }
        assertEquals(1, appeared.size)
    }

    @Test fun weakDetectionsCannotStartATrack() {
        val s = DetectionStabilizer()
        repeat(6) { assertTrue(s.update(listOf(det(61, 0.40f))).stable.isEmpty()) }
    }

    @Test fun weakDetectionsKeepAStableTrackAlive() {
        val s = DetectionStabilizer()
        repeat(3) { s.update(listOf(det(61, 0.6f))) }
        repeat(10) { assertEquals(1, s.update(listOf(det(61, 0.38f))).stable.size) }
    }
}
