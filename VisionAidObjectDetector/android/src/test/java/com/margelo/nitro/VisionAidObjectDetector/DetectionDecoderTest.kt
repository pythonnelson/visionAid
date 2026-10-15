package com.margelo.nitro.VisionAidObjectDetector

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DetectionDecoderTest {

    private val labels = LabelMap.fromLines(
        File("src/main/assets/coco_labels.txt").readLines().filter { it.isNotBlank() }
    )

    private fun decode(
        box: FloatArray, cls: Float, score: Float, count: Float = 1f, threshold: Float = 0.4f
    ) = DetectionDecoder.decode(arrayOf(box, FloatArray(4)), floatArrayOf(cls, 0f), floatArrayOf(score, 0f), count, labels, threshold)

    @Test fun labelFileHasNinetyEntriesMatchingModel() = assertEquals(90, labels.size)

    @Test fun labelsAreZeroBased() {
        assertEquals("person", labels.labelFor(0))
        assertEquals("car", labels.labelFor(2))
        assertEquals("bottle", labels.labelFor(43))
        assertEquals("chair", labels.labelFor(61))
        assertEquals("mouse", labels.labelFor(73))
        assertEquals("toothbrush", labels.labelFor(89))
    }

    @Test fun placeholderAndOutOfRangeClassesHaveNoLabel() {
        assertNull(labels.labelFor(11)) // n/a (street sign)
        assertNull(labels.labelFor(-1))
        assertNull(labels.labelFor(90))
    }

    @Test fun decodesBoxInYxOrder() {
        val r = decode(floatArrayOf(0.1f, 0.2f, 0.6f, 0.9f), 0f, 0.8f)
        val d = r.detections.single()
        assertEquals("person", d.label)
        assertEquals(0.2f, d.left, 1e-6f); assertEquals(0.1f, d.top, 1e-6f)
        assertEquals(0.9f, d.right, 1e-6f); assertEquals(0.6f, d.bottom, 1e-6f)
    }

    @Test fun clampsAndOrdersBox() {
        val d = decode(floatArrayOf(0.9f, 1.2f, -0.1f, 0.3f), 61f, 0.9f).detections.single()
        assertEquals(0f, d.top, 0f); assertEquals(0.9f, d.bottom, 1e-6f)
        assertEquals(0.3f, d.left, 1e-6f); assertEquals(1f, d.right, 0f)
    }

    @Test fun filtersByConfidenceAndCountsIt() {
        val r = decode(floatArrayOf(0.1f, 0.1f, 0.5f, 0.5f), 0f, 0.3f)
        assertTrue(r.detections.isEmpty()); assertEquals(1, r.belowThresholdCount)
    }

    @Test fun rejectsInvalidOutputs() {
        val ok = floatArrayOf(0.1f, 0.1f, 0.5f, 0.5f)
        assertEquals(1, decode(floatArrayOf(Float.NaN, 0f, 0.5f, 0.5f), 0f, 0.9f).invalidCount)
        assertEquals(1, decode(ok, 0f, Float.NaN).invalidCount)
        assertEquals(1, decode(ok, 0f, 1.5f).invalidCount)
        assertEquals(1, decode(ok, 11f, 0.9f).invalidCount)   // placeholder
        assertEquals(1, decode(ok, 200f, 0.9f).invalidCount)  // out of range
        assertEquals(1, decode(ok, 0.5f, 0.9f).invalidCount)  // not an integer
        assertEquals(1, decode(floatArrayOf(0.3f, 0.3f, 0.3f, 0.6f), 0f, 0.9f).invalidCount) // zero height
    }

    @Test fun countIsHonouredAndBounded() {
        assertEquals(0, decode(floatArrayOf(0.1f, 0.1f, 0.5f, 0.5f), 0f, 0.9f, count = 0f).detections.size)
        assertEquals(0, decode(floatArrayOf(0.1f, 0.1f, 0.5f, 0.5f), 0f, 0.9f, count = Float.NaN).rawCount)
        assertEquals(2, decode(floatArrayOf(0.1f, 0.1f, 0.5f, 0.5f), 0f, 0.9f, count = 99f).rawCount)
    }

    @Test fun sortsByScoreDescending() {
        val boxes = arrayOf(floatArrayOf(0f, 0f, .5f, .5f), floatArrayOf(.1f, .1f, .6f, .6f))
        val r = DetectionDecoder.decode(boxes, floatArrayOf(0f, 61f), floatArrayOf(0.5f, 0.9f), 2f, labels, 0.4f)
        assertEquals(listOf("chair", "person"), r.detections.map { it.label })
        assertNotNull(r.detections.first())
    }

    @Test fun rawSummaryShowsUnfilteredSlots() {
        val boxes = arrayOf(floatArrayOf(.1f, .2f, .3f, .4f), floatArrayOf(0f, 0f, 1f, 1f))
        val lines = DetectionDecoder.rawSummary(boxes, floatArrayOf(61f, 11f), floatArrayOf(0.05f, 0.9f), 2f, labels, 10)
        assertEquals(2, lines.size)
        assertTrue(lines[0].contains("label=chair") && lines[0].contains("score=0.05"))
        assertTrue(lines[1].contains("label=?"))
    }
}
