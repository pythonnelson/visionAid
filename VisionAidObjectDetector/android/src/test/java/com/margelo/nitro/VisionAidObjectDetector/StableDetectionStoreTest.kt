package com.margelo.nitro.VisionAidObjectDetector

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StableDetectionStoreTest {

    private val labels = LabelMap.fromLines(File("src/main/assets/coco_labels.txt").readLines().filter { it.isNotBlank() })
    private fun det(id: Int, score: Float = 0.9f) = Detection(id, labels.labelFor(id)!!, score, 0.1f, 0.2f, 0.5f, 0.6f)
    private fun upd(stable: List<Detection> = emptyList(), appeared: List<Detection> = emptyList(), gone: List<Int> = emptyList()) =
        DetectionStabilizer.Update(stable, appeared, gone)

    @Test fun startsEmpty() {
        val s = StableDetectionStore()
        assertTrue(s.snapshot().detections.isEmpty()); assertTrue(s.drain().isEmpty())
    }

    @Test fun snapshotHoldsLatestStableSetAndFrameSize() {
        val s = StableDetectionStore()
        s.update(upd(stable = listOf(det(61))), 720, 1280, 1000L, labels)
        val snap = s.snapshot()
        assertEquals(listOf("chair"), snap.detections.map { it.label })
        assertEquals(720, snap.frameWidth); assertEquals(1280, snap.frameHeight); assertEquals(1000L, snap.atMs)
        s.update(upd(), 720, 1280, 2000L, labels)
        assertTrue(s.snapshot().detections.isEmpty())
    }

    @Test fun appearedAndDisappearedEventsAreQueuedInOrderAndDrainedOnce() {
        val s = StableDetectionStore()
        s.update(upd(stable = listOf(det(0)), appeared = listOf(det(0))), 720, 1280, 1L, labels)
        s.update(upd(gone = listOf(0)), 720, 1280, 2L, labels)
        val ev = s.drain()
        assertEquals(listOf(true, false), ev.map { it.appeared })
        assertEquals(listOf("person", "person"), ev.map { it.detection.label })
        assertTrue(s.drain().isEmpty())
    }

    @Test fun disappearedEventKeepsLastKnownBox() {
        val s = StableDetectionStore()
        s.update(upd(stable = listOf(det(61))), 720, 1280, 1L, labels)
        s.update(upd(gone = listOf(61)), 720, 1280, 2L, labels)
        val e = s.drain().single()
        assertFalse(e.appeared); assertEquals(0.5f, e.detection.right, 0f)
    }

    @Test fun queueIsBoundedAndKeepsNewestEvents() {
        val s = StableDetectionStore(maxEvents = 3)
        for (id in listOf(0, 2, 43, 61, 72)) s.update(upd(appeared = listOf(det(id))), 1, 1, 1L, labels)
        assertEquals(listOf("bottle", "chair", "laptop"), s.drain().map { it.detection.label })
    }

    @Test fun clearForgetsEventsAndDetections() {
        val s = StableDetectionStore()
        s.update(upd(stable = listOf(det(0)), appeared = listOf(det(0))), 720, 1280, 5L, labels)
        s.clear()
        assertTrue(s.drain().isEmpty()); assertTrue(s.snapshot().detections.isEmpty())
    }
}
