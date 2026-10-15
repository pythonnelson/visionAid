package com.margelo.nitro.VisionAidObjectDetector

/** The stable detections after the most recent processed frame. */
data class StableSnapshot(
    /** Size of the UPRIGHT frame the normalised boxes refer to (0 until the first frame was analysed). */
    val frameWidth: Int,
    val frameHeight: Int,
    /** Time the snapshot was produced (monotonic ms); lets the reader ignore a stale snapshot. */
    val atMs: Long,
    val detections: List<Detection>
)

/** A change in the stable set. For "disappeared" the box is the last known one (or zeros). */
data class ObjectEvent(val appeared: Boolean, val detection: Detection)

/**
 * Thread-safe hand-over between the camera thread (writer) and JavaScript (reader).
 *
 * - [snapshot] always returns the latest stable set.
 * - [drain] returns every appeared/disappeared event since the last drain and empties the queue.
 *   The queue is bounded: if JavaScript stops reading, the OLDEST events are dropped (never unbounded memory).
 * Pure Kotlin so it is unit tested on the JVM.
 */
class StableDetectionStore(private val maxEvents: Int = 32) {
    private val lock = Any()
    private var latest = StableSnapshot(0, 0, 0L, emptyList())
    private val events = ArrayDeque<ObjectEvent>()

    fun update(update: DetectionStabilizer.Update, frameWidth: Int, frameHeight: Int, nowMs: Long, labels: LabelMap) {
        synchronized(lock) {
            val previous = latest.detections.associateBy { it.classId }
            for (d in update.appeared) push(ObjectEvent(true, d))
            for (id in update.disappeared) {
                val label = labels.labelFor(id) ?: continue
                push(ObjectEvent(false, previous[id] ?: Detection(id, label, 0f, 0f, 0f, 0f, 0f)))
            }
            latest = StableSnapshot(frameWidth, frameHeight, nowMs, update.stable)
        }
    }

    fun snapshot(): StableSnapshot = synchronized(lock) { latest }

    fun drain(): List<ObjectEvent> = synchronized(lock) {
        val out = events.toList(); events.clear(); out
    }

    /** Forget everything (e.g. when currency mode takes over the camera). */
    fun clear() = synchronized(lock) {
        events.clear(); latest = StableSnapshot(latest.frameWidth, latest.frameHeight, 0L, emptyList())
    }

    private fun push(e: ObjectEvent) { events.addLast(e); while (events.size > maxEvents) events.removeFirst() }
}
