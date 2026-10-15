package com.margelo.nitro.VisionAidObjectDetector

/**
 * Decides which labels are "really there" by looking at several consecutive frames,
 * so the app does not announce a new object every time one frame flips its mind.
 *
 * This layer is independent of the model: it only receives already decoded, already
 * confidence-filtered [Detection]s. It never changes what the model said, it only
 * waits for repeated agreement.
 *
 * Rules (counted in PROCESSED frames, not seconds):
 *  - A class becomes STABLE when it was seen in at least [hitsRequired] of the last
 *    [windowSize] frames  (default 3 of 5).
 *  - Fast path: a very confident detection (score >= [fastScore], default 0.80)
 *    becomes stable after [fastHitsRequired] hits (default 2), so an important
 *    object (e.g. a person) is not delayed longer than necessary.
 *  - Only detections with score >= [enterScore] can start or build up a class; a class that is
 *    already stable is also kept alive by weaker detections (hysteresis, see constructor).
 *  - A stable class is dropped after [missesToDrop] consecutive frames without it
 *    (default 3), so a stale label does not stay after the object disappears.
 *  - Classes that are not stable are forgotten once they have no hit in the window.
 *  - A class is reported in [Update.appeared] exactly once per stable period; it is
 *    announced again only after it was dropped and came back.
 * Trade-off: delay of 2-3 processed frames (~0.3 s at the current ~10 fps), and objects
 * that flicker in and out of detection are intentionally ignored.
 */
class DetectionStabilizer(
    private val windowSize: Int = 5,
    private val hitsRequired: Int = 3,
    private val fastScore: Float = 0.80f,
    private val fastHitsRequired: Int = 2,
    private val missesToDrop: Int = 3,
    /**
     * Hysteresis: a class needs a score >= [enterScore] to START being tracked, but once it
     * is stable, weaker detections (down to whatever the caller passes in) keep it alive.
     * Stops objects hovering around the threshold from flapping on and off.
     */
    private val enterScore: Float = 0.60f
) {

    data class Update(
        /** Currently stable detections (best box/score per class, highest score first). */
        val stable: List<Detection>,
        /** Classes that became stable in this frame. */
        val appeared: List<Detection>,
        /** Class ids that stopped being stable in this frame. */
        val disappeared: List<Int>
    )

    private class Track(val history: ArrayDeque<Boolean> = ArrayDeque()) {
        var stable = false
        var consecutiveMisses = 0
        var last: Detection? = null
        var maxScoreInWindow = 0f
    }

    private val tracks = HashMap<Int, Track>()

    fun reset() = tracks.clear()

    /** Call once per processed frame with that frame's accepted detections (may be empty). */
    fun update(detections: List<Detection>): Update {
        // Best detection per class in this frame.
        val best = detections.groupBy { it.classId }
            .mapValues { e -> e.value.maxByOrNull { it.score }!! }
            .filter { (classId, d) -> d.score >= enterScore || tracks[classId]?.stable == true }
        for (classId in best.keys) tracks.getOrPut(classId) { Track() }

        val appeared = ArrayList<Detection>()
        val disappeared = ArrayList<Int>()
        val iterator = tracks.entries.iterator()

        while (iterator.hasNext()) {
            val (classId, track) = iterator.next()
            val hit = best[classId]
            track.history.addLast(hit != null)
            while (track.history.size > windowSize) track.history.removeFirst()

            if (hit != null) {
                track.last = hit
                track.consecutiveMisses = 0
                track.maxScoreInWindow = maxOf(track.maxScoreInWindow, hit.score)
            } else {
                track.consecutiveMisses++
            }

            val hits = track.history.count { it }
            if (!track.stable) {
                val becomesStable = hits >= hitsRequired ||
                    (hits >= fastHitsRequired && track.maxScoreInWindow >= fastScore)
                if (becomesStable && track.last != null) {
                    track.stable = true
                    appeared.add(track.last!!)
                } else if (hits == 0) {
                    iterator.remove()
                }
            } else if (track.consecutiveMisses >= missesToDrop) {
                disappeared.add(classId)
                iterator.remove()
            }
        }

        val stable = tracks.values.filter { it.stable }.mapNotNull { it.last }
            .sortedByDescending { it.score }
        return Update(stable, appeared, disappeared)
    }
}
