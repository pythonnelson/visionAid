package com.margelo.nitro.VisionAidObjectDetector

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * One validated detection.
 *
 * The box is NORMALISED (0.0 – 1.0) and relative to the UPRIGHT camera image
 * (the image as the user would see it held in its natural orientation),
 * with the origin in the top-left corner.
 */
data class Detection(
    val classId: Int,
    val label: String,
    val score: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    override fun toString(): String =
        "$label (class=$classId) score=${"%.2f".format(score)} " +
            "box=[l=${"%.3f".format(left)}, t=${"%.3f".format(top)}, " +
            "r=${"%.3f".format(right)}, b=${"%.3f".format(bottom)}]"
}

/**
 * Converts the model's class number into a human readable name.
 *
 * EfficientDet-Lite0 uses a ZERO-BASED class index into a 90 entry label map
 * (0 = person, 61 = chair ...). There is NO background class and NO offset.
 * Entries that the COCO dataset never used are stored as "n/a" placeholders
 * and must stay in the file so the numbering is not shifted.
 */
class LabelMap(private val labels: List<String>) {

    val size: Int get() = labels.size

    /** Returns the label, or null if the id is out of range or a placeholder. */
    fun labelFor(classId: Int): String? {
        if (classId < 0 || classId >= labels.size) return null
        val label = labels[classId]
        return if (label.isEmpty() || label == PLACEHOLDER || label == "???") null else label
    }

    companion object {
        const val PLACEHOLDER = "n/a"

        fun fromLines(lines: List<String>): LabelMap =
            LabelMap(lines.map { it.trim() })
    }
}

/** Result of decoding one inference. Keeps raw/valid counts so logs can tell them apart. */
data class DecodeResult(
    val detections: List<Detection>,
    /** Number of slots the model says it filled (usually 25 – see README). */
    val rawCount: Int,
    /** Slots rejected because box/score/class was invalid (NaN, out of range, placeholder...). */
    val invalidCount: Int,
    /** Valid slots dropped because the score was below the threshold. */
    val belowThresholdCount: Int
)

/**
 * Turns the four raw output tensors of the TFLite_Detection_PostProcess op
 * into validated [Detection]s.
 *
 * Output contract (verified against efficientdet_lite0.tflite):
 *  - boxes   [1, N, 4] float32, order [ymin, xmin, ymax, xmax], normalised 0..1
 *  - classes [1, N]    float32, zero-based class index
 *  - scores  [1, N]    float32, 0..1
 *  - count   [1]       float32, number of filled slots
 * The model already applied non-maximum suppression, so we do NOT run it again.
 */
object DetectionDecoder {

    fun decode(
        boxes: Array<FloatArray>,
        classes: FloatArray,
        scores: FloatArray,
        count: Float,
        labels: LabelMap,
        confidenceThreshold: Float
    ): DecodeResult {
        val slots = min(boxes.size, min(classes.size, scores.size))
        val rawCount =
            if (count.isFinite()) count.roundToInt().coerceIn(0, slots) else 0

        val result = ArrayList<Detection>()
        var invalid = 0
        var belowThreshold = 0

        for (i in 0 until rawCount) {
            val score = scores[i]
            val classFloat = classes[i]
            val box = boxes[i]

            if (!score.isFinite() || score < 0f || score > 1f ||
                !classFloat.isFinite() || abs(classFloat - classFloat.roundToInt()) > 1e-3f ||
                box.size < 4 || box.any { !it.isFinite() }
            ) {
                invalid++
                continue
            }

            val classId = classFloat.roundToInt()
            val label = labels.labelFor(classId)
            if (label == null) {
                invalid++
                continue
            }

            // Model order is [ymin, xmin, ymax, xmax]; clamp, then order min/max.
            val yA = box[0].coerceIn(0f, 1f)
            val xA = box[1].coerceIn(0f, 1f)
            val yB = box[2].coerceIn(0f, 1f)
            val xB = box[3].coerceIn(0f, 1f)
            val top = min(yA, yB)
            val bottom = max(yA, yB)
            val left = min(xA, xB)
            val right = max(xA, xB)

            if (right - left <= 0f || bottom - top <= 0f) {
                invalid++
                continue
            }

            if (score < confidenceThreshold) {
                belowThreshold++
                continue
            }

            result.add(Detection(classId, label, score, left, top, right, bottom))
        }

        result.sortByDescending { it.score }
        return DecodeResult(result, rawCount, invalid, belowThreshold)
    }

    /**
     * Debug helper: the first [n] RAW model slots exactly as the model produced them
     * (before validation or thresholding), one text line per slot.
     */
    fun rawSummary(
        boxes: Array<FloatArray>,
        classes: FloatArray,
        scores: FloatArray,
        count: Float,
        labels: LabelMap,
        n: Int
    ): List<String> {
        val slots = min(boxes.size, min(classes.size, scores.size))
        return (0 until min(n, slots)).map { i ->
            val id = if (classes[i].isFinite()) classes[i].roundToInt() else -1
            "slot=$i class=${classes[i]} label=${labels.labelFor(id) ?: "?"} " +
                "score=${scores[i]} box_yxyx=${boxes[i].joinToString(",")} (count=$count)"
        }
    }
}
