package com.margelo.nitro.VisionAidObjectDetector

import java.nio.ByteBuffer

/**
 * Plain copy of the three planes of an Android YUV_420_888 image.
 * Kept free of android.* types so it can be unit tested on the JVM.
 *
 * [cropLeft]/[cropTop]/[cropWidth]/[cropHeight] describe the valid region
 * inside the (width x height) buffer (Image.cropRect).
 */
class YuvFrame(
    val y: ByteArray,
    val u: ByteArray,
    val v: ByteArray,
    val yRowStride: Int,
    val yPixelStride: Int,
    val uvRowStride: Int,
    val uvPixelStride: Int,
    val cropLeft: Int,
    val cropTop: Int,
    val cropWidth: Int,
    val cropHeight: Int
)

/** Maps coordinates of the upright (rotated) image back to the raw sensor buffer. */
object Orientation {

    /** Width of the upright image. [rotation] = clockwise degrees needed to make the frame upright. */
    fun uprightWidth(w: Int, h: Int, rotation: Int) = if (rotation == 90 || rotation == 270) h else w

    fun uprightHeight(w: Int, h: Int, rotation: Int) = if (rotation == 90 || rotation == 270) w else h

    /**
     * Converts a continuous point (u, v) of the upright image (0..uprightW, 0..uprightH)
     * into the continuous point (x, y) of the raw buffer (0..w, 0..h).
     * [out] receives x at index 0 and y at index 1.
     *
     * If [mirrored], the upright image is flipped left-to-right first.
     */
    fun toSource(
        u: Float, v: Float, w: Int, h: Int, rotation: Int, mirrored: Boolean, out: FloatArray
    ) {
        val uw = uprightWidth(w, h, rotation).toFloat()
        val uu = if (mirrored) uw - u else u
        when (rotation) {
            0 -> { out[0] = uu; out[1] = v }
            90 -> { out[0] = v; out[1] = h - uu }
            180 -> { out[0] = w - uu; out[1] = h - v }
            270 -> { out[0] = w - v; out[1] = uu }
            else -> throw IllegalArgumentException("Unsupported rotation: $rotation")
        }
    }
}

/**
 * Converts a YUV_420_888 camera frame straight into the model's input tensor:
 * crop -> rotate upright -> (optional mirror) -> resize (stretch) to
 * outW x outH -> RGB bytes (R,G,B order, 0..255).
 *
 * No Bitmap and no JPEG step is involved, so there is no compression loss.
 * Colour conversion is full-range BT.601 (same as Android's YuvImage / JPEG).
 * Each output pixel is the average of 4 samples inside its footprint, which
 * limits aliasing when a 1280x720 frame is shrunk to 320x320.
 */
object YuvPreprocessor {

    fun toRgbBuffer(
        frame: YuvFrame,
        rotation: Int,
        mirrored: Boolean,
        outW: Int,
        outH: Int,
        out: ByteBuffer
    ) {
        val cw = frame.cropWidth
        val ch = frame.cropHeight
        val uw = Orientation.uprightWidth(cw, ch, rotation)
        val uh = Orientation.uprightHeight(cw, ch, rotation)
        val stepX = uw.toFloat() / outW
        val stepY = uh.toFloat() / outH
        val p = FloatArray(2)
        val offsets = floatArrayOf(0.25f, 0.75f)

        out.clear()
        for (oy in 0 until outH) {
            for (ox in 0 until outW) {
                var r = 0; var g = 0; var b = 0
                for (dy in offsets) {
                    for (dx in offsets) {
                        Orientation.toSource(
                            (ox + dx) * stepX, (oy + dy) * stepY,
                            cw, ch, rotation, mirrored, p
                        )
                        val x = p[0].toInt().coerceIn(0, cw - 1) + frame.cropLeft
                        val y = p[1].toInt().coerceIn(0, ch - 1) + frame.cropTop
                        val rgb = yuvToRgb(frame, x, y)
                        r += (rgb shr 16) and 0xFF
                        g += (rgb shr 8) and 0xFF
                        b += rgb and 0xFF
                    }
                }
                out.put((r shr 2).toByte())
                out.put((g shr 2).toByte())
                out.put((b shr 2).toByte())
            }
        }
        out.rewind()
    }

    /**
     * Currency model input: the CENTRE SQUARE of the upright frame, resized to outSize x outSize,
     * written as FLOAT32 R,G,B values in 0..255 (the currency model rescales internally).
     * Same crop / rotation / mirror / colour conversion as [toRgbBuffer]; the object detector is not affected.
     */
    fun toFloatRgbSquare(frame: YuvFrame, rotation: Int, mirrored: Boolean, outSize: Int, out: ByteBuffer) {
        val cw = frame.cropWidth
        val ch = frame.cropHeight
        val uw = Orientation.uprightWidth(cw, ch, rotation)
        val uh = Orientation.uprightHeight(cw, ch, rotation)
        val side = minOf(uw, uh)
        val left = (uw - side) / 2f
        val top = (uh - side) / 2f
        val step = side.toFloat() / outSize
        val p = FloatArray(2)
        val offsets = floatArrayOf(0.25f, 0.75f)

        out.clear()
        for (oy in 0 until outSize) {
            for (ox in 0 until outSize) {
                var r = 0; var g = 0; var b = 0
                for (dy in offsets) {
                    for (dx in offsets) {
                        Orientation.toSource(left + (ox + dx) * step, top + (oy + dy) * step, cw, ch, rotation, mirrored, p)
                        val x = p[0].toInt().coerceIn(0, cw - 1) + frame.cropLeft
                        val y = p[1].toInt().coerceIn(0, ch - 1) + frame.cropTop
                        val rgb = yuvToRgb(frame, x, y)
                        r += (rgb shr 16) and 0xFF
                        g += (rgb shr 8) and 0xFF
                        b += rgb and 0xFF
                    }
                }
                out.putFloat(r / 4f)
                out.putFloat(g / 4f)
                out.putFloat(b / 4f)
            }
        }
        out.rewind()
    }

    /** Returns 0x00RRGGBB for the raw-buffer pixel (x, y). */
    fun yuvToRgb(f: YuvFrame, x: Int, y: Int): Int {
        val yy = f.y[y * f.yRowStride + x * f.yPixelStride].toInt() and 0xFF
        val uvIndex = (y shr 1) * f.uvRowStride + (x shr 1) * f.uvPixelStride
        val uu = (f.u[uvIndex].toInt() and 0xFF) - 128
        val vv = (f.v[uvIndex].toInt() and 0xFF) - 128

        val r = (yy + 1.402f * vv).toInt().coerceIn(0, 255)
        val g = (yy - 0.344136f * uu - 0.714136f * vv).toInt().coerceIn(0, 255)
        val b = (yy + 1.772f * uu).toInt().coerceIn(0, 255)
        return (r shl 16) or (g shl 8) or b
    }
}
