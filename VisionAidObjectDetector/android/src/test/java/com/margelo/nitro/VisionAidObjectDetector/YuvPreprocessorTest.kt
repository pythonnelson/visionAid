package com.margelo.nitro.VisionAidObjectDetector

import java.nio.ByteBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class YuvPreprocessorTest {

    /** Builds a frame where Y = x + 10*y (so every pixel is unique), neutral chroma. Uses padded strides. */
    private fun frame(w: Int, h: Int, rowPad: Int = 4, uvPixelStride: Int = 2): YuvFrame {
        val yStride = w + rowPad
        val y = ByteArray(yStride * h)
        for (r in 0 until h) for (c in 0 until w) y[r * yStride + c] = (c + 10 * r).toByte()
        val uvStride = w + rowPad
        val uv = ByteArray(uvStride * (h / 2)) { 128.toByte() }
        return YuvFrame(y, uv, uv, yStride, 1, uvStride, uvPixelStride, 0, 0, w, h)
    }

    private fun gray(f: YuvFrame, x: Int, y: Int) = YuvPreprocessor.yuvToRgb(f, x, y) and 0xFF

    @Test fun neutralChromaGivesGray() {
        val f = frame(4, 4)
        assertEquals((2 + 10 * 3), gray(f, 2, 3))
        val rgb = YuvPreprocessor.yuvToRgb(f, 2, 3)
        assertEquals(rgb shr 16 and 0xFF, rgb and 0xFF)
    }

    @Test fun colourConversionKnownValues() {
        // Y=76,U=85,V=255 is (approximately) pure red in full-range BT.601
        val f = YuvFrame(byteArrayOf(76.toByte()), byteArrayOf(85.toByte()), byteArrayOf(255.toByte()), 1, 1, 1, 1, 0, 0, 1, 1)
        val rgb = YuvPreprocessor.yuvToRgb(f, 0, 0)
        assertEquals(254, rgb shr 16 and 0xFF)  // R high
        assertEquals(0, rgb shr 8 and 0xFF)     // G low
        assertEquals(0, rgb and 0xFF)           // B low
    }

    @Test fun outputIsRgbOrder() {
        val red = YuvFrame(byteArrayOf(76.toByte()), byteArrayOf(85.toByte()), byteArrayOf(255.toByte()), 1, 1, 1, 1, 0, 0, 1, 1)
        val out = ByteBuffer.allocate(3)
        YuvPreprocessor.toRgbBuffer(red, 0, false, 1, 1, out)
        assertEquals(254, out.get(0).toInt() and 0xFF) // first byte is R
        assertEquals(0, out.get(2).toInt() and 0xFF)   // last byte is B
    }

    /** Rotation: a 4x2 landscape frame rotated 90 CW becomes 2x4; compare against hand-computed pixels. */
    @Test fun rotate90ClockwiseMovesTopLeftToTopRight() {
        val f = frame(4, 2)          // raw: row0 = 0 1 2 3, row1 = 10 11 12 13
        val out = ByteBuffer.allocate(2 * 4 * 3)
        YuvPreprocessor.toRgbBuffer(f, 90, false, 2, 4, out)
        // upright (2 wide, 4 tall) after rotating CW: row0 = 10 0, row1 = 11 1, row2 = 12 2, row3 = 13 3
        assertArrayEquals(intArrayOf(10, 0, 11, 1, 12, 2, 13, 3), grayValues(out, 8))
    }

    @Test fun rotate270AndRotate180() {
        val f = frame(4, 2)
        val o270 = ByteBuffer.allocate(24)
        YuvPreprocessor.toRgbBuffer(f, 270, false, 2, 4, o270)
        assertArrayEquals(intArrayOf(3, 13, 2, 12, 1, 11, 0, 10), grayValues(o270, 8))
        val o180 = ByteBuffer.allocate(24)
        YuvPreprocessor.toRgbBuffer(f, 180, false, 4, 2, o180)
        assertArrayEquals(intArrayOf(13, 12, 11, 10, 3, 2, 1, 0), grayValues(o180, 8))
    }

    @Test fun mirrorFlipsLeftRight() {
        val f = frame(4, 2)
        val out = ByteBuffer.allocate(24)
        YuvPreprocessor.toRgbBuffer(f, 0, true, 4, 2, out)
        assertArrayEquals(intArrayOf(3, 2, 1, 0, 13, 12, 11, 10), grayValues(out, 8))
    }

    @Test fun cropOffsetIsApplied() {
        val base = frame(4, 4)
        val cropped = YuvFrame(base.y, base.u, base.v, base.yRowStride, 1, base.uvRowStride, 2, 2, 2, 2, 2)
        val out = ByteBuffer.allocate(2 * 2 * 3)
        YuvPreprocessor.toRgbBuffer(cropped, 0, false, 2, 2, out)
        assertArrayEquals(intArrayOf(22, 23, 32, 33), grayValues(out, 4))
    }

    @Test fun uprightSizeSwapsForQuarterTurns() {
        assertEquals(720, Orientation.uprightWidth(1280, 720, 90))
        assertEquals(1280, Orientation.uprightHeight(1280, 720, 270))
        assertEquals(1280, Orientation.uprightWidth(1280, 720, 180))
    }

    private fun grayValues(b: ByteBuffer, n: Int) = IntArray(n) { b.get(it * 3).toInt() and 0xFF }

    // ---- currency model input: centre square, float RGB 0..255
    private fun floats(b: ByteBuffer, n: Int): List<Float> { val f = b.asFloatBuffer(); return List(n) { f.get(it * 3) } }

    @Test fun currencyInputIsCentreSquareAsFloat0to255() {
        val f = frame(6, 2)   // landscape 6x2, Y = x + 10*y. Centre square = x in 2..3
        val out = ByteBuffer.allocate(2 * 2 * 3 * 4)
        YuvPreprocessor.toFloatRgbSquare(f, 0, false, 2, out)
        assertEquals(listOf(2f, 3f, 12f, 13f), floats(out, 4))
        assertEquals(255f, run { val g = YuvFrame(byteArrayOf(255.toByte()), byteArrayOf(128.toByte()), byteArrayOf(128.toByte()), 1, 1, 1, 1, 0, 0, 1, 1)
            val o = ByteBuffer.allocate(12); YuvPreprocessor.toFloatRgbSquare(g, 0, false, 1, o); o.asFloatBuffer().get(0) }, 0.5f)
    }

    @Test fun currencyInputRespectsRotation() {
        val f = frame(4, 2)   // rotate 90: upright is 2 wide x 4 tall; centre square = rows 1..2 of the upright image
        val out = ByteBuffer.allocate(2 * 2 * 3 * 4)
        YuvPreprocessor.toFloatRgbSquare(f, 90, false, 2, out)
        // upright rows (from the existing rotation test): [10 0],[11 1],[12 2],[13 3] -> centre rows 1..2
        assertEquals(listOf(11f, 1f, 12f, 2f), floats(out, 4))
    }
}
