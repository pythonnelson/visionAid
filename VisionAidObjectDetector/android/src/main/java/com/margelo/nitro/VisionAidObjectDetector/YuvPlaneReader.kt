package com.margelo.nitro.VisionAidObjectDetector

import android.graphics.Rect
import android.media.Image
import java.nio.ByteBuffer

/** Copies the planes of a YUV_420_888 [Image] into reusable arrays (honouring strides) and returns a [YuvFrame]. */
class YuvPlaneReader {
    private var y = ByteArray(0)
    private var u = ByteArray(0)
    private var v = ByteArray(0)

    fun read(image: Image, crop: Rect): YuvFrame {
        val planes = image.planes
        require(planes.size == 3) { "YUV image must have 3 planes, has ${planes.size}" }
        val (yp, up, vp) = Triple(planes[0], planes[1], planes[2])
        require(up.rowStride == vp.rowStride && up.pixelStride == vp.pixelStride) { "U and V planes have different strides" }
        y = copy(yp.buffer, y); u = copy(up.buffer, u); v = copy(vp.buffer, v)
        val c = if (crop.isEmpty) Rect(0, 0, image.width, image.height) else crop
        return YuvFrame(y, u, v, yp.rowStride, yp.pixelStride, up.rowStride, up.pixelStride, c.left, c.top, c.width(), c.height())
    }

    private fun copy(buffer: ByteBuffer, reuse: ByteArray): ByteArray {
        val dup = buffer.duplicate()
        dup.rewind()
        val array = if (reuse.size == dup.remaining()) reuse else ByteArray(dup.remaining())
        dup.get(array)
        return array
    }
}
