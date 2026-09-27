package com.margelo.nitro.VisionAidObjectDetector

import android.graphics.ImageFormat
import android.media.Image
import android.util.Log

import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import com.margelo.nitro.camera.HybridFrameSpec
import com.margelo.nitro.camera.public.NativeFrame

import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

class HybridVisionAidObjectDetector :
    HybridVisionAidObjectDetectorSpec() {

    companion object {
        private const val TAG = "VisionAidMLKit"
    }

    private val detector = ObjectDetection.getClient(
        ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
            .enableMultipleObjects()
            .enableClassification()
            .build()
    )

    /**
     * Prevents multiple ML Kit tasks from running simultaneously.
     */
    private val processing = AtomicBoolean(false)

    override fun detect(frame: HybridFrameSpec) {

        /*
         * If ML Kit is still processing the previous frame,
         * skip this frame.
         */
        if (!processing.compareAndSet(false, true)) {
            return
        }

        try {

            /*
             * Convert the generic VisionCamera frame into
             * the native Android frame.
             */
            val nativeFrame = frame as? NativeFrame

            if (nativeFrame == null) {
                Log.e(TAG, "Could not unwrap VisionCamera frame")
                processing.set(false)
                return
            }

            /*
             * Get the ImageProxy owned by VisionCamera.
             */
            val imageProxy = nativeFrame.image

            /*
             * Get the underlying Android Image.
             */
            val mediaImage = imageProxy.image

            if (mediaImage == null) {
                Log.w(TAG, "Frame contains no media image")
                processing.set(false)
                return
            }

            /*
             * IMPORTANT:
             *
             * We copy the YUV data NOW while the camera frame
             * is still alive.
             *
             * ML Kit will receive this independent byte array,
             * so it no longer depends on the VisionCamera frame.
             */
            val nv21 = yuv420888ToNv21(mediaImage)

            val width = mediaImage.width
            val height = mediaImage.height

            val rotationDegrees =
                imageProxy.imageInfo.rotationDegrees

            /*
             * Create an InputImage backed by our copied data.
             */
            val inputImage = InputImage.fromByteArray(
                nv21,
                width,
                height,
                rotationDegrees,
                ImageFormat.NV21
            )

            /*
             * Now ML Kit can process the independent byte array
             * asynchronously.
             */
            detector.process(inputImage)
                .addOnSuccessListener { objects ->

                    Log.d(
                        TAG,
                        "ML Kit detected ${objects.size} object(s)"
                    )

                    for (obj in objects) {

                        Log.d(
                            TAG,
                            "Object trackingId=${obj.trackingId}, " +
                                    "box=${obj.boundingBox}"
                        )

                        for (label in obj.labels) {

                            Log.d(
                                TAG,
                                "Label=${label.text}, " +
                                        "confidence=${label.confidence}"
                            )
                        }
                    }
                }
                .addOnFailureListener { error ->

                    Log.e(
                        TAG,
                        "ML Kit detection failed",
                        error
                    )
                }
                .addOnCompleteListener {

                    /*
                     * Allow the next camera frame to be processed.
                     */
                    processing.set(false)
                }

        } catch (error: Exception) {

            Log.e(
                TAG,
                "Exception while preparing frame",
                error
            )

            processing.set(false)
        }
    }

    /**
     * Converts Android YUV_420_888 into an independent NV21
     * byte array.
     *
     * The returned byte array no longer depends on the
     * camera ImageProxy.
     */
    private fun yuv420888ToNv21(image: Image): ByteArray {

        val width = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride

        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride

        val output = ByteArray(
            width * height +
                    (width * height / 2)
        )

        var outputOffset = 0

        /*
         * Copy Y plane.
         */
        for (row in 0 until height) {

            val rowStart = row * yRowStride

            for (column in 0 until width) {

                output[outputOffset++] =
                    yBuffer.get(rowStart + column)
            }
        }

        /*
         * Copy V and U planes in NV21 order:
         *
         * V U V U V U ...
         */
        val chromaHeight = height / 2
        val chromaWidth = width / 2

        for (row in 0 until chromaHeight) {

            val vRowStart = row * vRowStride
            val uRowStart = row * uRowStride

            for (column in 0 until chromaWidth) {

                val vIndex =
                    vRowStart + column * vPixelStride

                val uIndex =
                    uRowStart + column * uPixelStride

                output[outputOffset++] =
                    vBuffer.get(vIndex)

                output[outputOffset++] =
                    uBuffer.get(uIndex)
            }
        }

        return output
    }
}
