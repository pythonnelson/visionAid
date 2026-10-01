package com.margelo.nitro.VisionAidObjectDetector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.util.Log

import com.margelo.nitro.camera.HybridFrameSpec
import com.margelo.nitro.camera.public.NativeFrame

import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

class HybridVisionAidObjectDetector :
    HybridVisionAidObjectDetectorSpec() {

    companion object {
        private const val TAG = "VisionAidTFLite"

        private const val MODEL_FILE =
            "efficientdet_lite0.tflite"

        private const val LABEL_FILE =
            "coco_labels.txt"

        private const val CONFIDENCE_THRESHOLD = 0.50f
    }

    private val interpreter: Interpreter

    private val labels: List<String>

    private val processing = AtomicBoolean(false)

    init {
        interpreter = Interpreter(
            loadModelFile(MODEL_FILE),
            Interpreter.Options().apply {
                setNumThreads(4)
            }
        )

        labels = loadLabels(LABEL_FILE)

        Log.d(
            TAG,
            "TFLite model loaded successfully"
        )

        Log.d(
            TAG,
            "Input shape: ${
                interpreter.getInputTensor(0).shape().contentToString()
            }"
        )

        Log.d(
            TAG,
            "Input type: ${
                interpreter.getInputTensor(0).dataType()
            }"
        )

        for (i in 0 until interpreter.outputTensorCount) {
            Log.d(
                TAG,
                "Output[$i] shape=${
                    interpreter.getOutputTensor(i)
                        .shape()
                        .contentToString()
                } type=${
                    interpreter.getOutputTensor(i).dataType()
                }"
            )
        }
    }

    override fun detect(frame: HybridFrameSpec) {

        /*
         * Don't allow multiple inference operations
         * at the same time.
         */
        if (!processing.compareAndSet(false, true)) {
            return
        }

        try {

            /*
             * Convert VisionCamera's frame to the
             * native Android frame.
             */
            val nativeFrame = frame as? NativeFrame

            if (nativeFrame == null) {
                Log.e(
                    TAG,
                    "Could not unwrap VisionCamera frame"
                )

                processing.set(false)
                return
            }

            val imageProxy = nativeFrame.image

            val mediaImage = imageProxy.image

            if (mediaImage == null) {
                Log.w(
                    TAG,
                    "Frame contains no media image"
                )

                processing.set(false)
                return
            }

            /*
             * Convert YUV camera data to a Bitmap.
             */
            val bitmap = yuvToBitmap(mediaImage)

            /*
             * Run object detection.
             */
            detectObjects(bitmap)

            bitmap.recycle()

        } catch (error: Exception) {

            Log.e(
                TAG,
                "TFLite detection error",
                error
            )

        } finally {

            processing.set(false)
        }
    }

    private fun detectObjects(bitmap: Bitmap) {

        val inputTensor =
            interpreter.getInputTensor(0)

        val inputShape =
            inputTensor.shape()

        val inputHeight =
            inputShape[1]

        val inputWidth =
            inputShape[2]

        /*
         * Resize camera image to the model's
         * required input size.
         */
        val resizedBitmap =
            Bitmap.createScaledBitmap(
                bitmap,
                inputWidth,
                inputHeight,
                true
            )

        val inputBuffer =
            bitmapToInputBuffer(
                resizedBitmap,
                inputTensor.dataType()
            )

        resizedBitmap.recycle()

        /*
         * EfficientDet Lite models expose four
         * detection outputs:
         *
         * 0 = boxes
         * 1 = classes
         * 2 = scores
         * 3 = number of detections
         */
        val outputBoxes =
            Array(1) {
                Array(
                    interpreter
                        .getOutputTensor(0)
                        .shape()[1]
                ) {
                    FloatArray(4)
                }
            }

        val outputClasses =
            Array(1) {
                FloatArray(
                    interpreter
                        .getOutputTensor(1)
                        .shape()[1]
                )
            }

        val outputScores =
            Array(1) {
                FloatArray(
                    interpreter
                        .getOutputTensor(2)
                        .shape()[1]
                )
            }

        val outputCount =
            FloatArray(1)

        val outputs =
            hashMapOf<Int, Any>(
                0 to outputBoxes,
                1 to outputClasses,
                2 to outputScores,
                3 to outputCount
            )

        interpreter.runForMultipleInputsOutputs(
            arrayOf(inputBuffer),
            outputs
        )

        val detectionCount =
            outputCount[0].toInt()

        for (i in 0 until detectionCount) {

            val confidence =
                outputScores[0][i]

            if (confidence < CONFIDENCE_THRESHOLD) {
                continue
            }

            val classIndex =
                outputClasses[0][i].toInt()

            if (
                classIndex < 0 ||
                classIndex >= labels.size
            ) {
                continue
            }

            val label =
                labels[classIndex]

            val box =
                outputBoxes[0][i]

            Log.d(
                TAG,
                "DETECTED: $label " +
                        "confidence=${"%.2f".format(confidence)} " +
                        "box=${box.contentToString()}"
            )
        }
    }

    private fun bitmapToInputBuffer(
        bitmap: Bitmap,
        dataType: DataType
    ): ByteBuffer {

        val pixelCount =
            bitmap.width * bitmap.height

        val bytesPerChannel =
            if (dataType == DataType.FLOAT32) {
                4
            } else {
                1
            }

        val buffer =
            ByteBuffer.allocateDirect(
                pixelCount * 3 * bytesPerChannel
            )

        buffer.order(
            ByteOrder.nativeOrder()
        )

        val pixels =
            IntArray(pixelCount)

        bitmap.getPixels(
            pixels,
            0,
            bitmap.width,
            0,
            0,
            bitmap.width,
            bitmap.height
        )

        for (pixel in pixels) {

            val r =
                (pixel shr 16) and 0xFF

            val g =
                (pixel shr 8) and 0xFF

            val b =
                pixel and 0xFF

            if (dataType == DataType.FLOAT32) {

                /*
                 * EfficientDet models commonly expect
                 * normalized float RGB input.
                 */
                buffer.putFloat(
                    r / 255.0f
                )

                buffer.putFloat(
                    g / 255.0f
                )

                buffer.putFloat(
                    b / 255.0f
                )

            } else {

                buffer.put(r.toByte())
                buffer.put(g.toByte())
                buffer.put(b.toByte())
            }
        }

        buffer.rewind()

        return buffer
    }

    private fun yuvToBitmap(
        image: Image
    ): Bitmap {

        val nv21 =
            yuv420888ToNv21(image)

        val yuvImage =
            YuvImage(
                nv21,
                ImageFormat.NV21,
                image.width,
                image.height,
                null
            )

        val outputStream =
            java.io.ByteArrayOutputStream()

        yuvImage.compressToJpeg(
            Rect(
                0,
                0,
                image.width,
                image.height
            ),
            90,
            outputStream
        )

        val jpegData =
            outputStream.toByteArray()

        return BitmapFactory.decodeByteArray(
            jpegData,
            0,
            jpegData.size
        )
    }

    private fun yuv420888ToNv21(
        image: Image
    ): ByteArray {

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

        val output =
            ByteArray(
                width * height +
                        width * height / 2
            )

        var outputOffset = 0

        for (row in 0 until height) {

            val rowStart =
                row * yRowStride

            for (column in 0 until width) {

                output[outputOffset++] =
                    yBuffer.get(
                        rowStart + column
                    )
            }
        }

        val chromaHeight =
            height / 2

        val chromaWidth =
            width / 2

        for (row in 0 until chromaHeight) {

            val vRowStart =
                row * vRowStride

            val uRowStart =
                row * uRowStride

            for (column in 0 until chromaWidth) {

                val vIndex =
                    vRowStart +
                            column * vPixelStride

                val uIndex =
                    uRowStart +
                            column * uPixelStride

                output[outputOffset++] =
                    vBuffer.get(vIndex)

                output[outputOffset++] =
                    uBuffer.get(uIndex)
            }
        }

        return output
    }

    private fun loadModelFile(
        fileName: String
    ): ByteBuffer {

        val inputStream =
            javaClass.classLoader!!
                .getResourceAsStream(
                    "assets/$fileName"
                )
                ?: throw IllegalStateException(
                    "Could not find model: $fileName"
                )

        val bytes =
            inputStream.readBytes()

        inputStream.close()

        val buffer =
            ByteBuffer.allocateDirect(
                bytes.size
            )

        buffer.order(
            ByteOrder.nativeOrder()
        )

        buffer.put(bytes)

        buffer.rewind()

        return buffer
    }

    private fun loadLabels(
        fileName: String
    ): List<String> {

        val inputStream =
            javaClass.classLoader!!
                .getResourceAsStream(
                    "assets/$fileName"
                )
                ?: throw IllegalStateException(
                    "Could not find labels: $fileName"
                )

        return BufferedReader(
            InputStreamReader(inputStream)
        ).useLines { lines ->
            lines.toList()
        }
    }
}
