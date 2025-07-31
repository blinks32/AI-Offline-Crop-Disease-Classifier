# API Documentation

## Overview

The ADTC Smart Crop Disease Classifier provides internal APIs for image processing, model inference, and result analysis. While primarily designed as a standalone mobile application, these APIs can be extended for integration with other agricultural systems.

## Core APIs

### 1. Image Classification API

#### `CropDiseaseClassifier`

Main class for crop disease detection functionality.

```kotlin
class CropDiseaseClassifier(private val context: Context) {
    
    /**
     * Initialize the TensorFlow Lite model
     * @return Boolean indicating successful initialization
     */
    suspend fun initialize(): Boolean
    
    /**
     * Classify a crop image for disease detection
     * @param bitmap Input image as Android Bitmap
     * @return ClassificationResult with predictions and confidence
     */
    suspend fun classifyImage(bitmap: Bitmap): ClassificationResult
    
    /**
     * Release model resources
     */
    fun close()
}
```

#### Usage Example

```kotlin
// Initialize classifier
val classifier = CropDiseaseClassifier(context)
val initialized = classifier.initialize()

if (initialized) {
    // Classify image
    val result = classifier.classifyImage(imageBitmap)
    
    // Process results
    when (result.topClass) {
        "Healthy" -> showHealthyResult(result.confidence)
        "Diseased" -> showDiseasedResult(result.confidence)
        "Non-Crop" -> showNonCropResult(result.confidence)
    }
    
    // Clean up
    classifier.close()
}
```

### 2. Data Models

#### `ClassificationResult`

```kotlin
data class ClassificationResult(
    val predictions: List<Prediction>,
    val topClass: String,
    val confidence: Float,
    val processingTime: Long,
    val isValid: Boolean
) {
    /**
     * Get prediction for specific class
     */
    fun getPredictionForClass(className: String): Prediction?
    
    /**
     * Check if result meets minimum confidence threshold
     */
    fun meetsConfidenceThreshold(threshold: Float = 0.7f): Boolean
}
```

#### `Prediction`

```kotlin
data class Prediction(
    val className: String,
    val confidence: Float,
    val displayName: String
) : Comparable<Prediction> {
    
    override fun compareTo(other: Prediction): Int {
        return other.confidence.compareTo(this.confidence)
    }
}
```

### 3. Image Processing API

#### `ImageProcessor`

Handles image preprocessing for model input.

```kotlin
object ImageProcessor {
    
    /**
     * Preprocess image for model inference
     * @param bitmap Input image
     * @param targetSize Target dimensions (default: 224x224)
     * @return Preprocessed ByteBuffer ready for model input
     */
    fun preprocessImage(
        bitmap: Bitmap, 
        targetSize: Int = 224
    ): ByteBuffer
    
    /**
     * Resize image maintaining aspect ratio
     * @param bitmap Input image
     * @param targetSize Target size for square output
     * @return Resized bitmap
     */
    fun resizeImage(bitmap: Bitmap, targetSize: Int): Bitmap
    
    /**
     * Normalize pixel values to [0,1] range
     * @param bitmap Input image
     * @return Normalized ByteBuffer
     */
    fun normalizePixels(bitmap: Bitmap): ByteBuffer
}
```

#### Usage Example

```kotlin
// Preprocess image for model
val preprocessed = ImageProcessor.preprocessImage(originalBitmap)

// Or step by step
val resized = ImageProcessor.resizeImage(originalBitmap, 224)
val normalized = ImageProcessor.normalizePixels(resized)
```

### 4. Camera Integration API

#### `CameraManager`

Manages camera operations and image capture.

```kotlin
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    
    /**
     * Initialize camera with preview
     * @param previewView Surface for camera preview
     * @param onImageCaptured Callback for captured images
     */
    fun initializeCamera(
        previewView: PreviewView,
        onImageCaptured: (Bitmap) -> Unit
    )
    
    /**
     * Capture image for analysis
     */
    fun captureImage()
    
    /**
     * Release camera resources
     */
    fun releaseCamera()
}
```

### 5. Model Management API

#### `ModelManager`

Handles TensorFlow Lite model operations.

```kotlin
class ModelManager(private val context: Context) {
    
    /**
     * Load model from assets
     * @param modelPath Path to .tflite file in assets
     * @return Loaded Interpreter instance
     */
    fun loadModel(modelPath: String = "model.tflite"): Interpreter?
    
    /**
     * Run inference on preprocessed input
     * @param interpreter TensorFlow Lite interpreter
     * @param input Preprocessed image data
     * @return Raw model output
     */
    fun runInference(
        interpreter: Interpreter, 
        input: ByteBuffer
    ): Array<FloatArray>
    
    /**
     * Process model output to predictions
     * @param output Raw model output
     * @param classLabels List of class names
     * @return List of predictions
     */
    fun processOutput(
        output: Array<FloatArray>,
        classLabels: List<String>
    ): List<Prediction>
}
```

## Configuration

### Model Configuration

```kotlin
object ModelConfig {
    const val MODEL_PATH = "model.tflite"
    const val INPUT_SIZE = 224
    const val NUM_CLASSES = 3
    const val CONFIDENCE_THRESHOLD = 0.7f
    
    val CLASS_LABELS = listOf("Healthy", "Diseased", "Non-Crop")
    val CLASS_COLORS = mapOf(
        "Healthy" to Color.GREEN,
        "Diseased" to Color.RED,
        "Non-Crop" to Color.GRAY
    )
}
```

### Performance Configuration

```kotlin
object PerformanceConfig {
    const val USE_GPU_ACCELERATION = true
    const val NUM_THREADS = 4
    const val MAX_INFERENCE_TIME_MS = 5000L
    const val ENABLE_PROFILING = false
}
```

## Error Handling

### Exception Types

```kotlin
sealed class ClassificationException(message: String) : Exception(message) {
    class ModelNotInitialized : ClassificationException("Model not initialized")
    class InvalidInput : ClassificationException("Invalid input image")
    class InferenceTimeout : ClassificationException("Inference timeout")
    class ModelCorrupted : ClassificationException("Model file corrupted")
}
```

### Error Handling Example

```kotlin
try {
    val result = classifier.classifyImage(bitmap)
    handleResult(result)
} catch (e: ClassificationException.ModelNotInitialized) {
    // Reinitialize model
    classifier.initialize()
} catch (e: ClassificationException.InvalidInput) {
    // Show error to user
    showError("Please capture a clear image of a crop leaf")
} catch (e: ClassificationException.InferenceTimeout) {
    // Retry or show timeout message
    showError("Analysis taking too long, please try again")
}
```

## Integration Examples

### Basic Integration

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var classifier: CropDiseaseClassifier
    private lateinit var cameraManager: CameraManager
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize components
        classifier = CropDiseaseClassifier(this)
        cameraManager = CameraManager(this, this)
        
        // Setup camera
        cameraManager.initializeCamera(previewView) { bitmap ->
            analyzeImage(bitmap)
        }
        
        // Initialize model
        lifecycleScope.launch {
            classifier.initialize()
        }
    }
    
    private fun analyzeImage(bitmap: Bitmap) {
        lifecycleScope.launch {
            try {
                val result = classifier.classifyImage(bitmap)
                displayResult(result)
            } catch (e: ClassificationException) {
                handleError(e)
            }
        }
    }
}
```

### Custom Model Integration

```kotlin
// Load custom model
val customClassifier = CropDiseaseClassifier(context).apply {
    modelPath = "custom_model.tflite"
    classLabels = listOf("Custom_Class_1", "Custom_Class_2")
    confidenceThreshold = 0.8f
}

// Use with custom preprocessing
val customProcessor = object : ImageProcessor {
    override fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Custom preprocessing logic
        return customPreprocessing(bitmap)
    }
}
```

## Performance Monitoring

### Metrics Collection

```kotlin
class PerformanceMonitor {
    
    fun trackInference(
        processingTime: Long,
        confidence: Float,
        modelSize: Long
    ) {
        // Log performance metrics
        Log.d("Performance", """
            Inference Time: ${processingTime}ms
            Confidence: $confidence
            Model Size: ${modelSize}MB
        """.trimIndent())
    }
    
    fun trackMemoryUsage() {
        val runtime = Runtime.getRuntime()
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        Log.d("Memory", "Used: ${usedMemory / 1024 / 1024}MB")
    }
}
```

## Testing APIs

### Unit Testing

```kotlin
@Test
fun testImageClassification() {
    val classifier = CropDiseaseClassifier(context)
    val testBitmap = createTestBitmap()
    
    runBlocking {
        classifier.initialize()
        val result = classifier.classifyImage(testBitmap)
        
        assertThat(result.isValid).isTrue()
        assertThat(result.predictions).hasSize(3)
        assertThat(result.confidence).isGreaterThan(0f)
    }
}
```

### Integration Testing

```kotlin
@Test
fun testEndToEndPipeline() {
    val testImage = loadTestImage("healthy_tomato.jpg")
    
    // Test complete pipeline
    val preprocessed = ImageProcessor.preprocessImage(testImage)
    val interpreter = ModelManager.loadModel()
    val output = ModelManager.runInference(interpreter, preprocessed)
    val predictions = ModelManager.processOutput(output, ModelConfig.CLASS_LABELS)
    
    assertThat(predictions.first().className).isEqualTo("Healthy")
}
```

## Future API Extensions

### Planned Features

1. **Batch Processing**: Process multiple images simultaneously
2. **Streaming Analysis**: Real-time video stream analysis
3. **Cloud Integration**: Optional cloud-based model updates
4. **Export APIs**: Export results to various formats (JSON, CSV)
5. **Plugin System**: Support for third-party model plugins

### API Versioning

```kotlin
object ApiVersion {
    const val MAJOR = 1
    const val MINOR = 0
    const val PATCH = 0
    const val VERSION_STRING = "$MAJOR.$MINOR.$PATCH"
}
```

---

This API documentation provides the foundation for integrating crop disease detection capabilities into other agricultural applications and systems.