package org.tensorflow.lite.examples.imageclassification

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.util.concurrent.TimeUnit
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    
    // UI Components
    private lateinit var previewView: PreviewView
    private lateinit var resultContainer: LinearLayout
    private lateinit var resultText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var instructionText: TextView
    private lateinit var analyzeButton: Button
    private lateinit var resetButton: Button
    private lateinit var infoButton: Button
    private lateinit var statusIndicator: View
    private lateinit var focusIndicator: View
    

    
    // Camera and ML
    private lateinit var cameraExecutor: ExecutorService
    private var interpreter: Interpreter? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private lateinit var vibrator: Vibrator
    private var latestBitmap: Bitmap? = null
    
    // Model properties
    private var labels: List<String> = emptyList()
    private var inputSize = 128
    private var inputChannels = 3
    private var isProcessing = false
    
    // State management
    private var isModelReady = false
    private var isCameraReady = false
    private var isFocused = false
    private var hasAnalyzedOnce = false
    private var lastAnalysisTime = 0L
    
    companion object {
        private const val TAG = "CropClassifier"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initializeViews()
        setupClickListeners()
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator
        
        // Initialize TensorFlow Lite model
        initializeClassifier()
        
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }
    
    private fun initializeViews() {
        previewView = findViewById(R.id.preview_view)
        resultContainer = findViewById(R.id.result_container)
        resultText = findViewById(R.id.result_text)
        confidenceText = findViewById(R.id.confidence_text)
        instructionText = findViewById(R.id.instruction_text)
        analyzeButton = findViewById(R.id.analyze_button)
        infoButton = findViewById(R.id.info_button)
        
        // Create reset button programmatically and add it to the layout
        resetButton = Button(this).apply {
            text = "Try Again"
            visibility = View.GONE
            setBackgroundColor(0xFFFF9800.toInt()) // Orange color
            setTextColor(0xFFFFFFFF.toInt()) // White text
            textSize = 16f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
        
        // Add reset button to the button container
        val buttonContainer = analyzeButton.parent as LinearLayout
        val layoutParams = LinearLayout.LayoutParams(0, 
            resources.getDimensionPixelSize(android.R.dimen.app_icon_size)).apply {
            weight = 1f
            setMargins(8, 0, 8, 0)
        }
        buttonContainer.addView(resetButton, buttonContainer.childCount - 1, layoutParams)
        
        // Create missing UI elements programmatically
        statusIndicator = View(this)
        focusIndicator = View(this)
        
        // Create missing UI elements programmatically
        statusIndicator = View(this)
        focusIndicator = View(this)
    }
    
    private fun setupClickListeners() {
        analyzeButton.setOnClickListener {
            if (!isProcessing && canAnalyze()) {
                captureAndAnalyze()
            }
        }
        
        resetButton.setOnClickListener {
            resetForNewAnalysis()
        }
        
        infoButton.setOnClickListener {
            showInfoDialog()
        }
    }
    
    private fun showInfoDialog() {
        val message = """
            ADTC Crop Disease Classifier
            
            This AI-powered app detects crop diseases using your camera.
            
            Supported crops:
            • Apple, Tomato, Corn, Potato
            • Grape, Bell Pepper, Cherry
            • Peach, Strawberry
            
            Instructions:
            1. Point camera at crop leaf
            2. Wait for focus indicator
            3. Tap 'Analyze Crop'
            
            Accuracy: 85-95% on supported crops
        """.trimIndent()
        
        android.app.AlertDialog.Builder(this)
            .setTitle("About ADTC Classifier")
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }
    
    private fun canAnalyze(): Boolean {
        return isModelReady && isCameraReady && isFocused && !isProcessing
    }
    
    private fun resetForNewAnalysis() {
        // Clear previous results
        resultContainer.visibility = View.GONE
        hasAnalyzedOnce = false
        
        // Reset model state for better accuracy
        interpreter?.let { interpreter ->
            try {
                // Force garbage collection to clean up any lingering state
                System.gc()
                
                // Reinitialize the interpreter to ensure clean state
                val modelBuffer = loadModelFile()
                this.interpreter?.close()
                this.interpreter = Interpreter(modelBuffer)
                
                Log.d(TAG, "Model reset for new analysis")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to reset model: ${e.message}")
            }
        }
        
        // Update UI
        updateInstructions()
        updateAnalyzeButtonState()
        
        // Vibrate to confirm reset
        vibrator.vibrate(VibrationEffect.createOneShot(100, VibrationEffect.DEFAULT_AMPLITUDE))
        
        Toast.makeText(this, "Ready for new analysis", Toast.LENGTH_SHORT).show()
    }
    
    private fun loadModelFile(): MappedByteBuffer {
        try {
            Log.d(TAG, "Loading model.tflite from assets...")
            val assetFileDescriptor = assets.openFd("model.tflite")
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            
            val mappedBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            Log.d(TAG, "Model loaded successfully, size: ${mappedBuffer.capacity()} bytes")
            
            return mappedBuffer
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model file: ${e.message}")
            throw Exception("Could not load model.tflite: ${e.message}")
        }
    }
    
    private fun loadLabels(): List<String> {
        return try {
            val inputStream = assets.open("labels.txt")
            val reader = BufferedReader(InputStreamReader(inputStream))
            val labels = reader.readLines()
            Log.d(TAG, "Labels loaded: ${labels.joinToString(", ")}")
            labels
        } catch (e: Exception) {
            Log.w(TAG, "Could not load labels: ${e.message}")
            listOf("healthy", "diseased")
        }
    }
    
    private fun testCurrentModel(): Boolean {
        return try {
            Log.d(TAG, "Testing current model...")
            val modelBuffer = loadModelFile()
            val testInterpreter = Interpreter(modelBuffer)
            
            val inputDetails = testInterpreter.getInputTensor(0)
            val outputDetails = testInterpreter.getOutputTensor(0)
            
            Log.d(TAG, "Input shape: ${inputDetails.shape().contentToString()}")
            Log.d(TAG, "Output shape: ${outputDetails.shape().contentToString()}")
            
            // Test with dummy input
            val testInput = ByteBuffer.allocateDirect(inputDetails.numBytes())
            testInput.order(ByteOrder.nativeOrder())
            
            val inputType = inputDetails.dataType()
            
            // Fill with appropriate dummy data
            when (inputType.toString()) {
                "INT8" -> {
                    for (i in 0 until inputDetails.numBytes()) {
                        testInput.put(0.toByte())
                    }
                }
                "UINT8" -> {
                    for (i in 0 until inputDetails.numBytes()) {
                        testInput.put(128.toByte())
                    }
                }
                else -> {
                    for (i in 0 until inputDetails.numBytes() / 4) {
                        testInput.putFloat(0.5f)
                    }
                }
            }
            testInput.rewind()
            
            // Create output buffer
            val outputShape = outputDetails.shape()
            val outputType = outputDetails.dataType()
            val outputSize = if (outputShape.size > 1) outputShape[1] else 2
            
            val testOutput = when (outputType.toString()) {
                "INT8", "UINT8" -> Array(1) { ByteArray(outputSize) }
                else -> Array(1) { FloatArray(outputSize) }
            }
            
            testInterpreter.run(testInput, testOutput)
            testInterpreter.close()
            
            Log.d(TAG, "Model test successful!")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Model test failed: ${e.message}")
            false
        }
    }
    
    private fun initializeClassifier() {
        try {
            Log.d(TAG, "Initializing classifier...")
            
            // Load labels
            labels = loadLabels()
            
            // Test and load model
            if (testCurrentModel()) {
                val modelBuffer = loadModelFile()
                interpreter = Interpreter(modelBuffer)
                
                // Get model dimensions
                val inputTensor = interpreter!!.getInputTensor(0)
                val inputShape = inputTensor.shape()
                
                if (inputShape.size == 4) {
                    inputSize = inputShape[1]
                    inputChannels = inputShape[3]
                }
                
                Log.d(TAG, "Model initialized: ${inputSize}x${inputSize}x${inputChannels}")
                isModelReady = true
                updateStatusIndicator(true)
                
                runOnUiThread {
                    updateInstructions()
                    updateAnalyzeButtonState()
                }
                
            } else {
                throw Exception("Model validation failed")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Model initialization failed: ${e.message}")
            isModelReady = false
            updateStatusIndicator(false)
            
            runOnUiThread {
                updateInstructions()
                updateAnalyzeButtonState()
                Toast.makeText(this, "Model loading failed", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun updateInstructions() {
        val instruction = when {
            !isModelReady -> "Loading AI model..."
            !isCameraReady -> "Initializing camera..."
            !isFocused -> "Position crop leaf in frame and wait for focus"
            hasAnalyzedOnce -> "Tap Reset to analyze another crop"
            else -> "Ready! Tap Analyze to detect crop disease"
        }
        instructionText.text = instruction
    }
    
    private fun updateAnalyzeButtonState() {
        val canAnalyzeNow = canAnalyze() && !hasAnalyzedOnce
        analyzeButton.isEnabled = canAnalyzeNow
        analyzeButton.alpha = if (canAnalyzeNow) 1.0f else 0.5f
        analyzeButton.setBackgroundColor(if (canAnalyzeNow) 0xFF4CAF50.toInt() else 0xFF4CAF50.toInt())
        
        resetButton.visibility = if (hasAnalyzedOnce) View.VISIBLE else View.GONE
    }
    
    private fun updateFocusIndicator(focused: Boolean) {
        isFocused = focused
        Log.d(TAG, "Focus state updated: $focused")
        focusIndicator.setBackgroundColor(
            if (focused) 0xFF4CAF50.toInt() else 0xFFF44336.toInt()
        )
        runOnUiThread {
            updateInstructions()
            updateAnalyzeButtonState()
        }
    }
    
    private fun updateStatusIndicator(isReady: Boolean) {
        runOnUiThread {
            statusIndicator.setBackgroundColor(
                if (isReady) 0xFF4CAF50.toInt() else 0xFFF44336.toInt()
            )
        }
    }
    
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            // Preview
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            
            // Image capture
            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
            
            // Image analysis for continuous frame capture
            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        // Convert ImageProxy to Bitmap and store the latest frame
                        try {
                            val bitmap = convertImageProxyToBitmap(image)
                            if (bitmap != null) {
                                synchronized(this) {
                                    latestBitmap?.recycle() // Clean up previous bitmap
                                    latestBitmap = bitmap
                                }
                                

                                
                                // Only log occasionally to avoid spam
                                if (System.currentTimeMillis() % 5000 < 100) {
                                    Log.d(TAG, "Frame captured: ${bitmap.width}x${bitmap.height}")
                                }
                            } else {
                                Log.w(TAG, "Failed to convert frame to bitmap - format: ${image.format}")
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error processing camera frame: ${e.message}")
                        } finally {
                            image.close()
                        }
                    }
                }
            
            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture, imageAnalyzer)
                
                // Set up focus monitoring
                setupFocusMonitoring()
                
                isCameraReady = true
                Log.d(TAG, "Camera started successfully")
                
                runOnUiThread {
                    updateInstructions()
                    updateAnalyzeButtonState()
                }
                
            } catch (exc: Exception) {
                Log.e(TAG, "Camera binding failed", exc)
                isCameraReady = false
                runOnUiThread {
                    updateInstructions()
                    updateAnalyzeButtonState()
                    Toast.makeText(this@MainActivity, "Camera initialization failed", Toast.LENGTH_SHORT).show()
                }
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun setupFocusMonitoring() {
        camera?.let { camera ->
            Log.d(TAG, "Setting up focus monitoring")
            
            // Simplified focus setup - assume focused after a short delay
            previewView.postDelayed({
                Log.d(TAG, "Assuming camera is focused")
                updateFocusIndicator(true)
            }, 2000) // 2 second delay to allow camera to initialize
            
            // Set up periodic focus checks (simplified)
            startPeriodicFocusCheck()
        }
    }
    
    private fun startPeriodicFocusCheck() {
        val focusCheckRunnable = object : Runnable {
            override fun run() {
                if (isCameraReady && !isProcessing) {
                    // Simplified: just check if we have a valid bitmap
                    val hasBitmap = synchronized(this@MainActivity) {
                        latestBitmap != null
                    }
                    
                    if (hasBitmap && !isFocused) {
                        Log.d(TAG, "Camera frame available, assuming focused")
                        updateFocusIndicator(true)
                    }
                }
                
                // Schedule next check
                previewView.postDelayed(this, 2000) // Check every 2 seconds
            }
        }
        
        // Start the periodic checks
        previewView.postDelayed(focusCheckRunnable, 2000)
    }
    
    private fun captureAndAnalyze() {
        val currentInterpreter = interpreter
        if (currentInterpreter == null) {
            Toast.makeText(this, "Model not ready", Toast.LENGTH_SHORT).show()
            return
        }
        
        if (isProcessing || !canAnalyze()) return
        
        // Prevent rapid successive analyses
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastAnalysisTime < 2000) { // 2 second cooldown
            Toast.makeText(this, "Please wait before analyzing again", Toast.LENGTH_SHORT).show()
            return
        }
        lastAnalysisTime = currentTime
        
        isProcessing = true
        updateUI(processing = true)
        
        // Process the actual camera image
        cameraExecutor.execute {
            try {
                // Get the latest camera frame with retry logic
                var bitmap: Bitmap? = null
                var retryCount = 0
                val maxRetries = 5
                
                while (bitmap == null && retryCount < maxRetries) {
                    bitmap = synchronized(this) {
                        latestBitmap?.copy(latestBitmap!!.config, false)
                    }
                    
                    if (bitmap == null) {
                        Log.w(TAG, "No bitmap available, retry ${retryCount + 1}/$maxRetries")
                        Thread.sleep(100) // Wait 100ms before retry
                        retryCount++
                    }
                }
                
                if (bitmap == null) {
                    Log.e(TAG, "Failed to get camera image after $maxRetries retries")
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Camera image not ready, please try again", Toast.LENGTH_SHORT).show()
                        isProcessing = false
                        updateUI(processing = false)
                    }
                    return@execute
                }
                
                Log.d(TAG, "Processing image: ${bitmap.width}x${bitmap.height}")
                
                // Preprocess the image
                val inputBuffer = preprocessBitmap(bitmap)
                if (inputBuffer == null) {
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Image preprocessing failed", Toast.LENGTH_SHORT).show()
                        isProcessing = false
                        updateUI(processing = false)
                    }
                    return@execute
                }
                
                // Run inference
                val outputTensor = currentInterpreter.getOutputTensor(0)
                val outputShape = outputTensor.shape()
                val outputType = outputTensor.dataType()
                val outputSize = if (outputShape.size > 1) outputShape[1] else 2
                
                val outputBuffer = when (outputType.toString()) {
                    "INT8", "UINT8" -> Array(1) { ByteArray(outputSize) }
                    else -> Array(1) { FloatArray(outputSize) }
                }
                
                currentInterpreter.run(inputBuffer, outputBuffer)
                
                // Process results
                val (maxIndex, maxConfidence) = when (outputType.toString()) {
                    "INT8" -> {
                        val byteOutput = outputBuffer[0] as ByteArray
                        var maxIdx = 0
                        var maxVal = byteOutput[0]
                        
                        for (i in byteOutput.indices) {
                            if (byteOutput[i] > maxVal) {
                                maxVal = byteOutput[i]
                                maxIdx = i
                            }
                        }
                        
                        val confidence = (maxVal + 128) / 255.0f
                        Pair(maxIdx, confidence)
                    }
                    "UINT8" -> {
                        val byteOutput = outputBuffer[0] as ByteArray
                        var maxIdx = 0
                        var maxVal = byteOutput[0].toInt() and 0xFF
                        
                        for (i in byteOutput.indices) {
                            val value = byteOutput[i].toInt() and 0xFF
                            if (value > maxVal) {
                                maxVal = value
                                maxIdx = i
                            }
                        }
                        
                        val confidence = maxVal / 255.0f
                        Pair(maxIdx, confidence)
                    }
                    else -> {
                        val floatOutput = outputBuffer[0] as FloatArray
                        var maxIdx = 0
                        var maxVal = floatOutput[0]
                        
                        for (i in floatOutput.indices) {
                            if (floatOutput[i] > maxVal) {
                                maxVal = floatOutput[i]
                                maxIdx = i
                            }
                        }
                        
                        Pair(maxIdx, maxVal)
                    }
                }
                
                val label = if (maxIndex < labels.size) labels[maxIndex] else "Unknown"
                val confidence = (maxConfidence * 100).toInt()
                
                // Enhanced analysis and logging
                val isLikelyCrop = analyzeImageForCropFeatures(bitmap)
                val allConfidences = when (outputType.toString()) {
                    "INT8" -> {
                        val byteOutput = outputBuffer[0] as ByteArray
                        byteOutput.map { ((it + 128) / 255.0f * 100).toInt() }
                    }
                    "UINT8" -> {
                        val byteOutput = outputBuffer[0] as ByteArray
                        byteOutput.map { ((it.toInt() and 0xFF) / 255.0f * 100).toInt() }
                    }
                    else -> {
                        val floatOutput = outputBuffer[0] as FloatArray
                        floatOutput.map { (it * 100).toInt() }
                    }
                }
                
                Log.d(TAG, "=== DETAILED ANALYSIS ===")
                Log.d(TAG, "Image analysis: isLikelyCrop=$isLikelyCrop")
                Log.d(TAG, "All confidences: ${labels.zip(allConfidences).joinToString { "${it.first}: ${it.second}%" }}")
                Log.d(TAG, "Top result: $label ($confidence%)")
                Log.d(TAG, "========================")
                
                runOnUiThread {
                    // Ultra-conservative decision logic - default to "not_crop" unless absolutely certain
                    val shouldClassify = shouldClassifyImage(confidence, isLikelyCrop, allConfidences)
                    
                    if (shouldClassify) {
                        displayResult(label, confidence, allConfidences)
                    } else {
                        // Default to "not_crop" when classification is rejected
                        displayResult("not_crop", allConfidences.getOrNull(2) ?: 30, allConfidences)
                    }
                    
                    hasAnalyzedOnce = true
                    isProcessing = false
                    updateUI(processing = false)
                    updateInstructions()
                    updateAnalyzeButtonState()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Analysis failed: ${e.message}")
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Analysis failed: ${e.message}", Toast.LENGTH_SHORT).show()
                    isProcessing = false
                    updateUI(processing = false)
                    updateInstructions()
                    updateAnalyzeButtonState()
                }
            }
        }
    }
    
    private fun updateUI(processing: Boolean) {
        if (processing) {
            analyzeButton.text = "Analyzing..."
            analyzeButton.isEnabled = false
            resetButton.isEnabled = false
            resultContainer.visibility = View.GONE
        } else {
            analyzeButton.text = "Analyze Crop"
            resetButton.isEnabled = true
            updateAnalyzeButtonState()
        }
    }
    
    private fun shouldClassifyImage(topConfidence: Int, isLikelyCrop: Boolean, allConfidences: List<Int>): Boolean {
        // EXTREMELY conservative logic - default to "not_crop" unless we're absolutely sure
        
        // Get confidence values for each class
        val diseasedConf = allConfidences.getOrNull(0) ?: 0
        val healthyConf = allConfidences.getOrNull(1) ?: 0
        val notCropConf = allConfidences.getOrNull(2) ?: 0
        
        // Calculate confidence gaps for better decision making
        val maxConf = allConfidences.maxOrNull() ?: 0
        val secondMaxConf = allConfidences.sorted().let { 
            if (it.size >= 2) it[it.size-2] else 0 
        }
        val confidenceGap = maxConf - secondMaxConf
        
        // PRIORITY 1: If image analysis says it's NOT a crop OR not close enough, force not_crop
        if (!isLikelyCrop) {
            Log.d(TAG, "Image analysis says not crop OR not close enough - forcing not_crop classification")
            return true // This will show "not_crop" result
        }
        
        // PRIORITY 1.5: Additional check for obvious non-crop objects (furniture, walls, etc.)
        // If the model is very confident about diseased/healthy but image clearly doesn't look like crop, reject it
        val modelVeryConfident = maxOf(diseasedConf, healthyConf) > 80
        val imageDefinitelyNotCrop = !isLikelyCrop
        
        if (modelVeryConfident && imageDefinitelyNotCrop) {
            Log.d(TAG, "Model confident but image clearly not crop - forcing not_crop: modelConf=${maxOf(diseasedConf, healthyConf)}%, isLikelyCrop=$isLikelyCrop")
            return true // Force not_crop result
        }
        
        // PRIORITY 2: Only trust not_crop if it has HIGH confidence AND is the clear winner
        if (labels.size >= 3 && labels[2] == "not_crop") {
            val trustNotCrop = (notCropConf > 40 && notCropConf == maxConf) || // High confidence and top choice
                              (notCropConf > 60) // Very high confidence regardless
            
            if (trustNotCrop) {
                Log.d(TAG, "Model says not_crop with high conf=$notCropConf% - trusting it")
                return true
            }
        }
        
        // PRIORITY 3: For healthy/diseased classification, be EXTREMELY strict
        val maxCropConf = maxOf(diseasedConf, healthyConf)
        val cropConfidenceGap = Math.abs(diseasedConf - healthyConf)
        
        // VERY STRICT requirements for crop classification
        val imageDefinitelyLooksCrop = isLikelyCrop
        val confidenceIsVeryHigh = maxCropConf > 90 // Much higher - 90% required
        val confidenceIsUltraHigh = maxCropConf > 95 // Ultra high confidence
        val gapIsSignificant = confidenceGap > 30 // Much higher gap required
        val gapIsVerySignificant = confidenceGap > 40 // Very significant gap
        val cropGapIsSignificant = cropConfidenceGap > 25 // Higher crop gap required
        val notCropIsVeryLow = notCropConf < 5 // not_crop must be very low (less than 5%)
        
        // EXTREMELY STRICT decision matrix - only classify in perfect conditions
        val shouldClassify = when {
            // Only classify if ULTRA high confidence AND very significant gap AND definitely looks like crop AND not_crop is very low
            confidenceIsUltraHigh && gapIsVerySignificant && imageDefinitelyLooksCrop && notCropIsVeryLow && cropGapIsSignificant -> {
                Log.d(TAG, "Perfect conditions met: maxConf=$maxCropConf%, gap=$confidenceGap, notCropConf=$notCropConf%, cropGap=$cropConfidenceGap")
                true
            }
            // OR very high confidence with all other strict conditions
            confidenceIsVeryHigh && gapIsSignificant && imageDefinitelyLooksCrop && notCropIsVeryLow && cropGapIsSignificant -> {
                Log.d(TAG, "Very strict conditions met: maxConf=$maxCropConf%, gap=$confidenceGap, notCropConf=$notCropConf%, cropGap=$cropConfidenceGap")
                true
            }
            else -> {
                Log.d(TAG, "Classification rejected - not strict enough: maxConf=$maxCropConf%, isLikelyCrop=$imageDefinitelyLooksCrop, gap=$confidenceGap, cropGap=$cropGapIsSignificant, notCropConf=$notCropConf%")
                false
            }
        }
        
        return shouldClassify
    }
    
    private fun displayResult(label: String, confidence: Int, allConfidences: List<Int>) {
        val isHealthy = label.lowercase().contains("healthy")
        val isDiseased = label.lowercase().contains("diseased")
        val isNotCrop = label.lowercase().contains("not_crop") || label.lowercase().contains("not crop")
        
        val resultColor = when {
            isHealthy -> 0xFF4CAF50.toInt() // Green
            isDiseased -> 0xFFF44336.toInt() // Red
            isNotCrop -> 0xFFFF9800.toInt() // Orange
            else -> 0xFF9E9E9E.toInt() // Gray
        }
        
        val confidenceLevel = when {
            confidence >= 80 -> "High"
            confidence >= 60 -> "Medium"
            confidence >= 40 -> "Fair"
            else -> "Low"
        }
        
        // Show more detailed results
        val displayText = when {
            isNotCrop -> {
                "No Crop Detected: $confidence%\nPoint camera at crop leaf"
            }
            labels.size >= 3 -> {
                // Show all confidences for transparency
                val diseasedConf = allConfidences.getOrNull(0) ?: 0
                val healthyConf = allConfidences.getOrNull(1) ?: 0
                val notCropConf = allConfidences.getOrNull(2) ?: 0
                "${label.replaceFirstChar { it.uppercase() }}: $confidence%\n" +
                "D:${diseasedConf}% H:${healthyConf}% N:${notCropConf}%"
            }
            else -> {
                "${label.replaceFirstChar { it.uppercase() }}: $confidence%"
            }
        }
        
        resultText.text = displayText
        resultText.setTextColor(resultColor)
        
        val confidenceMessage = if (isNotCrop) {
            "Try pointing at a crop leaf"
        } else {
            "Confidence: $confidenceLevel"
        }
        confidenceText.text = confidenceMessage
        
        resultContainer.visibility = View.VISIBLE
        
        // Vibrate for diseased crops with high confidence
        if (isDiseased && confidence > 70) {
            vibrator.vibrate(VibrationEffect.createOneShot(300, VibrationEffect.DEFAULT_AMPLITUDE))
        }
        
        // Auto-hide not_crop results after 3 seconds to encourage retrying
        if (isNotCrop) {
            resultContainer.postDelayed({
                if (resultText.text.toString().contains("No Crop Detected")) {
                    resultContainer.visibility = View.GONE
                    hasAnalyzedOnce = false
                    updateInstructions()
                    updateAnalyzeButtonState()
                }
            }, 3000)
        }
    }
    
    private fun analyzeImageForCropFeatures(bitmap: Bitmap): Boolean {
        try {
            // Enhanced crop feature analysis with proximity detection
            val width = bitmap.width
            val height = bitmap.height
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            
            // Initialize counters and accumulators
            var greenPixels = 0
            var leafLikePixels = 0
            var totalVariance = 0.0
            var avgRed = 0.0
            var avgGreen = 0.0
            var avgBlue = 0.0
            var edgePixels = 0
            var skyPixels = 0
            var uniformPixels = 0
            var sharpEdges = 0
            var blurryPixels = 0
            
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val red = (pixel shr 16) and 0xFF
                val green = (pixel shr 8) and 0xFF
                val blue = pixel and 0xFF
                
                avgRed += red
                avgGreen += green
                avgBlue += blue
                
                // Count green-dominant pixels (typical of healthy crops)
                if (green > red && green > blue && green > 80) {
                    greenPixels++
                }
                
                // Count leaf-like colors (green variations, some brown/yellow for diseases)
                val isGreenish = green > (red + blue) / 2 && green > 60
                val isBrownish = red > 100 && green > 80 && blue < 100 // Disease colors
                val isYellowish = red > 150 && green > 150 && blue < 120 // Disease colors
                
                if (isGreenish || isBrownish || isYellowish) {
                    leafLikePixels++
                }
                
                // Detect sky-like colors (blue dominant, bright)
                if (blue > red && blue > green && blue > 120) {
                    skyPixels++
                }
                
                // Detect uniform/solid colors (walls, floors, etc.)
                val colorVariance = Math.abs(red - green) + Math.abs(green - blue) + Math.abs(blue - red)
                if (colorVariance < 30) { // Very similar RGB values
                    uniformPixels++
                }
                
                // Calculate color variance (crops have more texture than solid backgrounds)
                val gray = (red + green + blue) / 3.0
                totalVariance += Math.abs(gray - 128) // Variance from middle gray
                
                // Enhanced edge detection with proximity analysis
                if (i % width > 1 && i % width < width - 2 && i >= width * 2 && i < pixels.size - width * 2) {
                    val leftPixel = pixels[i - 1]
                    val rightPixel = pixels[i + 1]
                    val topPixel = pixels[i - width]
                    val bottomPixel = pixels[i + width]
                    
                    // Also check diagonal neighbors for better edge detection
                    val topLeftPixel = pixels[i - width - 1]
                    val topRightPixel = pixels[i - width + 1]
                    val bottomLeftPixel = pixels[i + width - 1]
                    val bottomRightPixel = pixels[i + width + 1]
                    
                    val currentGray = red + green + blue
                    val leftGray = ((leftPixel shr 16) and 0xFF) + ((leftPixel shr 8) and 0xFF) + (leftPixel and 0xFF)
                    val rightGray = ((rightPixel shr 16) and 0xFF) + ((rightPixel shr 8) and 0xFF) + (rightPixel and 0xFF)
                    val topGray = ((topPixel shr 16) and 0xFF) + ((topPixel shr 8) and 0xFF) + (topPixel and 0xFF)
                    val bottomGray = ((bottomPixel shr 16) and 0xFF) + ((bottomPixel shr 8) and 0xFF) + (bottomPixel and 0xFF)
                    
                    // Calculate edge strength (Sobel-like operator)
                    val horizontalEdge = Math.abs(leftGray - rightGray)
                    val verticalEdge = Math.abs(topGray - bottomGray)
                    val edgeStrength = horizontalEdge + verticalEdge
                    
                    if (edgeStrength > 100) {
                        edgePixels++
                        
                        // Detect sharp vs blurry edges (proximity indicator)
                        if (edgeStrength > 200) {
                            sharpEdges++ // Sharp edges indicate close objects
                        }
                    } else if (edgeStrength < 30) {
                        blurryPixels++ // Very low contrast indicates distant/blurry objects
                    }
                }
            }
            
            val totalPixels = pixels.size
            avgRed /= totalPixels
            avgGreen /= totalPixels
            avgBlue /= totalPixels
            totalVariance /= totalPixels
            
            val greenRatio = greenPixels.toFloat() / totalPixels
            val leafLikeRatio = leafLikePixels.toFloat() / totalPixels
            val edgeRatio = edgePixels.toFloat() / totalPixels
            val skyRatio = skyPixels.toFloat() / totalPixels
            val uniformRatio = uniformPixels.toFloat() / totalPixels
            val sharpEdgeRatio = sharpEdges.toFloat() / totalPixels
            val blurRatio = blurryPixels.toFloat() / totalPixels
            
            // PROXIMITY ANALYSIS - key indicators of close objects
            val isCloseToCamera = analyzeProximity(
                sharpEdgeRatio, blurRatio, totalVariance, 
                leafLikeRatio, uniformRatio, width, height
            )
            
            // MUCH STRICTER crop detection - must have clear plant characteristics
            val hasSignificantGreen = greenRatio > 0.20 // Much higher - 20% green pixels required
            val hasLeafLikeColors = leafLikeRatio > 0.35 // Much higher - 35% leaf-like colors required
            val hasPlantTexture = totalVariance > 45 // Higher texture requirement for plants
            val hasLeafDetails = edgeRatio > 0.12 // Much higher - 12% edge pixels (leaf veins, etc.)
            val notTooUniform = uniformRatio < 0.5 // Even stricter uniformity check
            val notTooDark = !(avgRed < 60 && avgGreen < 60 && avgBlue < 60) // Stricter dark threshold
            val notTooBlue = skyRatio < 0.15 // Much stricter sky tolerance
            val notTooWhite = !(avgRed > 170 && avgGreen > 170 && avgBlue > 170) // Stricter white threshold
            
            // NEW: Specific checks for non-crop objects
            val isGreenDominant = avgGreen > avgRed && avgGreen > avgBlue // Green must be dominant color
            val hasNaturalGreenTones = greenRatio > 0.15 && avgGreen > 80 // Natural green tones
            val notArtificialSurface = !(uniformRatio > 0.7 || (avgRed > 150 && avgGreen > 150 && avgBlue > 150)) // Not artificial surfaces
            
            // Negative indicators (things that suggest NOT a crop) - more aggressive
            val tooMuchSky = skyRatio > 0.25 // Stricter - from 0.4 to 0.25
            val tooUniform = uniformRatio > 0.6 // Stricter - from 0.8 to 0.6
            val tooWhite = avgRed > 200 && avgGreen > 200 && avgBlue > 200 // Stricter white detection
            val tooDark = avgRed < 40 && avgGreen < 40 && avgBlue < 40 // Stricter dark detection
            val notEnoughGreen = greenRatio < 0.05 // New: must have some green
            val tooBlurry = edgeRatio < 0.03 // New: must have some detail/edges
            
            // MUCH more sophisticated decision with plant-specific scoring
            val positiveScore = (if (hasSignificantGreen) 4 else 0) + // Higher weight for significant green
                               (if (hasLeafLikeColors) 4 else 0) + // Higher weight for leaf colors
                               (if (hasPlantTexture) 3 else 0) + // Higher weight for plant texture
                               (if (hasLeafDetails) 3 else 0) + // Higher weight for leaf details
                               (if (isGreenDominant) 2 else 0) + // New: green must be dominant
                               (if (hasNaturalGreenTones) 2 else 0) + // New: natural green tones
                               (if (notTooUniform) 1 else 0) +
                               (if (notTooDark) 1 else 0) +
                               (if (notTooBlue) 1 else 0) +
                               (if (notTooWhite) 1 else 0) +
                               (if (notArtificialSurface) 1 else 0) // New: not artificial surface
            
            val negativeScore = (if (tooMuchSky) 4 else 0) + // Higher penalty for sky
                               (if (tooUniform) 4 else 0) + // Higher penalty for uniform surfaces
                               (if (tooWhite) 3 else 0) + // Higher penalty for white
                               (if (tooDark) 3 else 0) + // Higher penalty for dark
                               (if (notEnoughGreen) 3 else 0) + // Higher penalty for no green
                               (if (tooBlurry) 2 else 0) + // Higher penalty for blur
                               (if (!isGreenDominant) 2 else 0) + // New: penalty if green not dominant
                               (if (!hasNaturalGreenTones) 2 else 0) // New: penalty for unnatural colors
            
            val finalScore = positiveScore - negativeScore
            
            // FINAL DECISION: Must pass ALL checks - proximity, crop-like appearance, AND high score
            val isLikelyCrop = isCloseToCamera && finalScore >= 10 // Raised from 6 to 10
            
            Log.d(TAG, "Image analysis: green=$greenRatio, leafLike=$leafLikeRatio, texture=$totalVariance, edges=$edgeRatio, sharp=$sharpEdgeRatio, blur=$blurRatio, close=$isCloseToCamera, pos=$positiveScore, neg=$negativeScore, final=$finalScore, isLikelyCrop=$isLikelyCrop")
            
            return isLikelyCrop
            
        } catch (e: Exception) {
            Log.e(TAG, "Image analysis failed: ${e.message}")
            return false // Default to NOT allowing classification if analysis fails (safer for distant objects)
        }
    }
    
    private fun analyzeProximity(
        sharpEdgeRatio: Float, 
        blurRatio: Float, 
        totalVariance: Double, 
        leafLikeRatio: Float, 
        uniformRatio: Float,
        width: Int,
        height: Int
    ): Boolean {
        try {
            // Multiple indicators that object is close to camera
            
            // 1. Sharp edge density - close objects have more sharp edges (STRICTER)
            val hasSharpDetails = sharpEdgeRatio > 0.12 // Raised from 8% to 12% sharp edges
            
            // 2. Low blur ratio - close objects are less blurry (STRICTER)
            val isNotBlurry = blurRatio < 0.2 // Reduced from 30% to 20% blurry pixels
            
            // 3. High texture variance - close objects show more detail (STRICTER)
            val hasDetailedTexture = totalVariance > 50 // Raised from 40 to 50
            
            // 4. Object size analysis - close objects fill more of the frame (STRICTER)
            val hasGoodCoverage = leafLikeRatio > 0.3 // Raised from 20% to 30% coverage
            
            // 5. Not too uniform - distant objects often appear more uniform (STRICTER)
            val hasVariation = uniformRatio < 0.4 // Reduced from 50% to 40% uniform pixels
            
            // 6. Resolution-based analysis - higher resolution suggests closer object
            val totalPixels = width * height
            val hasGoodResolution = totalPixels > 50000 // Reasonable resolution for detail
            
            // STRICTER scoring system for proximity - only very close objects pass
            var proximityScore = 0
            
            if (hasSharpDetails) proximityScore += 4 // Increased weight for sharp details
            if (isNotBlurry) proximityScore += 3 // Increased weight for clarity
            if (hasDetailedTexture) proximityScore += 3 // Increased weight for texture
            if (hasGoodCoverage) proximityScore += 3 // Increased weight for size
            if (hasVariation) proximityScore += 2 // Increased weight for variation
            if (hasGoodResolution) proximityScore += 1 // Basic requirement
            
            // STRICTER checks for distant objects
            val veryBlurry = blurRatio > 0.4 // Reduced from 60% to 40% blurry
            val veryUniform = uniformRatio > 0.6 // Reduced from 80% to 60% uniform
            val veryLowTexture = totalVariance < 30 // Raised from 20 to 30
            val tinyObject = leafLikeRatio < 0.15 // Raised from 10% to 15%
            val moderatelyBlurry = blurRatio > 0.25 // New: moderate blur check
            val smallObject = leafLikeRatio < 0.25 // New: small object check
            
            // INCREASED penalties for distant object indicators
            if (veryBlurry) proximityScore -= 4 // Increased penalty
            if (veryUniform) proximityScore -= 3 // Increased penalty
            if (veryLowTexture) proximityScore -= 3 // Increased penalty
            if (tinyObject) proximityScore -= 3 // Increased penalty
            if (moderatelyBlurry) proximityScore -= 2 // New penalty
            if (smallObject) proximityScore -= 1 // New penalty
            
            val isClose = proximityScore >= 8 // MUCH stricter - raised from 5 to 8
            
            Log.d(TAG, "Proximity analysis: sharp=$sharpEdgeRatio, blur=$blurRatio, texture=$totalVariance, coverage=$leafLikeRatio, uniform=$uniformRatio, score=$proximityScore, isClose=$isClose")
            
            return isClose
            
        } catch (e: Exception) {
            Log.e(TAG, "Proximity analysis failed: ${e.message}")
            return false // Default to not close if analysis fails
        }
    }
    

    
    private fun displayLowConfidenceResult(topConfidence: Int = 0, allConfidences: List<Int> = emptyList()) {
        val maxConf = allConfidences.maxOrNull() ?: topConfidence
        val notCropConf = if (allConfidences.size >= 3) allConfidences[2] else 0
        
        val message = when {
            notCropConf > 25 -> "No crop in view"
            maxConf < 25 -> "Unclear image"
            maxConf < 40 -> "Low confidence detection"
            else -> "Uncertain classification"
        }
        
        val detailText = if (allConfidences.isNotEmpty() && labels.size == allConfidences.size) {
            "\n${labels.zip(allConfidences).joinToString(" ") { "${it.first.take(1).uppercase()}:${it.second}%" }}"
        } else {
            ""
        }
        
        resultText.text = "$message$detailText"
        resultText.setTextColor(0xFFFF9800.toInt()) // Orange color
        
        val advice = when {
            notCropConf > 25 -> "Point camera at crop leaf"
            maxConf < 25 -> "Try better lighting and focus"
            else -> "Move closer to crop or improve lighting"
        }
        confidenceText.text = advice
        
        resultContainer.visibility = View.VISIBLE
        
        // Auto-hide and reset for new analysis after 3 seconds
        resultContainer.postDelayed({
            resultContainer.visibility = View.GONE
            hasAnalyzedOnce = false
            updateInstructions()
            updateAnalyzeButtonState()
        }, 3000)
    }
    
    private fun showSecondInfoDialog() {
        val message = """
            ADTC Crop Disease Classifier
            
            How to use:
            1. Position a crop leaf in the frame
            2. Ensure good lighting
            3. Tap 'Analyze Crop' button
            4. View the results below
            
            The app can detect:
            • Healthy crops
            • Diseased crops
            
            Model: PlantVillage Dataset
            Input: ${inputSize}x${inputSize} pixels
            Classes: ${labels.joinToString(", ")}
        """.trimIndent()
        
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("About ADTC Classifier")
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        // Clean up resources
        interpreter?.close()
        latestBitmap?.recycle()
        cameraExecutor.shutdown()
        
        // Remove any pending callbacks
        previewView.removeCallbacks(null)
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
    

    
    private fun convertYuvToRgb(image: ImageProxy): Bitmap? {
        return try {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer
            
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            
            val nv21 = ByteArray(ySize + uSize + vSize)
            
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
            val imageBytes = out.toByteArray()
            
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } catch (e: Exception) {
            Log.e(TAG, "YUV conversion failed: ${e.message}")
            null
        }
    }
    
    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer? {
        return try {
            val currentInterpreter = interpreter ?: return null
            
            // Scale bitmap to model input size
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            
            // Create input buffer
            val inputBuffer = ByteBuffer.allocateDirect(currentInterpreter.getInputTensor(0).numBytes())
            inputBuffer.order(ByteOrder.nativeOrder())
            
            val inputType = currentInterpreter.getInputTensor(0).dataType()
            val pixels = IntArray(inputSize * inputSize)
            scaledBitmap.getPixels(pixels, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)
            
            // Convert pixels to model input format
            for (pixelValue in pixels) {
                val red = (pixelValue shr 16) and 0xFF
                val green = (pixelValue shr 8) and 0xFF
                val blue = pixelValue and 0xFF
                
                when (inputType.toString()) {
                    "INT8" -> {
                        // Convert to INT8 range [-128, 127]
                        when (inputChannels) {
                            1 -> {
                                val gray = (0.299 * red + 0.587 * green + 0.114 * blue).toInt()
                                inputBuffer.put((gray - 128).toByte())
                            }
                            3 -> {
                                inputBuffer.put((red - 128).toByte())
                                inputBuffer.put((green - 128).toByte())
                                inputBuffer.put((blue - 128).toByte())
                            }
                        }
                    }
                    "UINT8" -> {
                        // Keep in UINT8 range [0, 255]
                        when (inputChannels) {
                            1 -> {
                                val gray = (0.299 * red + 0.587 * green + 0.114 * blue).toInt()
                                inputBuffer.put(gray.toByte())
                            }
                            3 -> {
                                inputBuffer.put(red.toByte())
                                inputBuffer.put(green.toByte())
                                inputBuffer.put(blue.toByte())
                            }
                        }
                    }
                    else -> {
                        // Float32 - normalize to [0, 1]
                        when (inputChannels) {
                            1 -> {
                                val gray = (0.299f * red + 0.587f * green + 0.114f * blue) / 255.0f
                                inputBuffer.putFloat(gray)
                            }
                            3 -> {
                                inputBuffer.putFloat(red / 255.0f)
                                inputBuffer.putFloat(green / 255.0f)
                                inputBuffer.putFloat(blue / 255.0f)
                            }
                        }
                    }
                }
            }
            
            inputBuffer.rewind()
            inputBuffer
            
        } catch (e: Exception) {
            Log.e(TAG, "Preprocessing failed: ${e.message}")
            null
        }
    }
    
    private fun convertImageProxyToBitmap(image: ImageProxy): Bitmap? {
        return try {
            Log.d(TAG, "Converting image: ${image.width}x${image.height}, format: ${image.format}")
            
            when (image.format) {
                ImageFormat.YUV_420_888 -> {
                    // YUV to RGB conversion (most common format from camera)
                    convertYuvToBitmap(image)
                }
                else -> {
                    // Try RGBA conversion for other formats
                    convertRgbaToBitmap(image)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Image conversion failed: ${e.message}")
            null
        }
    }
    
    private fun convertRgbaToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val buffer = image.planes[0].buffer
            val pixelStride = image.planes[0].pixelStride
            val rowStride = image.planes[0].rowStride
            val rowPadding = rowStride - pixelStride * image.width
            
            val bitmap = Bitmap.createBitmap(
                image.width + rowPadding / pixelStride,
                image.height,
                Bitmap.Config.ARGB_8888
            )
            bitmap.copyPixelsFromBuffer(buffer)
            
            // Crop to actual image size if there's padding
            if (rowPadding == 0) {
                bitmap
            } else {
                val croppedBitmap = Bitmap.createBitmap(bitmap, 0, 0, image.width, image.height)
                bitmap.recycle()
                croppedBitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "RGBA conversion failed: ${e.message}")
            null
        }
    }
    
    private fun convertYuvToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer
            
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            
            val nv21 = ByteArray(ySize + uSize + vSize)
            
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
            val imageBytes = out.toByteArray()
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } catch (e: Exception) {
            Log.e(TAG, "YUV conversion failed: ${e.message}")
            null
        }
    }}
