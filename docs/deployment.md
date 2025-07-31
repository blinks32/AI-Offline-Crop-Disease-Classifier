# Deployment Guide

## Overview

This guide covers the complete deployment process for the ADTC Smart Crop Disease Classifier, from development builds to production distribution across multiple channels.

## Deployment Architecture

```
Development → Testing → Staging → Production
     ↓           ↓         ↓          ↓
Local Build → CI/CD → Beta Release → Store Release
```

## 1. Build Configuration

### Gradle Build Setup

#### `app/build.gradle.kts`

```kotlin
android {
    compileSdk 34
    
    defaultConfig {
        applicationId "com.adtc.cropdiseaseclassifier"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "1.0.0"
        
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }
    
    buildTypes {
        debug {
            isDebuggable = true
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            isMinifyEnabled = false
        }
        
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            
            // Signing configuration
            signingConfig = signingConfigs.getByName("release")
        }
        
        create("staging") {
            initWith(getByName("release"))
            applicationIdSuffix = ".staging"
            versionNameSuffix = "-staging"
            isDebuggable = true
        }
    }
    
    // Build variants for different markets
    flavorDimensions += "market"
    productFlavors {
        create("global") {
            dimension = "market"
            // Global market configuration
        }
        
        create("india") {
            dimension = "market"
            applicationIdSuffix = ".india"
            // India-specific configuration
        }
        
        create("africa") {
            dimension = "market"
            applicationIdSuffix = ".africa"
            // Africa-specific configuration
        }
    }
}
```

### ProGuard Configuration

#### `proguard-rules.pro`

```proguard
# Keep TensorFlow Lite classes
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.support.** { *; }

# Keep model-related classes
-keep class * extends org.tensorflow.lite.support.model.Model { *; }

# Keep camera classes
-keep class androidx.camera.** { *; }

# Keep application classes
-keep class com.adtc.cropdiseaseclassifier.** { *; }

# Remove logging in release builds
-assumenosideeffects class android.util.Log {
    public static boolean isLoggable(java.lang.String, int);
    public static int v(...);
    public static int i(...);
    public static int w(...);
    public static int d(...);
    public static int e(...);
}
```

## 2. Signing Configuration

### Keystore Generation

```bash
# Generate release keystore
keytool -genkey -v -keystore adtc-release-key.keystore \
    -alias adtc-key -keyalg RSA -keysize 2048 -validity 10000

# Generate upload keystore for Play Store
keytool -genkey -v -keystore adtc-upload-key.keystore \
    -alias adtc-upload -keyalg RSA -keysize 2048 -validity 10000
```

### Signing Configuration

#### `gradle.properties` (local)

```properties
# Signing configuration
ADTC_RELEASE_STORE_FILE=../keystores/adtc-release-key.keystore
ADTC_RELEASE_STORE_PASSWORD=your_store_password
ADTC_RELEASE_KEY_ALIAS=adtc-key
ADTC_RELEASE_KEY_PASSWORD=your_key_password

ADTC_UPLOAD_STORE_FILE=../keystores/adtc-upload-key.keystore
ADTC_UPLOAD_STORE_PASSWORD=your_upload_store_password
ADTC_UPLOAD_KEY_ALIAS=adtc-upload
ADTC_UPLOAD_KEY_PASSWORD=your_upload_key_password
```

#### Build script signing

```kotlin
android {
    signingConfigs {
        create("release") {
            storeFile = file(project.findProperty("ADTC_RELEASE_STORE_FILE") as String)
            storePassword = project.findProperty("ADTC_RELEASE_STORE_PASSWORD") as String
            keyAlias = project.findProperty("ADTC_RELEASE_KEY_ALIAS") as String
            keyPassword = project.findProperty("ADTC_RELEASE_KEY_PASSWORD") as String
        }
        
        create("upload") {
            storeFile = file(project.findProperty("ADTC_UPLOAD_STORE_FILE") as String)
            storePassword = project.findProperty("ADTC_UPLOAD_STORE_PASSWORD") as String
            keyAlias = project.findProperty("ADTC_UPLOAD_KEY_ALIAS") as String
            keyPassword = project.findProperty("ADTC_UPLOAD_KEY_PASSWORD") as String
        }
    }
}
```

## 3. CI/CD Pipeline

### GitHub Actions Workflow

#### `.github/workflows/android.yml`

```yaml
name: Android CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up JDK 17
      uses: actions/setup-java@v3
      with:
        java-version: '17'
        distribution: 'temurin'
    
    - name: Cache Gradle packages
      uses: actions/cache@v3
      with:
        path: |
          ~/.gradle/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
        restore-keys: |
          ${{ runner.os }}-gradle-
    
    - name: Grant execute permission for gradlew
      run: chmod +x gradlew
    
    - name: Run tests
      run: ./gradlew test
    
    - name: Run lint
      run: ./gradlew lint
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: app/build/reports/tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up JDK 17
      uses: actions/setup-java@v3
      with:
        java-version: '17'
        distribution: 'temurin'
    
    - name: Cache Gradle packages
      uses: actions/cache@v3
      with:
        path: |
          ~/.gradle/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
    
    - name: Build debug APK
      run: ./gradlew assembleDebug
    
    - name: Upload debug APK
      uses: actions/upload-artifact@v3
      with:
        name: debug-apk
        path: app/build/outputs/apk/debug/app-debug.apk

  release:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up JDK 17
      uses: actions/setup-java@v3
      with:
        java-version: '17'
        distribution: 'temurin'
    
    - name: Decode keystore
      env:
        ENCODED_STRING: ${{ secrets.KEYSTORE_BASE64 }}
      run: |
        echo $ENCODED_STRING | base64 -di > app/keystore.jks
    
    - name: Build release APK
      env:
        SIGNING_KEY_ALIAS: ${{ secrets.SIGNING_KEY_ALIAS }}
        SIGNING_KEY_PASSWORD: ${{ secrets.SIGNING_KEY_PASSWORD }}
        SIGNING_STORE_PASSWORD: ${{ secrets.SIGNING_STORE_PASSWORD }}
      run: ./gradlew assembleRelease
    
    - name: Build release AAB
      env:
        SIGNING_KEY_ALIAS: ${{ secrets.SIGNING_KEY_ALIAS }}
        SIGNING_KEY_PASSWORD: ${{ secrets.SIGNING_KEY_PASSWORD }}
        SIGNING_STORE_PASSWORD: ${{ secrets.SIGNING_STORE_PASSWORD }}
      run: ./gradlew bundleRelease
    
    - name: Upload release artifacts
      uses: actions/upload-artifact@v3
      with:
        name: release-artifacts
        path: |
          app/build/outputs/apk/release/app-release.apk
          app/build/outputs/bundle/release/app-release.aab
```

### Fastlane Configuration

#### `fastlane/Fastfile`

```ruby
default_platform(:android)

platform :android do
  desc "Runs all the tests"
  lane :test do
    gradle(task: "test")
  end

  desc "Build debug APK"
  lane :debug do
    gradle(task: "clean assembleDebug")
  end

  desc "Build release APK"
  lane :release do
    gradle(task: "clean assembleRelease")
  end

  desc "Deploy to Play Store Internal Testing"
  lane :internal do
    gradle(task: "clean bundleRelease")
    upload_to_play_store(
      track: 'internal',
      aab: 'app/build/outputs/bundle/release/app-release.aab'
    )
  end

  desc "Deploy to Play Store Beta"
  lane :beta do
    gradle(task: "clean bundleRelease")
    upload_to_play_store(
      track: 'beta',
      aab: 'app/build/outputs/bundle/release/app-release.aab'
    )
  end

  desc "Deploy to Play Store Production"
  lane :production do
    gradle(task: "clean bundleRelease")
    upload_to_play_store(
      track: 'production',
      aab: 'app/build/outputs/bundle/release/app-release.aab'
    )
  end
end
```

## 4. Google Play Store Deployment

### App Bundle Preparation

```bash
# Build release AAB
./gradlew bundleRelease

# Verify AAB contents
bundletool build-apks --bundle=app/build/outputs/bundle/release/app-release.aab \
  --output=app.apks

bundletool get-size total --apks=app.apks
```

### Play Console Configuration

#### Store Listing

```yaml
# store_listing.yml
title: "ADTC Smart Crop Disease Classifier"
short_description: "AI-powered crop disease detection for farmers"
full_description: |
  Instantly detect crop diseases using your smartphone camera. 
  Our AI-powered app helps farmers identify plant health issues 
  quickly and accurately, enabling early intervention and better yields.
  
  Features:
  • One-tap disease detection
  • Works offline - no internet required
  • Supports 10+ crop types
  • 85-95% accuracy rate
  • Professional agricultural guidance

category: "PRODUCTIVITY"
content_rating: "EVERYONE"
website: "https://your-website.com"
email: "support@your-domain.com"
phone: "+1-xxx-xxx-xxxx"
privacy_policy: "https://your-website.com/privacy"
```

#### Release Notes Template

```markdown
# Version 1.0.0 - Initial Release

## New Features
- AI-powered crop disease detection
- Support for 10+ crop species
- Offline operation capability
- Professional ADTC branding
- Intuitive one-tap interface

## Supported Crops
- Apple, Tomato, Corn, Potato
- Grape, Pepper, Cherry, Peach
- Strawberry, Squash

## Technical Improvements
- Optimized for low-end devices
- Fast inference (<1 second)
- Minimal battery usage
- Compact 1.7MB AI model

## Bug Fixes
- Initial stable release

---
For support, visit: https://your-website.com/support
```

### Staged Rollout Strategy

```yaml
# Rollout phases
internal_testing:
  duration: "1 week"
  users: "Development team + agricultural experts"
  focus: "Core functionality validation"

closed_testing:
  duration: "2 weeks"
  users: "100 beta testers"
  focus: "Real-world usage testing"

open_testing:
  duration: "2 weeks"
  users: "1000 public beta users"
  focus: "Performance and usability"

production:
  rollout: "Staged (20% → 50% → 100%)"
  duration: "1 week per stage"
  monitoring: "Crash rates, ANRs, user feedback"
```

## 5. Alternative Distribution

### Direct APK Distribution

#### APK Hosting Setup

```bash
# Create distribution directory
mkdir -p distribution/apk
mkdir -p distribution/metadata

# Copy APK and metadata
cp app/build/outputs/apk/release/app-release.apk distribution/apk/
cp app/build/outputs/mapping/release/mapping.txt distribution/metadata/

# Generate checksums
cd distribution/apk
sha256sum app-release.apk > app-release.apk.sha256
md5sum app-release.apk > app-release.apk.md5
```

#### Distribution Website

```html
<!DOCTYPE html>
<html>
<head>
    <title>ADTC Crop Disease Classifier - Download</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
    <h1>ADTC Smart Crop Disease Classifier</h1>
    <p>AI-powered crop disease detection for farmers</p>
    
    <div class="download-section">
        <h2>Download Options</h2>
        
        <div class="download-option">
            <h3>Google Play Store (Recommended)</h3>
            <a href="https://play.google.com/store/apps/details?id=com.adtc.cropdiseaseclassifier">
                <img src="google-play-badge.png" alt="Get it on Google Play">
            </a>
        </div>
        
        <div class="download-option">
            <h3>Direct APK Download</h3>
            <p>For regions without Google Play Store access</p>
            <a href="apk/app-release.apk" class="download-btn">
                Download APK (v1.0.0)
            </a>
            <p class="checksum">
                SHA256: <code id="sha256-hash">...</code><br>
                MD5: <code id="md5-hash">...</code>
            </p>
        </div>
    </div>
    
    <div class="installation-guide">
        <h2>Installation Instructions</h2>
        <ol>
            <li>Enable "Unknown Sources" in Android Settings</li>
            <li>Download the APK file</li>
            <li>Open the downloaded file to install</li>
            <li>Grant camera permissions when prompted</li>
        </ol>
    </div>
</body>
</html>
```

### Enterprise Distribution

#### MDM Integration

```xml
<!-- managed_app_config.xml -->
<managed-app-configuration>
    <app-restrictions>
        <restriction
            android:key="enable_analytics"
            android:restrictionType="bool"
            android:defaultValue="false"
            android:title="Enable Analytics"
            android:description="Allow usage analytics collection" />
        
        <restriction
            android:key="server_url"
            android:restrictionType="string"
            android:defaultValue=""
            android:title="Server URL"
            android:description="Custom server for model updates" />
    </app-restrictions>
</managed-app-configuration>
```

## 6. Monitoring and Analytics

### Crash Reporting Setup

#### Firebase Crashlytics

```kotlin
// Application class
class ADTCApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        
        // Initialize Firebase
        FirebaseApp.initializeApp(this)
        
        // Enable Crashlytics
        FirebaseCrashlytics.getInstance().setCrashlyticsCollectionEnabled(true)
        
        // Set user properties
        FirebaseCrashlytics.getInstance().setUserId("anonymous")
        FirebaseCrashlytics.getInstance().setCustomKey("app_version", BuildConfig.VERSION_NAME)
    }
}
```

### Performance Monitoring

```kotlin
// Performance tracking
class PerformanceTracker {
    private val firebasePerformance = FirebasePerformance.getInstance()
    
    fun trackInference(processingTime: Long, confidence: Float) {
        val trace = firebasePerformance.newTrace("model_inference")
        trace.start()
        
        // Add custom metrics
        trace.putMetric("processing_time_ms", processingTime)
        trace.putMetric("confidence_score", (confidence * 100).toLong())
        
        trace.stop()
    }
    
    fun trackCameraCapture() {
        val trace = firebasePerformance.newTrace("camera_capture")
        trace.start()
        // ... camera operations
        trace.stop()
    }
}
```

### Analytics Dashboard

```kotlin
// Custom analytics events
class AnalyticsManager {
    private val firebaseAnalytics = FirebaseAnalytics.getInstance(context)
    
    fun trackImageAnalysis(cropType: String, diseaseDetected: Boolean, confidence: Float) {
        val bundle = Bundle().apply {
            putString("crop_type", cropType)
            putBoolean("disease_detected", diseaseDetected)
            putDouble("confidence", confidence.toDouble())
        }
        firebaseAnalytics.logEvent("image_analysis", bundle)
    }
    
    fun trackUserEngagement(sessionDuration: Long, imagesAnalyzed: Int) {
        val bundle = Bundle().apply {
            putLong("session_duration", sessionDuration)
            putInt("images_analyzed", imagesAnalyzed)
        }
        firebaseAnalytics.logEvent("user_engagement", bundle)
    }
}
```

## 7. Update Strategy

### Over-the-Air Updates

#### App Updates

```kotlin
// In-app update manager
class UpdateManager(private val activity: Activity) {
    private val appUpdateManager = AppUpdateManagerFactory.create(activity)
    
    fun checkForUpdates() {
        val appUpdateInfoTask = appUpdateManager.appUpdateInfo
        
        appUpdateInfoTask.addOnSuccessListener { appUpdateInfo ->
            if (appUpdateInfo.updateAvailability() == UpdateAvailability.UPDATE_AVAILABLE
                && appUpdateInfo.isUpdateTypeAllowed(AppUpdateType.FLEXIBLE)) {
                
                // Start flexible update
                appUpdateManager.startUpdateFlowForResult(
                    appUpdateInfo,
                    AppUpdateType.FLEXIBLE,
                    activity,
                    UPDATE_REQUEST_CODE
                )
            }
        }
    }
}
```

#### Model Updates

```kotlin
// Model update manager
class ModelUpdateManager {
    fun checkForModelUpdates() {
        // Check server for new model versions
        // Download and validate new models
        // Replace existing model atomically
    }
    
    private fun downloadModel(modelUrl: String, version: String) {
        // Secure model download with integrity checks
    }
    
    private fun validateModel(modelFile: File): Boolean {
        // Validate model format and performance
        return true
    }
}
```

## 8. Rollback Strategy

### Emergency Rollback

```bash
# Rollback script
#!/bin/bash

# Stop current rollout
echo "Stopping current rollout..."
fastlane android halt_rollout

# Rollback to previous version
echo "Rolling back to previous version..."
fastlane android rollback_production

# Notify team
echo "Rollback completed. Notifying team..."
# Send notifications to team
```

### Automated Rollback Triggers

```yaml
# monitoring_rules.yml
rollback_triggers:
  crash_rate:
    threshold: 2.0  # 2% crash rate
    window: "1 hour"
    action: "automatic_rollback"
  
  anr_rate:
    threshold: 1.0  # 1% ANR rate
    window: "2 hours"
    action: "alert_team"
  
  user_rating:
    threshold: 3.0  # Below 3.0 stars
    window: "24 hours"
    action: "manual_review"
```

## 9. Security Considerations

### Code Obfuscation

```proguard
# Additional security rules
-obfuscationdictionary dictionary.txt
-classobfuscationdictionary dictionary.txt
-packageobfuscationdictionary dictionary.txt

# Protect against reverse engineering
-keepattributes !LocalVariableTable,!LocalVariableTypeTable
-renamesourcefileattribute SourceFile
```

### Certificate Pinning

```kotlin
// Network security config
class SecurityConfig {
    fun setupCertificatePinning() {
        val certificatePinner = CertificatePinner.Builder()
            .add("api.adtc.com", "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
            .build()
        
        val client = OkHttpClient.Builder()
            .certificatePinner(certificatePinner)
            .build()
    }
}
```

## 10. Documentation and Support

### Release Documentation

```markdown
# Release Checklist

## Pre-Release
- [ ] All tests passing
- [ ] Code review completed
- [ ] Security scan passed
- [ ] Performance benchmarks met
- [ ] Documentation updated

## Release Process
- [ ] Version number updated
- [ ] Release notes prepared
- [ ] APK/AAB signed and verified
- [ ] Store listing updated
- [ ] Screenshots updated

## Post-Release
- [ ] Monitor crash rates
- [ ] Check user feedback
- [ ] Verify analytics data
- [ ] Update support documentation
```

### Support Infrastructure

```yaml
# support_channels.yml
primary_support:
  email: "support@adtc.com"
  response_time: "24 hours"
  languages: ["English", "Spanish", "Hindi"]

community_support:
  github_issues: "https://github.com/your-org/adtc-app/issues"
  documentation: "https://docs.adtc.com"
  faq: "https://adtc.com/faq"

emergency_contact:
  email: "emergency@adtc.com"
  phone: "+1-xxx-xxx-xxxx"
  escalation: "Critical bugs, security issues"
```

---

This comprehensive deployment guide ensures reliable, secure, and scalable distribution of the ADTC Smart Crop Disease Classifier across multiple channels and markets.