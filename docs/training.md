# Model Training Guide

## Overview

This guide covers the complete process of training AI models for crop disease detection, from data preparation to model deployment. Our approach uses transfer learning with MobileNetV2 for mobile-optimized performance.

## Training Pipeline Overview

```
Data Collection → Preprocessing → Model Training → Validation → Optimization → Deployment
```

## 1. Data Preparation

### Dataset Requirements

#### Minimum Data Requirements
- **Images per class**: 1000+ high-quality images
- **Image resolution**: 224x224 pixels minimum
- **File format**: JPG, PNG
- **Lighting conditions**: Varied (natural daylight preferred)
- **Background**: Clean, focused on leaf/plant part

#### Recommended Dataset Structure
```
dataset/
├── train/
│   ├── healthy/
│   │   ├── healthy_001.jpg
│   │   ├── healthy_002.jpg
│   │   └── ...
│   ├── diseased/
│   │   ├── diseased_001.jpg
│   │   ├── diseased_002.jpg
│   │   └── ...
│   └── non_crop/
│       ├── non_crop_001.jpg
│       ├── non_crop_002.jpg
│       └── ...
├── validation/
│   ├── healthy/
│   ├── diseased/
│   └── non_crop/
└── test/
    ├── healthy/
    ├── diseased/
    └── non_crop/
```

### Data Collection Guidelines

#### High-Quality Images
- **Focus**: Sharp, well-focused images
- **Lighting**: Natural daylight, avoid harsh shadows
- **Angle**: Multiple angles of the same condition
- **Background**: Minimal background distractions
- **Resolution**: At least 224x224, preferably higher

#### Disease Documentation
- **Early stages**: Include early disease symptoms
- **Progression**: Document disease progression stages
- **Severity levels**: Mild, moderate, severe cases
- **Variations**: Different manifestations of same disease

#### Data Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
```

## 2. Model Architecture

### Base Architecture: MobileNetV2

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def create_model(num_classes=3, input_shape=(224, 224, 3)):
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
```

### Model Configuration

```python
# Model parameters
INPUT_SIZE = 224
NUM_CLASSES = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# Class labels
CLASS_LABELS = ['healthy', 'diseased', 'non_crop']
```

## 3. Training Process

### Complete Training Script

```python
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

def train_crop_disease_model():
    # Create model
    model = create_model(num_classes=NUM_CLASSES)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_2_accuracy']
    )
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history
```

### Fine-tuning Phase

```python
def fine_tune_model(model, train_generator, validation_generator):
    # Unfreeze top layers of base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    
    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    fine_tune_epochs = 10
    total_epochs = EPOCHS + fine_tune_epochs
    
    history_fine = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=EPOCHS,
        validation_data=validation_generator
    )
    
    return model, history_fine
```

## 4. Model Evaluation

### Evaluation Metrics

```python
def evaluate_model(model, test_generator):
    # Basic evaluation
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Classification Report:")
    print(classification_report(
        true_classes, 
        predicted_classes, 
        target_names=CLASS_LABELS
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:")
    print(cm)
    
    return test_accuracy, predictions
```

### Performance Visualization

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

## 5. Model Optimization

### Quantization for Mobile

```python
def quantize_model(model, representative_dataset):
    # Convert to TensorFlow Lite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for quantization
    converter.representative_dataset = representative_dataset
    
    # Ensure integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert model
    tflite_model = converter.convert()
    
    return tflite_model

def create_representative_dataset(test_generator):
    def representative_data_gen():
        for i in range(100):  # Use 100 samples
            batch = next(test_generator)
            yield [batch[0].astype(np.float32)]
    
    return representative_data_gen
```

### Model Size Comparison

```python
def compare_model_sizes(original_model, tflite_model):
    # Original model size
    original_model.save('temp_model.h5')
    original_size = os.path.getsize('temp_model.h5')
    
    # TFLite model size
    with open('quantized_model.tflite', 'wb') as f:
        f.write(tflite_model)
    tflite_size = os.path.getsize('quantized_model.tflite')
    
    print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Quantized model size: {tflite_size / 1024 / 1024:.2f} MB")
    print(f"Size reduction: {(1 - tflite_size/original_size) * 100:.1f}%")
    
    # Clean up
    os.remove('temp_model.h5')
```

## 6. Google Colab Training

### Complete Colab Notebook Structure

```python
# Cell 1: Setup and imports
!pip install tensorflow matplotlib scikit-learn

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Cell 2: Data upload and preparation
from google.colab import files
import zipfile
import os

# Upload dataset
uploaded = files.upload()

# Extract dataset
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')

# Cell 3: Model creation and training
# [Include complete training code from above]

# Cell 4: Model evaluation and visualization
# [Include evaluation code from above]

# Cell 5: Model quantization and export
# [Include quantization code from above]

# Cell 6: Download trained model
files.download('quantized_model.tflite')
```

### Colab-Specific Optimizations

```python
# Use GPU if available
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found')
else:
    print(f'Found GPU at: {device_name}')

# Enable mixed precision for faster training
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
```

## 7. Advanced Training Techniques

### Class Balancing

```python
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(train_generator):
    # Calculate class weights for imbalanced datasets
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict

# Use in training
class_weights = calculate_class_weights(train_generator)
model.fit(
    train_generator,
    class_weight=class_weights,
    # ... other parameters
)
```

### Custom Loss Functions

```python
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return focal_loss_fixed

# Use focal loss for imbalanced datasets
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=focal_loss(gamma=2, alpha=0.25),
    metrics=['accuracy']
)
```

## 8. Model Validation

### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_model(X, y, k_folds=5):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{k_folds}")
        
        # Create model for this fold
        model = create_model()
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train on fold
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=EPOCHS,
            verbose=0
        )
        
        # Evaluate fold
        val_score = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
        cv_scores.append(val_score)
        print(f"Fold {fold + 1} accuracy: {val_score:.4f}")
    
    print(f"Average CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    return cv_scores
```

## 9. Troubleshooting

### Common Issues and Solutions

#### Low Accuracy
- **Insufficient data**: Collect more training images
- **Poor data quality**: Review and clean dataset
- **Imbalanced classes**: Use class weights or data augmentation
- **Overfitting**: Add dropout, reduce model complexity

#### Slow Training
- **Use GPU**: Enable GPU acceleration in Colab
- **Reduce batch size**: If running out of memory
- **Mixed precision**: Enable for faster training
- **Data pipeline**: Optimize data loading

#### Model Too Large
- **Quantization**: Use INT8 quantization
- **Pruning**: Remove unnecessary parameters
- **Architecture**: Use more efficient base models
- **Knowledge distillation**: Train smaller student model

### Performance Optimization

```python
# Optimize data pipeline
AUTOTUNE = tf.data.AUTOTUNE

def optimize_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Use optimized datasets
train_ds = optimize_dataset(train_ds)
val_ds = optimize_dataset(val_ds)
```

## 10. Deployment Preparation

### Model Export

```python
def export_model_for_android(model, model_name="crop_disease_model"):
    # Save in multiple formats
    
    # 1. SavedModel format
    model.save(f"{model_name}_savedmodel")
    
    # 2. H5 format
    model.save(f"{model_name}.h5")
    
    # 3. TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(f"{model_name}.tflite", "wb") as f:
        f.write(tflite_model)
    
    print(f"Models exported:")
    print(f"- {model_name}_savedmodel/ (SavedModel)")
    print(f"- {model_name}.h5 (Keras H5)")
    print(f"- {model_name}.tflite (TensorFlow Lite)")
```

### Model Metadata

```python
def create_model_metadata(model, accuracy, class_labels):
    metadata = {
        "model_name": "ADTC Crop Disease Classifier",
        "version": "1.0.0",
        "created_date": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "class_labels": class_labels,
        "preprocessing": {
            "resize": [224, 224],
            "normalize": "0-1 range",
            "color_mode": "RGB"
        }
    }
    
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata
```

---

This comprehensive training guide provides everything needed to create, train, and deploy high-quality crop disease detection models for mobile applications.