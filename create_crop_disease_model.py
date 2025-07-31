#!/usr/bin/env python3
"""
Complete guide to create a TensorFlow Lite model for crop disease classification
This script is designed to work in Google Colab or Kaggle notebooks
"""

# Step 1: Install and import required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("TensorFlow version:", tf.__version__)

# Step 2: Configuration
CONFIG = {
    'IMAGE_SIZE': (128, 128),  # Match your Android app expectation
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'LEARNING_RATE': 0.001,
    'NUM_CLASSES': 2,  # healthy, diseased
    'MODEL_NAME': 'crop_disease_classifier'
}

def create_model(num_classes=2, input_shape=(128, 128, 3)):
    """
    Create a MobileNetV2-based model optimized for mobile deployment
    """
    # Use MobileNetV2 as base (lightweight, mobile-optimized)
    base_model = MobileNetV2(
        input_shape=input_shape,
        alpha=0.75,  # Width multiplier for smaller model
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

def prepare_data_generators(train_dir, val_dir=None):
    """
    Create data generators for training
    Assumes directory structure:
    train_dir/
        healthy/
            image1.jpg
            image2.jpg
        diseased/
            image1.jpg
            image2.jpg
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 if val_dir is None else 0.0
    )
    
    # Validation data (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=CONFIG['IMAGE_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        subset='training' if val_dir is None else None
    )
    
    if val_dir is None:
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=CONFIG['IMAGE_SIZE'],
            batch_size=CONFIG['BATCH_SIZE'],
            class_mode='categorical',
            subset='validation'
        )
    else:
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=CONFIG['IMAGE_SIZE'],
            batch_size=CONFIG['BATCH_SIZE'],
            class_mode='categorical'
        )
    
    return train_generator, val_generator

def train_model(model, train_generator, val_generator):
    """
    Train the model with transfer learning
    """
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-7
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=CONFIG['EPOCHS'],
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def fine_tune_model(model, train_generator, val_generator):
    """
    Fine-tune the model by unfreezing some layers
    """
    # Unfreeze the top layers of the base model
    base_model = model.layers[1]  # MobileNetV2 layer
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    
    # Freeze all the layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history_fine = model.fit(
        train_generator,
        epochs=5,  # Few epochs for fine-tuning
        validation_data=val_generator,
        verbose=1
    )
    
    return history_fine

def convert_to_tflite(model, model_name='crop_disease_model'):
    """
    Convert Keras model to TensorFlow Lite with optimization
    """
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Use integer quantization for smaller model
    # This requires a representative dataset
    def representative_dataset():
        for _ in range(100):
            # Generate random data that matches your input
            data = np.random.random((1, 128, 128, 3)).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        # Try quantized conversion
        tflite_model = converter.convert()
        print("✅ Quantized model created successfully")
    except Exception as e:
        print(f"⚠️ Quantized conversion failed: {e}")
        print("Falling back to float32 model...")
        
        # Fallback to float32
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        print("✅ Float32 model created successfully")
    
    # Save the model
    tflite_filename = f'{model_name}.tflite'
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)
    
    print(f"📱 Model saved as: {tflite_filename}")
    print(f"📊 Model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)")
    
    return tflite_model, tflite_filename

def test_tflite_model(tflite_filename):
    """
    Test the TensorFlow Lite model
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_filename)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n🔍 Model Details:")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
    
    # Test with random input
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.int8:
        test_input = np.random.randint(-128, 127, input_shape, dtype=np.int8)
    elif input_dtype == np.uint8:
        test_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    else:
        test_input = np.random.random(input_shape).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\n🧪 Test Inference:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output: {output}")
    
    if output_details[0]['dtype'] == np.int8:
        # Convert int8 output to probabilities
        probs = (output.astype(np.float32) + 128) / 255.0
        print(f"Converted probabilities: {probs}")
    
    print("✅ Model test successful!")
    
    return True

# Example usage for Google Colab/Kaggle
def main_colab_example():
    """
    Example workflow for Google Colab or Kaggle
    """
    print("🌱 Crop Disease Classification Model Creator")
    print("=" * 50)
    
    # Step 1: Create model
    print("\n1️⃣ Creating model architecture...")
    model = create_model(
        num_classes=CONFIG['NUM_CLASSES'],
        input_shape=(*CONFIG['IMAGE_SIZE'], 3)
    )
    
    print(f"Model created with input shape: {CONFIG['IMAGE_SIZE']}")
    model.summary()
    
    # Step 2: If you have data, uncomment and modify these lines:
    """
    # Prepare data (modify paths as needed)
    train_dir = '/content/crop_disease_dataset/train'  # Colab path
    val_dir = '/content/crop_disease_dataset/val'      # Optional
    
    train_gen, val_gen = prepare_data_generators(train_dir, val_dir)
    
    # Train model
    print("\n2️⃣ Training model...")
    history = train_model(model, train_gen, val_gen)
    
    # Fine-tune (optional)
    print("\n3️⃣ Fine-tuning model...")
    history_fine = fine_tune_model(model, train_gen, val_gen)
    """
    
    # Step 3: For demo, create a simple trained model
    print("\n2️⃣ Creating demo model with random weights...")
    # Add some dummy training to initialize weights properly
    dummy_x = np.random.random((10, *CONFIG['IMAGE_SIZE'], 3)).astype(np.float32)
    dummy_y = tf.keras.utils.to_categorical(
        np.random.randint(0, CONFIG['NUM_CLASSES'], 10), 
        CONFIG['NUM_CLASSES']
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
    
    # Step 4: Convert to TensorFlow Lite
    print("\n3️⃣ Converting to TensorFlow Lite...")
    tflite_model, tflite_filename = convert_to_tflite(model, CONFIG['MODEL_NAME'])
    
    # Step 5: Test the model
    print("\n4️⃣ Testing TensorFlow Lite model...")
    test_tflite_model(tflite_filename)
    
    print(f"\n✅ Complete! Your model is ready: {tflite_filename}")
    print("\n📋 Next steps:")
    print("1. Download the .tflite file")
    print("2. Replace app/src/main/assets/model.tflite in your Android project")
    print("3. Update labels.txt if needed")
    print("4. Build and test your Android app")
    
    return tflite_filename

if __name__ == "__main__":
    main_colab_example()