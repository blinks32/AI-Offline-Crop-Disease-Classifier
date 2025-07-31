#!/usr/bin/env python3
"""
Create a simple TensorFlow Lite model for crop disease classification
This creates a minimal model that takes 224x224x3 images and outputs 2 classes
"""

import tensorflow as tf
import numpy as np

def create_simple_model():
    # Create a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: healthy, diseased
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create some dummy training data
    x_train = np.random.random((10, 224, 224, 3)).astype(np.float32)
    y_train = np.random.randint(0, 2, (10,))
    
    # Train for just 1 epoch to initialize weights
    model.fit(x_train, y_train, epochs=1, verbose=0)
    
    return model

def convert_to_tflite(model, output_path):
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model saved to {output_path}")
    print(f"Model size: {len(tflite_model)} bytes")
    
    return tflite_model

if __name__ == "__main__":
    print("Creating simple crop disease classification model...")
    
    # Create the model
    model = create_simple_model()
    print("Model created successfully")
    
    # Convert to TensorFlow Lite
    tflite_model = convert_to_tflite(model, "simple_model.tflite")
    
    # Test the model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nModel details:")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
    
    # Test with dummy input
    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nTest inference successful!")
    print(f"Output: {output}")
    print(f"Predicted class: {np.argmax(output[0])}")
    print(f"Confidence: {np.max(output[0]):.2f}")