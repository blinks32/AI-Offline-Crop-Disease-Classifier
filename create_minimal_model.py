#!/usr/bin/env python3
"""
Create a minimal TensorFlow Lite model for testing
"""

try:
    import tensorflow as tf
    import numpy as np
    
    print("Creating minimal model...")
    
    # Create a very simple model
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    with open('app/src/main/assets/model_new.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model created successfully!")
    print(f"Size: {len(tflite_model)} bytes")
    print("Saved as: app/src/main/assets/model_new.tflite")
    
    # Test the model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test inference
    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print("Test inference successful!")
    print(f"Output: {output}")
    
except ImportError:
    print("TensorFlow not available. Creating a placeholder file...")
    # Create a minimal placeholder file
    with open('app/src/main/assets/model_placeholder.txt', 'w') as f:
        f.write("TensorFlow not available to create model. Please provide a valid model.tflite file.")
    print("Created placeholder file. You need to provide a valid TensorFlow Lite model.")

except Exception as e:
    print(f"Error creating model: {e}")
    print("Please provide a valid TensorFlow Lite model file.")