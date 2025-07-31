#!/usr/bin/env python3
"""
Create a working TensorFlow Lite model using an older approach
"""

import tensorflow as tf
import numpy as np

# Use TF 2.x compatible approach
def create_minimal_model():
    # Create a simple model using the functional API
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input_image')
    
    # Simple processing layers
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', name='predictions')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='crop_classifier')
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create dummy data and train for 1 epoch to initialize weights
    dummy_x = np.random.random((5, 224, 224, 3)).astype(np.float32)
    dummy_y = np.random.randint(0, 2, (5,))
    
    model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
    
    return model

def convert_to_tflite_v2(model):
    # Use concrete function approach for better compatibility
    @tf.function
    def representative_dataset():
        for _ in range(10):
            yield [tf.random.normal((1, 224, 224, 3))]
    
    # Convert using the newer approach
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for quantization
    converter.representative_dataset = representative_dataset
    
    try:
        tflite_model = converter.convert()
        return tflite_model
    except Exception as e:
        print(f"Quantized conversion failed: {e}")
        # Try without optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        return converter.convert()

if __name__ == "__main__":
    print("Creating working TensorFlow Lite model...")
    
    try:
        # Create the model
        model = create_minimal_model()
        print("✅ Keras model created successfully")
        
        # Convert to TensorFlow Lite
        tflite_model = convert_to_tflite_v2(model)
        print("✅ TensorFlow Lite conversion successful")
        
        # Save the model
        output_path = "app/src/main/assets/model.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Model saved to {output_path}")
        print(f"📊 Model size: {len(tflite_model)} bytes")
        
        # Test the model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n📋 Model Details:")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Input type: {input_details[0]['dtype']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        print(f"   Output type: {output_details[0]['dtype']}")
        
        # Test inference
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"\n🧪 Test Inference:")
        print(f"   Output: {output}")
        print(f"   Predicted class: {np.argmax(output[0])}")
        print(f"   Confidence: {np.max(output[0]):.3f}")
        
        print("\n✅ Model is ready to use!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()