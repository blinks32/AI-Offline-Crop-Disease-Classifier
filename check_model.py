#!/usr/bin/env python3
"""
Check the current model.tflite file to see what's wrong
"""

import os

def check_model_file():
    model_path = "app/src/main/assets/model.tflite"
    
    if not os.path.exists(model_path):
        print("❌ Model file does not exist!")
        return
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"📊 Model file size: {file_size} bytes ({file_size/1024:.1f} KB)")
    
    if file_size < 1000:
        print("⚠️ Model file is very small - likely corrupted or placeholder")
        return
    
    # Try to read the first few bytes to check if it's a valid TFLite file
    try:
        with open(model_path, 'rb') as f:
            header = f.read(16)
            
        print(f"📋 File header (first 16 bytes): {header.hex()}")
        
        # TFLite files should start with specific magic bytes
        # The first 4 bytes should be the TFLite identifier
        magic_bytes = header[:4]
        print(f"🔍 Magic bytes: {magic_bytes.hex()}")
        
        # Check if it looks like a TFLite file
        if len(header) >= 8:
            # TFLite files have a specific structure
            print("✅ File appears to be a binary file (not empty)")
            
            # Try to load with TensorFlow if available
            try:
                import tensorflow as tf
                
                print("🧪 Testing with TensorFlow...")
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                print("✅ Model loaded successfully with TensorFlow!")
                print(f"📥 Input shape: {input_details[0]['shape']}")
                print(f"📤 Output shape: {output_details[0]['shape']}")
                print(f"🔢 Input type: {input_details[0]['dtype']}")
                print(f"🔢 Output type: {output_details[0]['dtype']}")
                
                # Test inference
                import numpy as np
                test_input = np.random.random(input_details[0]['shape']).astype(input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                
                print(f"🧪 Test inference successful!")
                print(f"📊 Output: {output}")
                
            except ImportError:
                print("⚠️ TensorFlow not available for testing")
            except Exception as e:
                print(f"❌ TensorFlow test failed: {e}")
                print("🔧 This might be why the Android app is crashing")
        else:
            print("❌ File header is too short - file might be corrupted")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    print("🔍 Checking model.tflite file...")
    check_model_file()