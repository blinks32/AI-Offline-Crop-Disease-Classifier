# 🌱 Crop Disease Dataset Guide

## Best Datasets for Crop Disease Classification

### 1. **PlantVillage Dataset** (Recommended)
- **Kaggle**: https://www.kaggle.com/emmarex/plantdisease
- **Size**: 54,000+ images
- **Classes**: 38 classes (14 crop species, healthy + diseased)
- **Quality**: High-quality, labeled images
- **Best for**: General crop disease detection

### 2. **New Plant Diseases Dataset**
- **Kaggle**: https://www.kaggle.com/vipoooool/new-plant-diseases-dataset
- **Size**: 87,000+ images
- **Classes**: 38 classes
- **Quality**: Augmented version of PlantVillage
- **Best for**: Training robust models

### 3. **Plant Disease Recognition Dataset**
- **Kaggle**: https://www.kaggle.com/rashikrahmanpritom/plant-disease-recognition-dataset
- **Size**: 2,000+ images
- **Classes**: Multiple crop diseases
- **Quality**: Real-world images
- **Best for**: Realistic scenarios

### 4. **Crop Disease Dataset**
- **Kaggle**: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- **Size**: Various sizes available
- **Classes**: Focused on specific crops
- **Best for**: Specialized applications

## Quick Setup in Google Colab

### Option 1: Using Kaggle API
```python
# Install Kaggle
!pip install kaggle

# Upload your kaggle.json file (from Kaggle Account settings)
from google.colab import files
files.upload()  # Upload kaggle.json

# Move to correct location
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d emmarex/plantdisease
!unzip plantdisease.zip
```

### Option 2: Direct Upload
```python
from google.colab import files
uploaded = files.upload()  # Upload your zip file
!unzip your-dataset.zip
```

### Option 3: Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Access your dataset from Google Drive
dataset_path = '/content/drive/MyDrive/crop_disease_dataset'
```

## Dataset Structure

Organize your data like this:
```
dataset/
├── train/
│   ├── healthy/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── diseased/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── val/
    ├── healthy/
    └── diseased/
```

## Data Preprocessing Tips

### 1. **Image Size**
- Use 128x128 or 224x224 for mobile optimization
- Your current Android app expects 128x128

### 2. **Data Augmentation**
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### 3. **Class Balance**
- Ensure equal samples for healthy/diseased
- Use class weights if imbalanced:
```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(y_train), 
                                   y=y_train)
```

## Model Architecture Recommendations

### 1. **MobileNetV2** (Recommended for Android)
- Optimized for mobile devices
- Good accuracy vs size tradeoff
- Fast inference

### 2. **EfficientNet-B0**
- Better accuracy
- Slightly larger size
- Good for high-end devices

### 3. **Custom CNN**
- Smallest size
- Fastest inference
- May need more training data

## Training Tips

### 1. **Transfer Learning**
```python
base_model = MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False  # Freeze initially
```

### 2. **Fine-tuning**
```python
# After initial training
base_model.trainable = True
# Freeze early layers, fine-tune later ones
for layer in base_model.layers[:100]:
    layer.trainable = False
```

### 3. **Callbacks**
```python
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]
```

## TensorFlow Lite Conversion

### 1. **INT8 Quantization** (Recommended)
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

### 2. **Float16 Quantization**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

## Testing Your Model

### 1. **Accuracy Check**
```python
# Test on validation set
test_loss, test_acc = model.evaluate(val_generator)
print(f'Test accuracy: {test_acc:.3f}')
```

### 2. **TFLite Model Test**
```python
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
# Run test inference
```

### 3. **Android Integration Test**
- Replace model.tflite in your Android project
- Test with real camera input
- Check inference speed and accuracy

## Common Issues & Solutions

### 1. **Model Too Large**
- Use MobileNetV2 with alpha < 1.0
- Apply INT8 quantization
- Reduce input image size

### 2. **Low Accuracy**
- Use more training data
- Apply data augmentation
- Try transfer learning
- Increase model complexity

### 3. **Slow Inference**
- Use quantized models
- Optimize for mobile (MobileNet)
- Reduce input size

### 4. **Android Integration Issues**
- Check input/output data types
- Verify image preprocessing
- Match model input size

## Next Steps

1. **Choose a dataset** from the recommendations above
2. **Run the Colab notebook** with your chosen dataset
3. **Download the generated .tflite file**
4. **Replace the model** in your Android project
5. **Test and iterate** based on results

Good luck with your crop disease classifier! 🌱