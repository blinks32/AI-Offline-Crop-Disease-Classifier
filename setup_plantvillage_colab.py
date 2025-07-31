#!/usr/bin/env python3
"""
Setup script for PlantVillage dataset in Google Colab
Run this in a Colab cell after uploading your dataset
"""

import os
import shutil
from pathlib import Path

def setup_plantvillage_dataset(dataset_path="PlantVillage"):
    """
    Reorganize PlantVillage dataset for binary classification (healthy vs diseased)
    """
    
    print("🌱 Setting up PlantVillage dataset for crop disease classification...")
    
    # Create output directories
    output_dir = Path("crop_disease_dataset")
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # Create subdirectories
    for split in ["train", "val"]:
        for class_name in ["healthy", "diseased"]:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Find all class directories
    dataset_path = Path(dataset_path)
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    print(f"📁 Found {len(class_dirs)} classes in dataset")
    
    healthy_count = 0
    diseased_count = 0
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name.lower()
        
        # Determine if healthy or diseased
        if "healthy" in class_name:
            target_class = "healthy"
        else:
            target_class = "diseased"
        
        # Get all images in this class
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        
        print(f"   {class_dir.name}: {len(image_files)} images → {target_class}")
        
        # Split into train/val (80/20)
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy files
        for img_file in train_files:
            dest = train_dir / target_class / f"{class_dir.name}_{img_file.name}"
            shutil.copy2(img_file, dest)
        
        for img_file in val_files:
            dest = val_dir / target_class / f"{class_dir.name}_{img_file.name}"
            shutil.copy2(img_file, dest)
        
        if target_class == "healthy":
            healthy_count += len(image_files)
        else:
            diseased_count += len(image_files)
    
    # Print summary
    train_healthy = len(list((train_dir / "healthy").glob("*")))
    train_diseased = len(list((train_dir / "diseased").glob("*")))
    val_healthy = len(list((val_dir / "healthy").glob("*")))
    val_diseased = len(list((val_dir / "diseased").glob("*")))
    
    print(f"\n✅ Dataset reorganized successfully!")
    print(f"📊 Summary:")
    print(f"   Training set:")
    print(f"     - Healthy: {train_healthy} images")
    print(f"     - Diseased: {train_diseased} images")
    print(f"   Validation set:")
    print(f"     - Healthy: {val_healthy} images")
    print(f"     - Diseased: {val_diseased} images")
    print(f"   Total: {healthy_count + diseased_count} images")
    
    return str(train_dir), str(val_dir)

# Example usage for Colab
def colab_setup_example():
    """
    Complete setup example for Google Colab
    """
    print("🚀 PlantVillage Dataset Setup for Google Colab")
    print("=" * 50)
    
    # Step 1: Upload and extract dataset
    print("\n1️⃣ Upload your PlantVillage dataset zip file:")
    print("   Run this in a Colab cell:")
    print("""
    from google.colab import files
    uploaded = files.upload()  # Upload plantvillage.zip
    
    # Extract the dataset
    import zipfile
    with zipfile.ZipFile('plantvillage.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    """)
    
    # Step 2: Reorganize dataset
    print("\n2️⃣ Reorganize dataset:")
    print("   After extraction, run:")
    print("""
    # Find the extracted folder name (might be 'PlantVillage' or similar)
    import os
    folders = [f for f in os.listdir('.') if os.path.isdir(f) and 'plant' in f.lower()]
    print("Available folders:", folders)
    
    # Use the setup function
    train_dir, val_dir = setup_plantvillage_dataset(folders[0])  # Use first matching folder
    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")
    """)
    
    # Step 3: Use in training
    print("\n3️⃣ Use in your training script:")
    print("""
    # In your training notebook, use these paths:
    TRAIN_DIR = 'crop_disease_dataset/train'
    VAL_DIR = 'crop_disease_dataset/val'
    
    # Create data generators
    train_gen, val_gen = create_data_generators(TRAIN_DIR, VAL_DIR)
    """)

if __name__ == "__main__":
    colab_setup_example()