"""
Quick fix cell to add to your Colab notebook
This will properly handle your plantVillage folder with 6 subdirectories
"""

# Add this cell to your Colab notebook to fix the detection issue

# Manual dataset exploration and setup
def explore_and_setup_dataset(folder_name="plantVillage"):
    from pathlib import Path
    import shutil
    import numpy as np
    
    print(f"🔍 Exploring {folder_name} folder...")
    
    folder_path = Path(folder_name)
    if not folder_path.exists():
        print(f"❌ Folder {folder_name} doesn't exist!")
        return None, None
    
    # List all subdirectories and their contents
    subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
    print(f"📁 Found {len(subdirs)} subdirectories:")
    
    total_images = 0
    class_info = []
    
    for subdir in subdirs:
        # Count images in each subdirectory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(subdir.glob(f"*{ext}")))
        
        if image_files:
            class_info.append((subdir, len(image_files)))
            total_images += len(image_files)
            
            # Determine if healthy or diseased
            class_type = "🟢 healthy" if "healthy" in subdir.name.lower() else "🔴 diseased"
            print(f"   📁 {subdir.name}: {len(image_files)} images ({class_type})")
        else:
            print(f"   📁 {subdir.name}: 0 images (empty)")
    
    print(f"\n📊 Total images found: {total_images}")
    
    if total_images == 0:
        print("❌ No images found in any subdirectory!")
        return None, None
    
    # Now reorganize for binary classification
    print(f"\n🔄 Reorganizing for binary classification...")
    
    # Create output directories
    output_dir = Path("crop_disease_dataset")
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # Remove existing output directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Create directory structure
    for split in ["train", "val"]:
        for class_name in ["healthy", "diseased"]:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    healthy_total = 0
    diseased_total = 0
    
    for class_dir, image_count in class_info:
        class_name = class_dir.name.lower()
        
        # Determine target class
        if "healthy" in class_name:
            target_class = "healthy"
        else:
            target_class = "diseased"
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(class_dir.glob(f"*{ext}")))
        
        if image_files:
            print(f"   Processing {class_dir.name}: {len(image_files)} images → {target_class}")
            
            # Shuffle for random split
            np.random.shuffle(image_files)
            
            # Split 80/20
            split_idx = int(len(image_files) * 0.8)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            # Copy files
            for i, img_file in enumerate(train_files):
                dest = train_dir / target_class / f"{class_dir.name}_{i:04d}.jpg"
                try:
                    shutil.copy2(img_file, dest)
                except Exception as e:
                    print(f"     ⚠️ Error copying {img_file.name}: {e}")
            
            for i, img_file in enumerate(val_files):
                dest = val_dir / target_class / f"{class_dir.name}_{i:04d}.jpg"
                try:
                    shutil.copy2(img_file, dest)
                except Exception as e:
                    print(f"     ⚠️ Error copying {img_file.name}: {e}")
            
            # Update counters
            if target_class == "healthy":
                healthy_total += len(image_files)
            else:
                diseased_total += len(image_files)
    
    # Verify results
    train_healthy = len(list((train_dir / "healthy").glob("*")))
    train_diseased = len(list((train_dir / "diseased").glob("*")))
    val_healthy = len(list((val_dir / "healthy").glob("*")))
    val_diseased = len(list((val_dir / "diseased").glob("*")))
    
    print(f"\n✅ Dataset reorganization completed!")
    print(f"📊 Training set: {train_healthy} healthy, {train_diseased} diseased")
    print(f"📊 Validation set: {val_healthy} healthy, {val_diseased} diseased")
    print(f"📊 Total processed: {train_healthy + train_diseased + val_healthy + val_diseased} images")
    
    if train_healthy + train_diseased + val_healthy + val_diseased > 0:
        return str(train_dir), str(val_dir)
    else:
        print("❌ No images were successfully processed!")
        return None, None

# Run the exploration and setup
TRAIN_DIR, VAL_DIR = explore_and_setup_dataset("plantVillage")

if TRAIN_DIR and VAL_DIR:
    print(f"\n🎉 Success! Dataset ready for training:")
    print(f"   📁 Training data: {TRAIN_DIR}")
    print(f"   📁 Validation data: {VAL_DIR}")
else:
    print("\n❌ Dataset setup failed. Please check your plantVillage folder structure.")