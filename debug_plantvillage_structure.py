#!/usr/bin/env python3
"""
Debug script to explore PlantVillage dataset structure
Run this in Colab to understand your dataset structure
"""

import os
from pathlib import Path

def explore_dataset_structure(dataset_path="plantVillage", max_depth=3):
    """
    Explore the structure of your PlantVillage dataset
    """
    print(f"🔍 Exploring dataset structure: {dataset_path}")
    print("=" * 50)
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path '{dataset_path}' does not exist!")
        print("Available directories:")
        for item in Path('.').iterdir():
            if item.is_dir():
                print(f"   📁 {item.name}")
        return
    
    def explore_directory(path, depth=0, max_depth=3):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        items = list(path.iterdir()) if path.is_dir() else []
        
        # Count files by extension
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in items if f.is_file() and f.suffix.lower() in image_extensions]
        directories = [d for d in items if d.is_dir()]
        
        if depth == 0:
            print(f"📁 {path.name}/")
        else:
            print(f"{indent}📁 {path.name}/ ({len(image_files)} images, {len(directories)} subdirs)")
        
        # Show some example files
        if image_files and len(image_files) <= 5:
            for img in image_files[:3]:
                print(f"{indent}  🖼️ {img.name}")
        elif image_files:
            print(f"{indent}  🖼️ {image_files[0].name}")
            print(f"{indent}  🖼️ ... and {len(image_files)-1} more images")
        
        # Recurse into subdirectories
        for subdir in sorted(directories)[:10]:  # Limit to first 10 dirs
            explore_directory(subdir, depth + 1, max_depth)
        
        if len(directories) > 10:
            print(f"{indent}  ... and {len(directories)-10} more directories")
    
    explore_directory(dataset_path)
    
    # Summary statistics
    print("\n📊 Dataset Summary:")
    total_images = 0
    class_counts = {}
    
    def count_images(path, class_name=""):
        nonlocal total_images
        if path.is_dir():
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            images = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
            
            if images:
                total_images += len(images)
                if class_name:
                    class_counts[class_name] = len(images)
                return len(images)
            
            # Recurse into subdirectories
            for subdir in path.iterdir():
                if subdir.is_dir():
                    count_images(subdir, subdir.name if not class_name else class_name)
    
    count_images(dataset_path)
    
    print(f"   Total images found: {total_images}")
    if class_counts:
        print("   Classes found:")
        healthy_total = 0
        diseased_total = 0
        
        for class_name, count in sorted(class_counts.items()):
            status = "🟢 healthy" if "healthy" in class_name.lower() else "🔴 diseased"
            print(f"     {class_name}: {count} images ({status})")
            
            if "healthy" in class_name.lower():
                healthy_total += count
            else:
                diseased_total += count
        
        print(f"\n   Binary classification summary:")
        print(f"     🟢 Healthy: {healthy_total} images")
        print(f"     🔴 Diseased: {diseased_total} images")
        print(f"     📊 Balance ratio: {healthy_total/(healthy_total+diseased_total)*100:.1f}% healthy")

def find_plantvillage_root():
    """
    Try to find the actual PlantVillage dataset root
    """
    print("🔍 Searching for PlantVillage dataset...")
    
    # Common names for PlantVillage dataset
    possible_names = [
        'plantVillage', 'PlantVillage', 'plant_village', 'Plant_Village',
        'plantvillage', 'PLANTVILLAGE', 'New Plant Diseases Dataset',
        'PlantVillage-Dataset', 'plant-village'
    ]
    
    current_dir = Path('.')
    found_datasets = []
    
    # Check current directory
    for item in current_dir.iterdir():
        if item.is_dir():
            # Check if it's a PlantVillage-like dataset
            subdirs = [d.name for d in item.iterdir() if d.is_dir()]
            
            # Look for characteristic PlantVillage class names
            plant_indicators = ['apple', 'corn', 'tomato', 'potato', 'grape']
            health_indicators = ['healthy', 'disease', 'scab', 'rust', 'blight']
            
            has_plants = any(any(plant in subdir.lower() for plant in plant_indicators) for subdir in subdirs)
            has_health = any(any(health in subdir.lower() for health in health_indicators) for subdir in subdirs)
            
            if has_plants and has_health and len(subdirs) > 5:
                found_datasets.append(item)
                print(f"✅ Found potential dataset: {item.name} ({len(subdirs)} classes)")
    
    if found_datasets:
        print(f"\n📁 Recommended dataset to use: {found_datasets[0].name}")
        return str(found_datasets[0])
    else:
        print("❌ No PlantVillage-like dataset found")
        print("Available directories:")
        for item in current_dir.iterdir():
            if item.is_dir():
                subdirs = len([d for d in item.iterdir() if d.is_dir()])
                print(f"   📁 {item.name} ({subdirs} subdirectories)")
        return None

# Usage example for Colab
if __name__ == "__main__":
    print("🌱 PlantVillage Dataset Structure Explorer")
    print("=" * 50)
    
    # First, try to find the dataset
    dataset_path = find_plantvillage_root()
    
    if dataset_path:
        # Explore the found dataset
        explore_dataset_structure(dataset_path)
    else:
        print("\n💡 Manual exploration:")
        print("Try running: explore_dataset_structure('your_folder_name')")