"""
Debug cell to understand your plantVillage folder structure
Add this to your Colab notebook first
"""

# Debug: Explore plantVillage folder structure
from pathlib import Path

def debug_plantvillage_structure():
    folder_path = Path("plantVillage")
    
    print("🔍 Detailed exploration of plantVillage folder:")
    print("=" * 50)
    
    if not folder_path.exists():
        print("❌ plantVillage folder doesn't exist!")
        return
    
    # List all items in plantVillage
    items = list(folder_path.iterdir())
    print(f"📁 plantVillage contains {len(items)} items:")
    
    for item in items:
        if item.is_dir():
            # Count files in subdirectory
            files = list(item.iterdir())
            image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
            
            print(f"\n   📁 {item.name}/")
            print(f"      Total files: {len(files)}")
            print(f"      Image files: {len(image_files)}")
            
            # Show first few files as examples
            if image_files:
                print(f"      Sample images:")
                for img in image_files[:3]:
                    print(f"        🖼️ {img.name}")
                if len(image_files) > 3:
                    print(f"        ... and {len(image_files) - 3} more")
            
            # Check if it has subdirectories (nested structure)
            subdirs = [d for d in files if d.is_dir()]
            if subdirs:
                print(f"      Subdirectories: {len(subdirs)}")
                for subdir in subdirs[:3]:
                    sub_images = len([f for f in subdir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
                    print(f"        📁 {subdir.name}/ ({sub_images} images)")
                if len(subdirs) > 3:
                    print(f"        ... and {len(subdirs) - 3} more subdirectories")
        else:
            print(f"   📄 {item.name}")
    
    print("\n" + "=" * 50)
    
    # Summary
    total_direct_images = 0
    total_nested_images = 0
    
    for item in items:
        if item.is_dir():
            # Direct images in this directory
            direct_images = [f for f in item.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
            total_direct_images += len(direct_images)
            
            # Images in subdirectories
            for subitem in item.iterdir():
                if subitem.is_dir():
                    nested_images = [f for f in subitem.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
                    total_nested_images += len(nested_images)
    
    print(f"📊 Summary:")
    print(f"   Direct images (in main subdirectories): {total_direct_images}")
    print(f"   Nested images (in sub-subdirectories): {total_nested_images}")
    print(f"   Total images found: {total_direct_images + total_nested_images}")
    
    if total_direct_images > 0:
        print(f"\n✅ Found images directly in subdirectories - this is the standard PlantVillage format")
    elif total_nested_images > 0:
        print(f"\n⚠️ Images are nested deeper - need to adjust processing")
    else:
        print(f"\n❌ No images found - check if files were extracted properly")

# Run the debug
debug_plantvillage_structure()