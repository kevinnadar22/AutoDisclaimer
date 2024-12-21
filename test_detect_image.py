import moondream as md
from PIL import Image
import time
import psutil
import json
import os
from tqdm import tqdm


def process_single_image(image_path, prompt):
    """Process a single image with the model"""
    model = md.vl(model="moondream-0_5b-int8.mf")
    image = Image.open(image_path)
    encoded = model.encode_image(image)
    result = model.detect(encoded, prompt)
    return {"path": image_path, "coordinates": result}


def test_multiple_images(prompt="cigarette or vape"):
    """Test processing multiple images in parallel"""
    print("\n=== Starting Sequential Image Processing Test ===")
    print(f"Using prompt: '{prompt}'\n")

    # Get all images from Images folder
    image_folder = "images/20241221_135955"
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Performance tracking
    start_time = time.time()
    initial_cpu = psutil.cpu_percent()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024

    # Process images sequentially
    results = []
    for img_path in tqdm(image_files, desc="Processing images", unit="image"):
        result = process_single_image(img_path, prompt)
        results.append(result)

    # Performance measurements
    end_time = time.time()
    final_cpu = psutil.cpu_percent()
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024

    # Print results
    print("\n=== Results ===")
    for result in results:
        print(f"\nImage: {result['path']}")
        print(f"Found {len(result['coordinates'])} points")
        print(f"Points: {json.dumps(result['coordinates'], indent=2)}")

    print("\n=== Performance Metrics ===")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"CPU Usage: {final_cpu:.1f}%")
    print(f"Memory Usage: {mem_after - mem_before:.1f} MB")
    print(f"Total Memory: {mem_after:.1f} MB")


if __name__ == "__main__":
    test_multiple_images()
