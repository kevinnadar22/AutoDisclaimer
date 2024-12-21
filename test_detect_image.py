import moondream as md
from PIL import Image
import time
import psutil
import json
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm


def process_single_image(args):
    """Process a single image with the model"""
    image_path, prompt = args
    model = md.vl(model="moondream-0_5b-int8.mf")
    image = Image.open(image_path)
    encoded = model.encode_image(image)
    result = model.point(encoded, prompt)
    return {"path": image_path, "coordinates": result["answer"]}


def test_multiple_images(prompt="cigarette or vape"):
    """Test processing multiple images in parallel"""
    print("\n=== Starting Parallel Image Processing Test ===")
    print(f"Using prompt: '{prompt}'\n")

    # Get all images from Images folder
    image_folder = "images"
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Performance tracking
    start_time = time.time()
    initial_cpu = psutil.cpu_percent()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024

    # Process images in parallel with 10 workers
    args_list = [(img_path, prompt) for img_path in image_files]
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(process_single_image, args_list),
            total=len(args_list),
            desc="Processing images",
            unit="image"
        ))

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
