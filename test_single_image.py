import moondream as md
from PIL import Image
import time
import psutil
import json

def test_single_image(image_path, prompt="cigarette or vape"):
    """Test processing a single image and measure performance"""
    print("\n=== Starting Single Image Test ===")
    print(f"Testing image: {image_path}")
    print(f"Using prompt: '{prompt}'\n")

    # Initialize model and measure loading time
    load_start = time.time()
    model = md.vl(model="moondream-2b-int8.mf")
    load_time = time.time() - load_start
    print(f"Model load time: {load_time:.2f} seconds")

    # Process image and measure performance
    start_time = time.time()
    initial_cpu = psutil.cpu_percent()
    
    # Memory usage before
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # Process image
    image = Image.open(image_path)
    encoded = model.encode_image(image)
    result = model.point(encoded, prompt)
    coordinates = result["points"]

    # Performance measurements
    end_time = time.time()
    final_cpu = psutil.cpu_percent()
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # Print results
    print("\n=== Results ===")
    print(f"Found {len(coordinates)} points")
    print(f"Points: {json.dumps(coordinates, indent=2)}")

    print("\n=== Performance Metrics ===")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"CPU Usage: {final_cpu:.1f}%")
    print(f"Memory Usage: {mem_after - mem_before:.1f} MB")
    print(f"Total Memory: {mem_after:.1f} MB")

if __name__ == "__main__":
    # Replace with your image path
    IMAGE_PATH = "/workspaces/Auto-Disclaimer-Adder/images/20241220_233946/collage_1.jpg"
    test_single_image(IMAGE_PATH) 