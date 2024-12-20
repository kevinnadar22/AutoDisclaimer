import moondream as md
from PIL import Image
import concurrent.futures
import time
from functools import partial
import psutil
import threading

# Global variable for CPU measurements
cpu_percentages = []

def monitor_cpu():
    """Function to monitor CPU usage"""
    while monitor_cpu.running:
        cpu_percentages.append(psutil.cpu_percent(interval=0.1))
        time.sleep(0.1)

def process_single_query(model, encoded, query):
    return model.point(encoded, query)

def main():
    # Initialize model
    model = md.vl(model="moondream-2b-int8.mf")

    # Process the image (done once)
    image = Image.open("images/collage_1.jpg")
    encoded = model.encode_image(image)

    # Create 10 identical queries
    queries = ["lighter or cigarette or smoke"] * 10

    # Start CPU monitoring
    monitor_cpu.running = True
    cpu_thread = threading.Thread(target=monitor_cpu)
    cpu_thread.start()

    # Start timing
    start_time = time.time()

    # Run queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        process_query = partial(process_single_query, model, encoded)
        results = list(executor.map(process_query, queries))

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Stop CPU monitoring
    monitor_cpu.running = False
    cpu_thread.join()

    # Calculate CPU statistics
    avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
    max_cpu = max(cpu_percentages) if cpu_percentages else 0

    # Print results
    print("\n=== Performance Results ===")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Average time per instance: {execution_time/10:.2f} seconds")
    print(f"Average CPU usage: {avg_cpu:.1f}%")
    print(f"Peak CPU usage: {max_cpu:.1f}%")
    print("\n=== Query Results ===")
    for i, result in enumerate(results, 1):
        print(f"Instance {i} Answer: {result}")

if __name__ == "__main__":
    main()