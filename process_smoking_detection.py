import os
import json
import torch
import argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm
import threading
import queue
import time
from functools import lru_cache


def load_model():
    """Load and initialize the model for smoking detection"""
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
        trust_remote_code=True,
    )
    return model


# Cache image encoding to avoid re-encoding the same image
@lru_cache(maxsize=32)
def encode_image_cached(image_path):
    """Cache image encoding to avoid redundant work"""
    # Convert path to string for caching
    if not isinstance(image_path, str):
        image_path = str(image_path)

    # Load and encode the image
    img = Image.open(image_path)
    return img


def process_image(model, image_path, frames_data, detect_only=False):
    """Process a single image and detect if it contains smoking"""
    start_time = time.time()

    # Load the image
    img = encode_image_cached(image_path)
    load_time = time.time() - start_time

    # Encode image
    encode_start = time.time()
    encoded = model.encode_image(img)
    encode_time = time.time() - encode_start

    # Run query for smoking detection
    query_start = time.time()
    result = model.query(
        encoded,
        "Does the image contain any form of smoking, including cigarettes, vapes, tobacco products, or visible smoke? Answer strictly 'yes' or 'no'.",
        max_tokens=10,
    )
    query_time = time.time() - query_start

    # Check if smoking was detected
    smoking_detected = result["answer"].strip().lower() == "yes"

    # Optional debug timing info
    if os.environ.get("DEBUG_TIMING"):
        print(f"Image: {os.path.basename(image_path)}")
        print(f"  Load time: {load_time:.3f}s")
        print(f"  Encode time: {encode_time:.3f}s")
        print(f"  Query time: {query_time:.3f}s")
        print(f"  Total time: {time.time() - start_time:.3f}s")
        print(f"  Result: {'Smoking' if smoking_detected else 'No smoking'}")

    # Get image dimensions
    img_width, img_height = img.size

    # Create a copy of the image for annotation if not in detect-only mode
    annotated_img = None
    if not detect_only and smoking_detected:
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)

        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            font = ImageFont.load_default()

        # Find the frame in frames_data
        for frame in frames_data:
            if os.path.basename(image_path) == frame["path"]:
                # Update smoking status
                frame["smoking"] = smoking_detected

                # Draw red rectangle around the entire image
                draw.rectangle(
                    [(0, 0), (img_width, img_height)], outline="red", width=3
                )

                # Add text label
                draw.text((10, 10), "SMOKING DETECTED", fill="red", font=font)

                # Add timestamp information if available
                if "from_sec" in frame and "to_sec" in frame:
                    timestamp_text = f"Time: {frame['from_sec']}s - {frame['to_sec']}s"
                    draw.text(
                        (10, img_height - 30), timestamp_text, fill="red", font=font
                    )
                break

    return frames_data, annotated_img, smoking_detected


def process_images_in_batches(model, image_paths, input_dir, frames_data, detect_only=False, batch_size=8):
    """Process images in batches for better GPU utilization"""
    results = []
    smoking_count = 0
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        
        # Process each image in the batch
        for image_name in batch:
            image_path = os.path.join(input_dir, image_name)
            if os.path.exists(image_path):
                frames_data, annotated_img, smoking_detected = process_image(
                    model, image_path, frames_data, detect_only=detect_only
                )
                
                if smoking_detected:
                    smoking_count += 1
                
                results.append((image_name, annotated_img, smoking_detected))
            else:
                print(f"Warning: Image file not found: {image_path}")
    
    return results, smoking_count


def worker_thread(model, task_queue, result_queue, input_dir, frames_data, detect_only):
    """Worker thread function to process images"""
    while True:
        try:
            # Get a batch of images from the queue
            batch = task_queue.get(block=False)
            if batch is None:  # Sentinel value to stop the thread
                break
                
            # Process the batch
            for image_name in batch:
                image_path = os.path.join(input_dir, image_name)
                if os.path.exists(image_path):
                    updated_frames, annotated_img, smoking_detected = process_image(
                        model, image_path, frames_data.copy(), detect_only=detect_only
                    )
                    
                    # Put the result in the result queue
                    result_queue.put((image_name, annotated_img, smoking_detected, updated_frames))
                else:
                    print(f"Warning: Image file not found: {image_path}")
                    
            # Mark the task as done
            task_queue.task_done()
        except queue.Empty:
            # No more tasks
            break


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process images for smoking detection")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect smoking, don't annotate images",
    )
    parser.add_argument(
        "--input-dir",
        default="output_collages",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir",
        default="annotated_images",
        help="Directory to save annotated images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images to process in each batch for better GPU utilization",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads (use 1 for best GPU efficiency)",
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        help="Print detailed timing information",
    )
    args = parser.parse_args()

    if args.debug_timing:
        os.environ["DEBUG_TIMING"] = "1"

    # Login to Hugging Face
    login("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")

    # Paths
    input_dir = args.input_dir
    json_path = os.path.join(input_dir, "frames_info.json")
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load frames info
    with open(json_path, "r") as f:
        frames_data = json.load(f)

    print(f"Loaded {len(frames_data)} frames from JSON file")

    # Load model
    start_time = time.time()
    model = load_model()
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Get unique image paths
    image_paths = list(set(frame["path"] for frame in frames_data))
    print(f"Found {len(image_paths)} unique images to process")

    # Process images
    total_start_time = time.time()
    smoking_count = 0

    if args.threads > 1:
        # Using threading instead of multiprocessing to avoid pickling issues
        print(f"Processing with {args.threads} threads")
        
        # Create task and result queues
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Split images into batches and add to task queue
        for i in range(0, len(image_paths), args.batch_size):
            batch = image_paths[i:i + args.batch_size]
            task_queue.put(batch)
            
        # Add sentinel values to stop threads
        for _ in range(args.threads):
            task_queue.put(None)
            
        # Create and start worker threads
        threads = []
        for _ in range(args.threads):
            thread = threading.Thread(
                target=worker_thread,
                args=(model, task_queue, result_queue, input_dir, frames_data, args.detect_only)
            )
            thread.start()
            threads.append(thread)
            
        # Process results as they come in
        processed_count = 0
        with tqdm(total=len(image_paths)) as pbar:
            while processed_count < len(image_paths):
                try:
                    # Get a result from the queue with a timeout
                    image_name, annotated_img, smoking_detected, updated_frames = result_queue.get(timeout=1)
                    
                    # Update frames data with results
                    for frame in updated_frames:
                        for orig_frame in frames_data:
                            if os.path.basename(frame["path"]) == os.path.basename(orig_frame["path"]):
                                orig_frame["smoking"] = frame["smoking"]
                    
                    # Save annotated image if available
                    if smoking_detected:
                        smoking_count += 1
                    if annotated_img:
                        output_path = os.path.join(output_dir, image_name)
                        annotated_img.save(output_path)
                        
                    processed_count += 1
                    pbar.update(1)
                except queue.Empty:
                    # Check if all threads are still alive
                    if all(not thread.is_alive() for thread in threads):
                        break
                        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
    else:
        # Process images in batches for better GPU utilization
        print(f"Processing sequentially with batch size {args.batch_size}")
        results, smoking_count = process_images_in_batches(
            model, 
            image_paths, 
            input_dir, 
            frames_data, 
            detect_only=args.detect_only,
            batch_size=args.batch_size
        )
        
        # Save annotated images
        for image_name, annotated_img, _ in tqdm(results, desc="Saving annotated images"):
            if annotated_img:
                output_path = os.path.join(output_dir, image_name)
                annotated_img.save(output_path)

    # Save updated frames info
    with open(json_path, "w") as f:
        json.dump(frames_data, f, indent=2)

    total_time = time.time() - total_start_time
    print(f"Processing complete in {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / len(image_paths):.2f} seconds")
    print(
        f"Detected smoking in {smoking_count} of {len(image_paths)} images ({smoking_count/len(image_paths)*100:.1f}%)"
    )
    print(f"Updated frames_info.json with smoking detection results")


if __name__ == "__main__":
    main()
