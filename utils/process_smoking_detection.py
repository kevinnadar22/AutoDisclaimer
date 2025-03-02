import os
import json
import torch
import argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm
import time
from functools import lru_cache
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from typing import List, Tuple
import gc


def load_model():
    """Load and initialize the model for smoking detection"""
    print("Loading model...")
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
        trust_remote_code=True,
    )
    return model


# Cache image encoding to avoid re-encoding the same image
@lru_cache(maxsize=8)  # Reduced cache size
def encode_image_cached(image_path):
    """Cache image encoding to avoid redundant work"""
    # Convert path to string for caching
    if not isinstance(image_path, str):
        image_path = str(image_path)

    # Load and encode the image
    img = Image.open(image_path)
    return img


def custom_collate(batch: List[Tuple[Image.Image, str]]):
    """Custom collate function to handle PIL Images"""
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], input_dir: str):
        self.image_paths = image_paths
        self.input_dir = input_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        image_path = os.path.join(self.input_dir, self.image_paths[idx])
        return encode_image_cached(image_path), self.image_paths[idx]


def parallel_encode_images(model, image_paths, input_dir, batch_size=2):
    """Encode all images in parallel before processing"""
    print("Encoding images in parallel...")
    dataset = ImageDataset(image_paths, input_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=2,  # Reduced number of workers
        pin_memory=True,
        collate_fn=custom_collate,
        persistent_workers=True
    )
    
    encoded_images = {}
    
    with torch.cuda.device(0):
        for batch_images, batch_paths in tqdm(dataloader, desc="Encoding images"):
            try:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    # Process batch of images
                    encoded = [model.encode_image(img) for img in batch_images]
                    # Store encoded images
                    for enc, path in zip(encoded, batch_paths):
                        encoded_images[path] = enc.cpu()  # Move to CPU to save GPU memory
                    
                    # Clear GPU cache after each batch
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Clear cache and try again with smaller batch
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("OOM error, processing one by one...")
                    for img, path in zip(batch_images, batch_paths):
                        try:
                            with torch.no_grad(), torch.cuda.amp.autocast():
                                enc = model.encode_image(img)
                                encoded_images[path] = enc.cpu()
                                torch.cuda.empty_cache()
                        except:
                            print(f"Failed to process image: {path}")
                            continue
                else:
                    raise e
                
    return encoded_images


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
    # result = model.query(
    #     encoded,
    #     "Does the image contain any form of smoking, including cigarettes, vapes, tobacco products, or visible smoke? Answer strictly 'yes' or 'no'.",
    #     settings={"max_tokens": 10},
    # )
    result = model.point(
        img,
        "Does the image show a person actively smoking or holding a cigarette, vape, cigar?"
    )
    query_time = time.time() - query_start

    # Check if smoking was detected
    # smoking_detected = result["answer"].strip().lower() == "yes"
    smoking_detected = len(result["points"]) > 0
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

    # Update frames_data regardless of detect_only mode
    for frame in frames_data:
        if os.path.basename(image_path) == frame["path"]:
            frame["smoking"] = smoking_detected
            break

    return frames_data, annotated_img, smoking_detected


def process_images_in_batches(
    model, image_paths, input_dir, frames_data, detect_only=False, batch_size=2
):
    """Process images in batches for better GPU utilization"""
    results = []
    smoking_count = 0

    # First, encode all images in parallel
    encoded_images = parallel_encode_images(model, image_paths, input_dir, batch_size)

    # Process encoded images in smaller batches
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        torch.cuda.empty_cache()  # Clear GPU cache before each batch

        # Process each image in the batch
        for image_name in batch:
            image_path = os.path.join(input_dir, image_name)
            if os.path.exists(image_path):
                try:
                    with torch.cuda.amp.autocast():
                        # Use pre-encoded image
                        img = encode_image_cached(image_path)
                        
                        # Run point detection using pre-encoded image
                        result = model.point(
                            img,
                            "Does the image contain any form of smoking, including cigarettes, vapes, tobacco products, or visible smoke?"
                        )
                        
                        smoking_detected = len(result["points"]) > 0

                        # Create annotated image if needed
                        annotated_img = None
                        if not detect_only and smoking_detected:
                            annotated_img = create_annotated_image(img, image_path, frames_data)

                        if smoking_detected:
                            smoking_count += 1

                        # Update frames data
                        for frame in frames_data:
                            if frame["path"] == image_name:
                                frame["smoking"] = smoking_detected
                                break

                        results.append((image_name, annotated_img, smoking_detected))
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error processing {image_name}, skipping...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
            else:
                print(f"Warning: Image file not found: {image_path}")

        # Clear some memory after each batch
        gc.collect()
        torch.cuda.empty_cache()

    return results, smoking_count


def create_annotated_image(img, image_path, frames_data):
    """Create annotated image with smoking detection markers"""
    img_width, img_height = img.size
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

    return annotated_img


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
        default=2,
        help="Number of images to process in each batch for better GPU utilization",
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
    results, smoking_count = process_images_in_batches(
        model,
        image_paths,
        input_dir,
        frames_data,
        detect_only=args.detect_only,
        batch_size=args.batch_size,
    )

    # Save annotated images
    if not args.detect_only:
        for image_name, annotated_img, _ in tqdm(
            results, desc="Saving annotated images"
        ):
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
