import os
import json
import torch
import argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm

def load_model():
    """Load and initialize the model for point detection"""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream-next")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream-next",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
        attn_implementation="flash_attention_2"
    )
    return model, tokenizer

def process_collage(model, collage_path, frames_data, detect_only=False):
    """Process a collage image and update smoking status for frames in it"""
    print(f"Processing {collage_path}...")
    
    # Load the image
    img = Image.open(collage_path)
    
    # Encode image
    encoded = model.encode_image(img)
    
    # Run point detection for smoking-related content
    result = model.point(encoded, "smoke, cigarette, tobacco, vape")
    
    # If no points detected, return unchanged frames_data
    if not result or len(result) == 0:
        return frames_data, None
    
    # Get image dimensions
    img_width, img_height = img.size
    
    # Create a copy of the image for annotation if not in detect-only mode
    annotated_img = None
    if not detect_only:
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            font = ImageFont.load_default()
    
    # For each detected point
    for point_group in result:
        for point in point_group:
            # Convert normalized coordinates to pixel coordinates
            x_pixel = point['x'] * img_width
            y_pixel = point['y'] * img_height
            
            # Check each frame in this collage to see if the point falls within it
            for frame in frames_data:
                if os.path.basename(collage_path) != frame["path"]:
                    continue
                
                # Get frame position in collage
                row, col = frame["row"], frame["col"]
                frame_width = img_width / 3  # Assuming 3x3 grid as in split.py
                frame_height = img_height / 3
                
                # Calculate frame boundaries
                frame_left = col * frame_width
                frame_right = (col + 1) * frame_width
                frame_top = row * frame_height
                frame_bottom = (row + 1) * frame_height
                
                # Check if point is within this frame
                if (frame_left <= x_pixel <= frame_right and 
                    frame_top <= y_pixel <= frame_bottom):
                    # Update smoking status
                    frame["smoking"] = True
                    print(f"Smoking detected in frame {frame['frame']} at position ({point['x']}, {point['y']})")
                    
                    # Annotate the image if not in detect-only mode
                    if not detect_only and annotated_img:
                        # Draw red rectangle
                        draw.rectangle(
                            [(frame_left, frame_top), (frame_right, frame_bottom)],
                            outline="red",
                            width=3
                        )
                        
                        # Add text label
                        draw.text(
                            (frame_left + 10, frame_top + 10),
                            "SMOKING DETECTED",
                            fill="red",
                            font=font
                        )
                        
                        # Mark the detection point
                        point_size = 5
                        draw.ellipse(
                            [(x_pixel - point_size, y_pixel - point_size), 
                             (x_pixel + point_size, y_pixel + point_size)],
                            fill="yellow"
                        )
    
    return frames_data, annotated_img

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process collages for smoking detection and annotation")
    parser.add_argument("--detect-only", action="store_true", help="Only detect smoking, don't annotate images")
    parser.add_argument("--annotate-only", action="store_true", help="Only annotate images based on existing JSON data")
    args = parser.parse_args()
    
    # Paths
    collages_dir = "output_collages"
    json_path = os.path.join(collages_dir, "frames_info.json")
    output_dir = "annotated_collages"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load frames info
    with open(json_path, 'r') as f:
        frames_data = json.load(f)
    
    print(f"Loaded {len(frames_data)} frames from JSON file")
    
    if args.annotate_only:
        # Only annotate images based on existing JSON data
        annotate_from_json(frames_data, collages_dir, output_dir)
    else:
        # Login to Hugging Face
        login("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")
        
        # Load model
        model, _ = load_model()
        
        # Get unique collage paths
        collage_paths = set(frame["path"] for frame in frames_data)
        print(f"Found {len(collage_paths)} unique collages to process")
        
        # Process each collage
        for collage_name in tqdm(collage_paths):
            collage_path = os.path.join(collages_dir, collage_name)
            if os.path.exists(collage_path):
                frames_data, annotated_img = process_collage(
                    model, collage_path, frames_data, detect_only=args.detect_only
                )
                
                # Save annotated image if available
                if annotated_img:
                    output_path = os.path.join(output_dir, collage_name)
                    annotated_img.save(output_path)
                    print(f"Saved annotated image: {output_path}")
            else:
                print(f"Warning: Collage file not found: {collage_path}")
        
        # Save updated frames info
        with open(json_path, 'w') as f:
            json.dump(frames_data, f, indent=2)
        
        print("Processing complete. Updated frames_info.json with smoking detection results.")

def annotate_from_json(frames_data, collages_dir, output_dir):
    """Annotate images based on existing smoking detection data in JSON"""
    # Group frames by collage
    collages = {}
    for frame in frames_data:
        collage_path = frame["path"]
        if collage_path not in collages:
            collages[collage_path] = []
        collages[collage_path].append(frame)
    
    print(f"Found {len(collages)} unique collages to process")
    
    # Process each collage
    for collage_path, frames in tqdm(collages.items()):
        # Check if any frame in this collage has smoking=True
        has_smoking = any(frame["smoking"] for frame in frames)
        
        if has_smoking:
            # Load the image
            img_path = os.path.join(collages_dir, collage_path)
            try:
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                
                # Try to load a font, use default if not available
                try:
                    font = ImageFont.truetype("Arial", 20)
                except IOError:
                    font = ImageFont.load_default()
                
                # Get image dimensions
                img_width, img_height = img.size
                
                # Draw rectangles around frames with smoking
                for frame in frames:
                    if frame["smoking"]:
                        # Get frame position
                        row, col = frame["row"], frame["col"]
                        frame_width = img_width / 3  # Assuming 3x3 grid
                        frame_height = img_height / 3
                        
                        # Calculate frame boundaries
                        frame_left = col * frame_width
                        frame_right = (col + 1) * frame_width
                        frame_top = row * frame_height
                        frame_bottom = (row + 1) * frame_height
                        
                        # Draw red rectangle
                        draw.rectangle(
                            [(frame_left, frame_top), (frame_right, frame_bottom)],
                            outline="red",
                            width=3
                        )
                        
                        # Add text label
                        draw.text(
                            (frame_left + 10, frame_top + 10),
                            "SMOKING DETECTED",
                            fill="red",
                            font=font
                        )
                
                # Save the annotated image
                output_path = os.path.join(output_dir, collage_path)
                img.save(output_path)
                print(f"Saved annotated image: {output_path}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"Annotation complete. Annotated images saved to {output_dir}")

if __name__ == "__main__":
    main() 