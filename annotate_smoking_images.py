import json
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def annotate_smoking_images():
    """
    Annotate collage images where smoking is detected and save them in a new directory.
    """
    # Paths
    collages_dir = "output_collages"
    json_path = os.path.join(collages_dir, "frames_info.json")
    output_dir = "annotated_collages"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load frames info
    with open(json_path, "r") as f:
        frames_data = json.load(f)

    print(f"Loaded {len(frames_data)} frames from JSON file")

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
                            width=3,
                        )

                        # Add text label
                        draw.text(
                            (frame_left + 10, frame_top + 10),
                            "SMOKING DETECTED",
                            fill="red",
                            font=font,
                        )

                # Save the annotated image
                output_path = os.path.join(output_dir, collage_path)
                img.save(output_path)
                print(f"Saved annotated image: {output_path}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"Annotation complete. Annotated images saved to {output_dir}")


if __name__ == "__main__":
    annotate_smoking_images()
