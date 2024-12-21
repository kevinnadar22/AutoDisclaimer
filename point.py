import moondream as md
from PIL import Image
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial


PROMPT = "cigarette or vape"


def is_point_in_frame(point, frame_coords):
    """Check if a point falls within a frame's boundaries"""
    x, y = point["x"], point["y"]
    top_left = frame_coords["top_left"]
    bottom_right = frame_coords["bottom_right"]

    return (
        top_left["x"] <= x <= bottom_right["x"]
        and top_left["y"] <= y <= bottom_right["y"]
    )


def process_single_image(model, image_path):
    """Process a single image and return coordinates where smoking is detected"""
    image = Image.open(image_path)
    encoded = model.encode_image(image)
    result = model.point(encoded, PROMPT)
    coordinates = result["points"]
    return image_path, coordinates



def update_json_with_smoking_frames(json_path, smoking_points):
    """Update JSON file marking frames that contain smoking points"""
    with open(json_path, "r") as f:
        data = json.load(f)

    # For each collage and frame, check if any smoking points fall within it
    for collage in data["collages"]:
        for frame in collage["frames"]:
            frame["smoking"] = any(
                is_point_in_frame(point, frame["coordinates"])
                for point in smoking_points
            )
    new_json_path = json_path.replace(".json", "_new.json")
    # Write back the updated data
    with open(new_json_path, "w") as f:
        json.dump(data, f, indent=4)

    return True



def process_folder(base_folder):
    """Process the latest folder and return list of smoking coordinates"""
    # Initialize model
    model = md.vl(model="moondream-2b-int8.mf")

    json_path = os.path.join(base_folder, "coordinates.json")
    if not os.path.exists(json_path):
        print(f"coordinates.json not found in {base_folder}")
        return []

    all_smoking_points = []
    collage_files = [
        f
        for f in os.listdir(base_folder)
        if f.startswith("collage_") and f.endswith(".jpg")
    ]

    # Process images in parallel
    print("Processing images for smoking detection...")
    process_func = partial(process_single_image, model)

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_path = {
            executor.submit(
                process_func, os.path.join(base_folder, collage_file)
            ): collage_file
            for collage_file in collage_files
        }

        for future in as_completed(future_to_path):
            collage_file = future_to_path[future]
            try:
                image_path, smoking_points = future.result()
                if smoking_points:
                    all_smoking_points.extend(smoking_points)
                    print(
                        f"Detected smoking in {collage_file} at coordinates: {smoking_points}"
                    )
            except Exception as e:
                print(f"Error processing {collage_file}: {e}")

    # Update JSON with smoking information
    if all_smoking_points:
        print("\nUpdating JSON file with smoking frame information...")
        update_json_with_smoking_frames(json_path, all_smoking_points)

    return all_smoking_points


def main():
    # Get the most recent folder
    images_dir = Path("images")
    if not images_dir.exists():
        print("Images directory not found")
        return

    folders = [f for f in images_dir.iterdir() if f.is_dir()]
    if not folders:
        print("No folders found in images directory")
        return

    latest_folder = max(folders, key=lambda x: x.stat().st_mtime)
    print(f"Processing folder: {latest_folder}")

    # Process the folder and get smoking coordinates
    smoking_points = process_folder(latest_folder)

    # Output results
    if smoking_points:
        print("\nSmoking detected at the following coordinates:")
        # save to file
        with open("smoking_points.json", "w") as f:
            json.dump(smoking_points, f, indent=2)
    else:
        print("\nNo smoking detected in any frames")


if __name__ == "__main__":
    main()
