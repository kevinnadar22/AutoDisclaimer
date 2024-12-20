import cv2
import numpy as np
from PIL import Image, ImageDraw
import json
import os
from datetime import datetime

# remove all the images in the current directory
os.system("rm -rf images")


def create_folder_structure():
    # Create main images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")

    # Create a timestamped folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join("images", timestamp)
    os.makedirs(run_folder)

    return run_folder


def extract_and_create_collage(video_path):
    run_folder = create_folder_structure()
    coordinates_data = {
        "video_path": video_path,
        "frame_size": {"width": 320, "height": 180},
        "grid_size": {"rows": 3, "cols": 3},
        "collages": [],
    }

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Extract 6 frames per second
    frames_per_second = 6
    frame_interval = fps // frames_per_second
    total_frames_to_extract = int(duration * frames_per_second)

    # Calculate number of frames per collage (9 frames per collage for 3x3)
    frames_per_collage = 9
    num_collages = total_frames_to_extract // frames_per_collage

    target_size = (320, 180)  # Frame size
    rows, cols = 3, 3  # Grid size

    for collage_num in range(num_collages):
        frames = []
        frame_positions = []  # Store original frame positions
        start_frame = collage_num * frames_per_collage * frame_interval

        # Extract 9 frames for this collage
        for i in range(frames_per_collage):
            frame_pos = start_frame + (i * frame_interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_positions.append(frame_pos)

        if len(frames) == frames_per_collage:
            # Create collage
            resized_frames = [cv2.resize(f, target_size) for f in frames]
            collage = Image.new("RGB", (target_size[0] * cols, target_size[1] * rows))

            # Store frame information for this collage
            collage_info = {
                "collage_number": collage_num + 1,
                "filename": f"collage_{collage_num + 1}.jpg",
                "frames": [],
            }

            # Place frames and store coordinates
            for idx, frame in enumerate(resized_frames):
                frame_img = Image.fromarray(frame)
                x = (idx % cols) * target_size[0]
                y = (idx // cols) * target_size[1]
                collage.paste(frame_img, (x, y))

                # Store frame information
                frame_info = {
                    "frame_number": idx + 1,
                    "video_frame_position": int(frame_positions[idx]),
                    "timestamp": frame_positions[idx] / fps,
                    "coordinates": {
                        "top_left": {"x": x, "y": y},
                        "bottom_right": {
                            "x": x + target_size[0],
                            "y": y + target_size[1],
                        },
                    },
                    "smoking": False,
                    "drinking": False,
                }
                collage_info["frames"].append(frame_info)

            # Save the collage
            collage_path = os.path.join(run_folder, f"collage_{collage_num + 1}.jpg")
            collage.save(collage_path)
            coordinates_data["collages"].append(collage_info)

    cap.release()

    # Save coordinates data to JSON
    json_path = os.path.join(run_folder, "coordinates.json")
    with open(json_path, "w") as f:
        json.dump(coordinates_data, f, indent=4)

    return run_folder


# Usage
if __name__ == "__main__":
    video_path = "video.mp4"
    output_folder = extract_and_create_collage(video_path)
    print(f"Processing complete. Output saved in: {output_folder}")
