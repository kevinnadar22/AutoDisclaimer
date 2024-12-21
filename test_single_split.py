import cv2
from PIL import Image
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


def extract_frames(video_path):
    run_folder = create_folder_structure()
    coordinates_data = {
        "video_path": video_path,
        "frame_size": {"width": 320, "height": 180},
        "frames": [],
    }

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Extract 10 frames per second
    frames_per_second = 3
    frame_interval = fps // frames_per_second
    total_frames_to_extract = int(duration * frames_per_second)

    target_size = (320, 180)  # Frame size

    for frame_num in range(total_frames_to_extract):
        frame_pos = frame_num * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()

        if ret:
            # Convert and resize frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, target_size)
            frame_img = Image.fromarray(resized_frame)

            # Save the frame
            frame_path = os.path.join(run_folder, f"frame_{frame_num + 1}.jpg")
            frame_img.save(frame_path)

            # Store frame information
            frame_info = {
                "frame_number": frame_num + 1,
                "filename": f"frame_{frame_num + 1}.jpg",
                "video_frame_position": int(frame_pos),
                "timestamp": frame_pos / fps,
                "smoking": False,
                "drinking": False,
            }
            coordinates_data["frames"].append(frame_info)

    cap.release()

    # Save coordinates data to JSON
    json_path = os.path.join(run_folder, "coordinates.json")
    with open(json_path, "w") as f:
        json.dump(coordinates_data, f, indent=4)

    return run_folder


# Usage
if __name__ == "__main__":
    video_path = "video.mp4"
    output_folder = extract_frames(video_path)
    print(f"Processing complete. Output saved in: {output_folder}")
