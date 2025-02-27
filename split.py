import cv2
import numpy as np
import os
import json
from typing import Tuple, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import time


def split_video_and_create_collages(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    collage_grid: Tuple[int, int] = (3, 3),
    max_frames: Optional[int] = None,
    resize_frames: Optional[Tuple[int, int]] = None,
    num_workers: int = 4,
) -> List[str]:
    """
    Split a video into frames and create collages of those frames.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the collages
        frame_interval: Extract every nth frame
        collage_grid: Tuple of (rows, cols) for the collage layout
        max_frames: Maximum number of frames to process (None for all)
        resize_frames: Optional tuple (width, height) to resize frames before collage
        num_workers: Number of worker threads for parallel processing

    Returns:
        List of paths to the generated collage images
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {total_frames} frames, {fps} FPS, Resolution: {width}x{height}")

    # Calculate how many frames to process
    if max_frames is not None:
        frames_to_process = min(total_frames, max_frames)
    else:
        frames_to_process = total_frames

    # Extract all frames first to avoid parallel access to the video capture object
    frames = []
    frame_count = 0
    frame_info = []  # List to store information about each frame

    while frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Only resize if explicitly requested and preserve original dimensions for collage
            if resize_frames:
                frame = cv2.resize(frame, resize_frames)
            
            frames.append(frame)

        frame_count += 1

        # Print progress periodically
        if frame_count % 100 == 0:
            print(f"Extracted {frame_count}/{frames_to_process} frames...")

    # Release the video capture object
    cap.release()

    print(f"Total frames extracted: {len(frames)}")

    if not frames:
        print("No frames were extracted from the video!")
        return []

    # Calculate how many collages we can create with the frames we have
    frames_per_collage = collage_grid[0] * collage_grid[1]
    num_collages = (len(frames) + frames_per_collage - 1) // frames_per_collage

    print(
        f"Creating {num_collages} collages with {frames_per_collage} frames per collage"
    )

    # Create a black frame for padding if needed
    black_frame = np.zeros_like(frames[0])
    
    # Get frame dimensions for calculating positions
    frame_h, frame_w = frames[0].shape[:2]
    rows, cols = collage_grid

    # Function to process a batch of frames into a collage
    def process_collage(collage_idx):
        start_idx = collage_idx * frames_per_collage
        end_idx = min(start_idx + frames_per_collage, len(frames))
        
        # Get frames for this collage
        collage_frames = frames[start_idx:end_idx]
        
        # If we didn't get enough frames, pad with black frames
        if len(collage_frames) < frames_per_collage:
            collage_frames.extend(
                [black_frame] * (frames_per_collage - len(collage_frames))
            )

        # Create the collage
        collage = np.zeros((frame_h * rows, frame_w * cols, 3), dtype=np.uint8)

        for i, frame in enumerate(collage_frames):
            r, c = i // cols, i % cols
            # Ensure the frame has the correct dimensions
            if frame.shape[:2] != (frame_h, frame_w):
                frame = cv2.resize(frame, (frame_w, frame_h))
            collage[
                r * frame_h : (r + 1) * frame_h, c * frame_w : (c + 1) * frame_w
            ] = frame
            
            # Only store info for real frames (not padding)
            if start_idx + i < len(frames):
                # Calculate the center position of this frame in the collage
                center_x = c * frame_w + frame_w // 2
                center_y = r * frame_h + frame_h // 2
                
                # Store frame information
                frame_data = {
                    "frame": start_idx + i,
                    "path": f"collage_{collage_idx:04d}.jpg",
                    "x": center_x,
                    "y": center_y,
                    "collage_idx": collage_idx,
                    "position_in_collage": i,
                    "row": r,
                    "col": c,
                    "smoking": False
                }
                frame_info.append(frame_data)

        # Save the collage
        output_path = os.path.join(output_dir, f"collage_{collage_idx:04d}.jpg")
        cv2.imwrite(output_path, collage)
        return output_path

    # Process collages in parallel
    start_time = time.time()
    collage_paths = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_collage, i) for i in range(num_collages)]
        for i, future in enumerate(futures):
            try:
                collage_paths.append(future.result())
                if (i + 1) % 10 == 0 or i == len(futures) - 1:
                    print(f"Created {i + 1}/{num_collages} collages...")
            except Exception as e:
                print(f"Error creating collage {i}: {str(e)}")

    elapsed = time.time() - start_time
    print(
        f"Processed {len(frames)} frames into {len(collage_paths)} collages in {elapsed:.2f} seconds"
    )
    print(f"Processing speed: {len(frames) / elapsed:.2f} frames per second")

    # Clean up any existing blank collages beyond what we need
    for i in range(num_collages, 999):
        file_path = os.path.join(output_dir, f"collage_{i:04d}.jpg")
        if os.path.exists(file_path):
            print(f"Removing unnecessary collage: {file_path}")
            os.remove(file_path)
    
    # Sort frame information by frame number
    frame_info.sort(key=lambda x: x["frame"])
    print(f"Sorting {len(frame_info)} frame entries by frame number...")
    
    # Save frame information to JSON file
    json_path = os.path.join(output_dir, "frames_info.json")
    with open(json_path, 'w') as f:
        json.dump(frame_info, f, indent=2)
    
    print(f"Frame information saved to {json_path}")

    return collage_paths


if __name__ == "__main__":
    # Example usage
    video_path = "video.mp4"  # Replace with your video path
    output_dir = "output_collages"

    # Create 3x3 collages, taking every 2nd frame
    collage_paths = split_video_and_create_collages(
        video_path=video_path,
        output_dir=output_dir,
        frame_interval=2,
        collage_grid=(3, 3),
        resize_frames=None,  # Don't resize frames to preserve original dimensions
    )

    print(f"Created {len(collage_paths)} collages")
