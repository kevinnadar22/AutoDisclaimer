import cv2
import numpy as np
import os
import json
from typing import Tuple, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import time
import gc


def split_video_and_create_collages(
    video_path: str,
    output_dir: str,
    num_frames: int = 1,  # Number of frames per second to extract
    collage_grid: Optional[Tuple[int, int]] = (3, 3),  # None for single images
    max_frames: Optional[int] = None,
    resize_frames: Optional[Tuple[int, int]] = None,
    num_workers: int = 4,
) -> List[str]:
    """
    Split a video into frames and optionally create collages of those frames.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the collages/frames
        num_frames: Number of frames to extract per second
        collage_grid: Tuple of (rows, cols) for collage layout, or None for single frames
        max_frames: Maximum number of frames to process (None for all)
        resize_frames: Optional tuple (width, height) to resize frames
        num_workers: Number of worker threads for parallel processing

    Returns:
        List of paths to the generated images
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

    # Calculate frame interval based on fps and desired frames per second
    frame_interval = int(fps / num_frames)
    if frame_interval < 1:
        frame_interval = 1

    # Calculate how many frames to process
    if max_frames is not None:
        frames_to_process = min(total_frames, max_frames)
    else:
        frames_to_process = total_frames

    # Create JSON file for frame info and open it in append mode
    json_path = os.path.join(output_dir, "frames_info.json")
    
    # Initialize frame info as an empty list
    frame_info = []
    
    # For very large videos, we'll process frames in chunks to avoid memory issues
    chunk_size = 1000  # Process 1000 frames at a time
    
    # Function to extract frames in chunks
    def extract_frames_chunk(start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        timestamps = []
        extracted_indices = []
        
        for frame_idx in range(start_frame, min(end_frame, frames_to_process)):
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # Only resize if explicitly requested
                if resize_frames:
                    frame = cv2.resize(frame, resize_frames)
                    
                frames.append(frame)
                
                # Calculate timestamp
                timestamp_sec = frame_idx / fps
                timestamps.append(timestamp_sec)
                extracted_indices.append(frame_idx)
                
        cap.release()
        return frames, timestamps, extracted_indices
    
    # If we're creating individual frames (no collage)
    if collage_grid is None:
        output_paths = []
        frame_count = 0
        chunk_start = 0
        
        while chunk_start < frames_to_process:
            chunk_end = min(chunk_start + chunk_size, frames_to_process)
            print(f"Processing frames {chunk_start} to {chunk_end}...")
            
            frames, timestamps, extracted_indices = extract_frames_chunk(chunk_start, chunk_end)
            
            # Process this chunk of frames
            for i, (frame, timestamp, orig_idx) in enumerate(zip(frames, timestamps, extracted_indices)):
                # Calculate time segments
                if i < len(timestamps) - 1:
                    end_time = timestamps[i + 1]
                else:
                    # For the last frame, add the average duration
                    if i > 0:
                        avg_duration = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
                        end_time = timestamp + avg_duration
                    else:
                        end_time = timestamp + (1 / fps * frame_interval)
                
                # Save frame
                output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(output_path, frame)
                output_paths.append(output_path)
                
                # Store frame information
                frame_data = {
                    "frame": frame_count,
                    "original_frame_idx": orig_idx,
                    "path": f"frame_{frame_count:06d}.jpg",
                    "from_sec": round(timestamp, 2),
                    "to_sec": round(end_time, 2),
                    "smoking": False,
                }
                frame_info.append(frame_data)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Saved {frame_count} frames...")
            
            # Clear memory
            del frames, timestamps, extracted_indices
            gc.collect()
            
            # Move to next chunk
            chunk_start = chunk_end
        
        # Save all frame information to JSON at once
        with open(json_path, "w") as f:
            json.dump(frame_info, f)
            
        print(f"Saved {frame_count} individual frames")
        return output_paths
    
    # For collage mode, we need to determine how many frames we'll extract first
    # to calculate the number of collages
    total_extracted_frames = 0
    for frame_idx in range(0, frames_to_process, frame_interval):
        total_extracted_frames += 1
        
    frames_per_collage = collage_grid[0] * collage_grid[1]
    num_collages = (total_extracted_frames + frames_per_collage - 1) // frames_per_collage
    
    print(f"Will create approximately {num_collages} collages with {frames_per_collage} frames per collage")
    
    # Process collages in chunks
    collage_paths = []
    frame_count = 0
    collage_count = 0
    chunk_start = 0
    
    # Function to process a batch of frames into a collage
    def process_collage(collage_idx, batch_frames, batch_timestamps, batch_indices):
        nonlocal frame_count
        
        # Create a black frame for padding if needed
        if len(batch_frames) > 0:
            black_frame = np.zeros_like(batch_frames[0])
            frame_h, frame_w = batch_frames[0].shape[:2]
        else:
            return None, []  # No frames to process
            
        rows, cols = collage_grid
        
        # If we don't have enough frames, pad with black frames
        while len(batch_frames) < frames_per_collage:
            batch_frames.append(black_frame)
            
        # Create the collage
        collage = np.zeros((frame_h * rows, frame_w * cols, 3), dtype=np.uint8)
        batch_frame_info = []
        
        for i, frame in enumerate(batch_frames):
            if i >= frames_per_collage:
                break
                
            r, c = i // cols, i % cols
            
            # Ensure the frame has the correct dimensions
            if frame.shape[:2] != (frame_h, frame_w):
                frame = cv2.resize(frame, (frame_w, frame_h))
                
            collage[r * frame_h : (r + 1) * frame_h, c * frame_w : (c + 1) * frame_w] = frame
            
            # Only store info for real frames (not padding)
            if i < len(batch_timestamps):
                # Calculate the center position of this frame in the collage
                center_x = c * frame_w + frame_w // 2
                center_y = r * frame_h + frame_h // 2
                
                # Calculate time segments
                timestamp = batch_timestamps[i]
                if i < len(batch_timestamps) - 1:
                    end_time = batch_timestamps[i + 1]
                else:
                    # For the last frame, estimate based on previous frames
                    if i > 0:
                        avg_duration = (batch_timestamps[-1] - batch_timestamps[0]) / (len(batch_timestamps) - 1)
                        end_time = timestamp + avg_duration
                    else:
                        end_time = timestamp + (1 / fps * frame_interval)
                
                # Store frame information
                frame_data = {
                    "frame": frame_count + i,
                    "original_frame_idx": batch_indices[i],
                    "path": f"collage_{collage_idx:06d}.jpg",
                    "from_sec": round(timestamp, 2),
                    "to_sec": round(end_time, 2),
                    "x": center_x,
                    "y": center_y,
                    "collage_idx": collage_idx,
                    "position_in_collage": i,
                    "row": r,
                    "col": c,
                    "smoking": False,
                }
                batch_frame_info.append(frame_data)
        
        # Save the collage
        output_path = os.path.join(output_dir, f"collage_{collage_idx:06d}.jpg")
        cv2.imwrite(output_path, collage)
        
        # Free memory
        del collage
        
        return output_path, batch_frame_info
    
    while chunk_start < frames_to_process:
        chunk_end = min(chunk_start + chunk_size, frames_to_process)
        print(f"Processing frames {chunk_start} to {chunk_end}...")
        
        frames, timestamps, extracted_indices = extract_frames_chunk(chunk_start, chunk_end)
        
        # Process frames into collages
        for i in range(0, len(frames), frames_per_collage):
            batch_frames = frames[i:i+frames_per_collage]
            batch_timestamps = timestamps[i:i+frames_per_collage]
            batch_indices = extracted_indices[i:i+frames_per_collage]
            
            if not batch_frames:
                continue
                
            collage_path, batch_frame_info = process_collage(
                collage_count, batch_frames, batch_timestamps, batch_indices
            )
            
            if collage_path:
                collage_paths.append(collage_path)
                frame_info.extend(batch_frame_info)
                frame_count += len(batch_timestamps)
                collage_count += 1
                
                if collage_count % 10 == 0:
                    print(f"Created {collage_count} collages...")
        
        # Clear memory
        del frames, timestamps, extracted_indices
        gc.collect()
        
        # Move to next chunk
        chunk_start = chunk_end
    
    # Sort frame information by frame number
    frame_info.sort(key=lambda x: x["frame"])
    
    # Save all frame information to JSON at once
    with open(json_path, "w") as f:
        json.dump(frame_info, f)
    
    print(f"Created {collage_count} collages with {frame_count} frames")
    print(f"Frame information saved to {json_path}")
    
    return collage_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split video into frames or collages")
    parser.add_argument("--video", default="video.mp4", help="Path to input video file")
    parser.add_argument("--output-dir", default="output_collages", help="Directory to save output")
    parser.add_argument("--num-frames", type=int, default=1, help="Number of frames to extract per second")
    parser.add_argument("--collage", action="store_true", help="Create collages instead of individual frames")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows in collage")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in collage")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--resize", type=str, help="Resize frames to WxH (e.g., 640x480)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Parse resize dimensions if provided
    resize_frames = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split("x"))
            resize_frames = (width, height)
        except ValueError:
            print(f"Invalid resize format: {args.resize}. Expected WxH (e.g., 640x480)")
            exit(1)
    
    # Set collage grid or None for individual frames
    collage_grid = (args.rows, args.cols) if args.collage else None
    
    # Create collages or individual frames
    output_paths = split_video_and_create_collages(
        video_path=args.video,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        collage_grid=collage_grid,
        max_frames=args.max_frames,
        resize_frames=resize_frames,
        num_workers=args.workers,
    )
    
    print(f"Created {len(output_paths)} output files")
