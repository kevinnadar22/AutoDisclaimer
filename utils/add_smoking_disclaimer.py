import os
import cv2
import json
import argparse
from tqdm import tqdm
import numpy as np


def add_disclaimer_to_video(video_path, frames_info_path, output_path, disclaimer_image_path=None, progress_callback=None):
    """
    Add a smoking disclaimer to a video based on smoking detection results.
    The disclaimer will appear exactly during the timestamps where smoking was detected.
    
    Args:
        video_path: Path to the input video file
        frames_info_path: Path to the JSON file with smoking detection results
        output_path: Path to save the output video
        disclaimer_image_path: Optional path to a custom disclaimer image
        progress_callback: Optional callback function to report progress (0.0 to 1.0)
    """
    # Load smoking detection results
    with open(frames_info_path, "r") as f:
        frames_info = json.load(f)

    # Create a list of smoking segments (from_sec, to_sec)
    smoking_segments = []
    for frame in frames_info:
        if (
            "smoking" in frame
            and frame["smoking"]
            and "from_sec" in frame
            and "to_sec" in frame
        ):
            smoking_segments.append((frame["from_sec"], frame["to_sec"]))
        elif "smoking" in frame and frame["smoking"]:
            # If from_sec and to_sec are not available, use frame number and fps
            frame_num = frame.get("frame_num", 0)
            fps = frame.get("fps", 30)  # Default to 30 fps if not available
            time_sec = frame_num / fps
            smoking_segments.append((time_sec, time_sec + 1/fps))  # Add a small segment

    if not smoking_segments:
        print("No smoking detected in the video. No disclaimer needed.")
        if progress_callback:
            progress_callback(1.0)  # Complete the progress
        return

    print(f"Found {len(smoking_segments)} smoking segments")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {total_frames} frames, {fps} FPS, Resolution: {width}x{height}")

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load disclaimer image if provided
    disclaimer_image = None
    if disclaimer_image_path and os.path.exists(disclaimer_image_path):
        try:
            disclaimer_image = cv2.imread(disclaimer_image_path, cv2.IMREAD_UNCHANGED)
            # Resize image to a reasonable size (e.g., 1/4 of the video width)
            target_width = width // 4
            aspect_ratio = disclaimer_image.shape[1] / disclaimer_image.shape[0]
            target_height = int(target_width / aspect_ratio)
            disclaimer_image = cv2.resize(disclaimer_image, (target_width, target_height))
            
            # If image has alpha channel, handle it properly
            if disclaimer_image.shape[2] == 4:
                # Split the image into color and alpha channels
                bgr = disclaimer_image[:, :, :3]
                alpha = disclaimer_image[:, :, 3]
                # Create a normalized alpha channel
                alpha_norm = alpha / 255.0
                # Store the processed image and alpha
                disclaimer_image = (bgr, alpha_norm)
            else:
                # If no alpha channel, just use the image as is
                disclaimer_image = (disclaimer_image, None)
        except Exception as e:
            print(f"Error loading disclaimer image: {e}")
            disclaimer_image = None

    # Set up disclaimer text
    disclaimer_text = "WARNING: This video contains smoking, which is harmful to health"
    font = cv2.FONT_HERSHEY_DUPLEX  # More movie-like font
    font_scale = 0.7  # Slightly smaller for bottom left
    thickness = 2

    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(
        disclaimer_text, font, font_scale, thickness
    )

    # Calculate position (bottom left with padding)
    padding = 20  # Padding from edges
    text_x = padding
    text_y = height - padding  # Position from bottom

    # Process the video
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate current time in video
            current_time = frame_count / fps

            # Check if current time is in any smoking segment
            show_disclaimer = False
            for from_sec, to_sec in smoking_segments:
                if from_sec <= current_time <= to_sec:
                    show_disclaimer = True
                    break

            # Add disclaimer if needed
            if show_disclaimer:
                if disclaimer_image is not None:
                    # Use custom disclaimer image
                    img, alpha = disclaimer_image
                    
                    # Calculate position (bottom left corner with padding)
                    img_x = padding
                    img_y = height - img.shape[0] - padding
                    
                    if alpha is not None:
                        # If we have an alpha channel, blend the image
                        for c in range(0, 3):
                            frame[img_y:img_y+img.shape[0], img_x:img_x+img.shape[1], c] = \
                                frame[img_y:img_y+img.shape[0], img_x:img_x+img.shape[1], c] * (1 - alpha) + \
                                img[:, :, c] * alpha
                    else:
                        # Simple overlay without alpha blending
                        frame[img_y:img_y+img.shape[0], img_x:img_x+img.shape[1]] = img
                else:
                    # Use text disclaimer
                    # Draw black background with gradient fade
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay,
                        (text_x - 10, text_y - text_height - 10),
                        (text_x + text_width + 10, text_y + 10),
                        (0, 0, 0),
                        -1,
                    )
                    # Apply gradient alpha for smoother look
                    alpha = 0.7
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # Draw text with border for better visibility
                    # Draw black border
                    cv2.putText(
                        frame,
                        disclaimer_text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness + 2,
                        cv2.LINE_AA,
                    )
                    # Draw white text
                    cv2.putText(
                        frame,
                        disclaimer_text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

            # Write the frame
            out.write(frame)

            # Update progress
            frame_count += 1
            pbar.update(1)
            
            # Call progress callback if provided
            if progress_callback and total_frames > 0:
                progress_callback(frame_count / total_frames)

    # Release resources
    cap.release()
    out.release()

    print(f"Video with disclaimer saved to: {output_path}")
    
    # Final progress update
    if progress_callback:
        progress_callback(1.0)


def main():
    parser = argparse.ArgumentParser(description="Add smoking disclaimer to video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--frames-info",
        default="output_collages/frames_info.json",
        help="Path to JSON file with smoking detection results",
    )
    parser.add_argument(
        "--output",
        default="video_with_disclaimer.mp4",
        help="Path to output video file",
    )
    parser.add_argument(
        "--disclaimer-image",
        help="Optional path to a custom disclaimer image",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Add disclaimer to video
    add_disclaimer_to_video(
        video_path=args.video,
        frames_info_path=args.frames_info,
        output_path=args.output,
        disclaimer_image_path=args.disclaimer_image,
    )


if __name__ == "__main__":
    main()
