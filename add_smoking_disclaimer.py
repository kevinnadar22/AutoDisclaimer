import os
import cv2
import json
import argparse
from tqdm import tqdm


def add_disclaimer_to_video(video_path, frames_info_path, output_path):
    """
    Add a smoking disclaimer to a video based on smoking detection results.
    The disclaimer will appear exactly during the timestamps where smoking was detected.
    """
    # Load smoking detection results
    with open(frames_info_path, "r") as f:
        frames_info = json.load(f)
    
    # Create a list of smoking segments (from_sec, to_sec)
    smoking_segments = []
    for frame in frames_info:
        if "smoking" in frame and frame["smoking"] and "from_sec" in frame and "to_sec" in frame:
            smoking_segments.append((frame["from_sec"], frame["to_sec"]))
    
    if not smoking_segments:
        print("No smoking detected in the video. No disclaimer needed.")
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
    
    # Set up disclaimer text
    disclaimer_text = "WARNING: This video contains smoking, which is harmful to health"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Calculate text size
    (text_width, text_height), _ = cv2.getTextSize(
        disclaimer_text, font, font_scale, thickness
    )
    
    # Calculate position (centered at bottom)
    text_x = (width - text_width) // 2
    text_y = height - 50
    
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
                # Draw black background
                cv2.rectangle(
                    frame,
                    (text_x - 10, text_y - text_height - 10),
                    (text_x + text_width + 10, text_y + 10),
                    (0, 0, 0),
                    -1,
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
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video with disclaimer saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Add smoking disclaimer to video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--frames-info",
        default="output_collages/frames_info.json",
        help="Path to JSON file with smoking detection results",
    )
    parser.add_argument(
        "--output", default="video_with_disclaimer.mp4", help="Path to output video file"
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
    )


if __name__ == "__main__":
    main() 