import streamlit as st
import os
import cv2
import json
import shutil
from PIL import Image
import numpy as np
from split import split_video_and_create_collages
from process_smoking_detection import load_model, process_image
from tqdm import tqdm
import time
import uuid
import datetime


def process_video_with_disclaimer(
    video_path, frames_info_path, output_path, disclaimer_image_path=None, progress_callback=None
):
    """Add disclaimer to video at smoking timestamps with optimized export"""
    # Load smoking detection results
    with open(frames_info_path, "r") as f:
        frames_info = json.load(f)

    # Create a list of smoking segments
    smoking_segments = []
    for frame in frames_info:
        if (
            "smoking" in frame
            and frame["smoking"]
            and "from_sec" in frame
            and "to_sec" in frame
        ):
            smoking_segments.append((frame["from_sec"], frame["to_sec"]))

    if not smoking_segments:
        st.warning(
            "No smoking detected in the video. Creating output without disclaimers."
        )
        smoking_count = 0
    else:
        smoking_count = len(smoking_segments)
        st.info(f"Found {smoking_count} smoking segments")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(
        f"Video info: {total_frames} frames, {fps} FPS, Resolution: {width}x{height}"
    )

    # Create VideoWriter object with optimized settings
    # Use H.264 codec for better compatibility and compression
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4v for compatibility
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load and prepare disclaimer image if provided
    disclaimer_img = None
    alpha = None
    if disclaimer_image_path and os.path.exists(disclaimer_image_path):
        st.text("Loading and preparing disclaimer image...")
        disclaimer_img = cv2.imread(disclaimer_image_path, cv2.IMREAD_UNCHANGED)
        if disclaimer_img is not None:
            # Resize disclaimer image to reasonable size (20% of video height)
            target_height = int(height * 0.2)
            aspect_ratio = disclaimer_img.shape[1] / disclaimer_img.shape[0]
            target_width = int(target_height * aspect_ratio)
            disclaimer_img = cv2.resize(disclaimer_img, (target_width, target_height))

            # If image has alpha channel, use it for transparency
            if disclaimer_img.shape[2] == 4:
                alpha = disclaimer_img[:, :, 3] / 255.0
                disclaimer_img = disclaimer_img[:, :, :3]
            else:
                alpha = np.ones((disclaimer_img.shape[0], disclaimer_img.shape[1]))
            st.text("Disclaimer image prepared successfully")

    # Set up default text disclaimer
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

    # Calculate position for image disclaimer (bottom left)
    if disclaimer_img is not None:
        img_x = 20  # 20 pixels from left
        img_y = height - disclaimer_img.shape[0] - 20  # 20 pixels from bottom

    # Progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process the video
    frame_count = 0
    smoking_frames = 0
    start_time = time.time()
    
    # Process in batches for better performance
    batch_size = 30  # Process 30 frames at a time for better I/O performance
    
    while True:
        frames_batch = []
        positions = []
        
        # Read a batch of frames
        for _ in range(batch_size):
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
                    smoking_frames += 1
                    break
            
            frames_batch.append((frame, show_disclaimer))
            positions.append(frame_count)
            frame_count += 1
        
        if not frames_batch:
            break
        
        # Process the batch
        for i, (frame, show_disclaimer) in enumerate(frames_batch):
            # Add disclaimer if needed
            if show_disclaimer:
                if disclaimer_img is not None and alpha is not None:
                    # Add image disclaimer
                    for c in range(3):  # RGB channels
                        frame[
                            img_y : img_y + disclaimer_img.shape[0],
                            img_x : img_x + disclaimer_img.shape[1],
                            c,
                        ] = (
                            frame[
                                img_y : img_y + disclaimer_img.shape[0],
                                img_x : img_x + disclaimer_img.shape[1],
                                c,
                            ]
                            * (1 - alpha)
                            + disclaimer_img[:, :, c] * alpha
                        )
                else:
                    # Add text disclaimer with gradient fade
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay,
                        (text_x - 10, text_y - text_height - 10),
                        (text_x + text_width + 10, text_y + 10),
                        (0, 0, 0),
                        -1
                    )
                    # Apply gradient alpha for smoother look
                    alpha_value = 0.7
                    frame = cv2.addWeighted(overlay, alpha_value, frame, 1 - alpha_value, 0)
                    
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
        
        # Update progress and stats
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Calculate time remaining and processing speed
        elapsed_time = time.time() - start_time
        frames_remaining = total_frames - frame_count
        if frame_count > 0:
            fps_processing = frame_count / elapsed_time
            time_remaining = frames_remaining / fps_processing
            
            # Format time remaining as MM:SS
            mins_remaining = int(time_remaining // 60)
            secs_remaining = int(time_remaining % 60)
            
            # Update status with frame count, smoking percentage, FPS, and ETA
            status_text.text(
                f"Processing: {frame_count}/{total_frames} frames | "
                f"Smoking: {smoking_frames} frames ({smoking_frames/frame_count*100:.1f}%) | "
                f"Speed: {fps_processing:.1f} FPS | "
                f"ETA: {mins_remaining:02d}:{secs_remaining:02d}"
            )
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(progress)

    # Release resources
    cap.release()
    out.release()

    st.success(f"Video processing complete! {smoking_frames} frames contained smoking.")
    return output_path, smoking_count


def main():
    st.title("Smoking Detection & Disclaimer Adder")
    st.write("Upload a video to detect smoking and add disclaimers")

    # Create project directories
    downloads_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    
    # File uploader for video
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        # Generate a unique ID for this session
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(downloads_dir, f"{timestamp}_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Save uploaded video to session directory
        video_filename = f"input_{os.path.splitext(video_file.name)[0]}.mp4"
        video_path = os.path.join(session_dir, video_filename)
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        
        st.success(f"Video uploaded successfully: {video_filename}")

        # Parameters for frame extraction
        frames_per_second = st.slider("Frames per second to analyze", 1, 24, 1, 
                                     help="Higher values provide more accurate detection but increase processing time")

        # File uploader for disclaimer image
        disclaimer_image = st.file_uploader(
            "Upload Disclaimer Image (optional)", type=["png", "jpg", "jpeg"]
        )
        disclaimer_image_path = None
        if disclaimer_image:
            disclaimer_image_path = os.path.join(session_dir, "disclaimer.png")
            # Convert to PNG to preserve transparency
            img = Image.open(disclaimer_image)
            img.save(disclaimer_image_path, "PNG")
            st.image(img, caption="Disclaimer Image", width=300)

        if st.button("Process Video"):
            try:
                with st.spinner("Initializing..."):
                    # Create output directories
                    output_dir = os.path.join(session_dir, "frames")
                    os.makedirs(output_dir, exist_ok=True)

                    # Step 1: Split video into frames
                    progress_text = st.empty()
                    progress_text.text("Step 1/3: Extracting frames...")
                    
                    # Calculate max frames based on video length to avoid excessive processing
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps
                    cap.release()
                    
                    # Limit max frames for very long videos
                    max_frames = None
                    if duration > 600:  # If video is longer than 10 minutes
                        max_frames = int(600 * fps)  # Process only first 10 minutes
                        st.warning(f"Video is {duration:.1f} seconds long. Processing only the first 10 minutes to save time.")
                    
                    # Extract frames
                    split_video_and_create_collages(
                        video_path=video_path,
                        output_dir=output_dir,
                        num_frames=frames_per_second,
                        collage_grid=None,  # Don't create collages
                        max_frames=max_frames,
                        resize_frames=(640, 360)  # Resize frames for faster processing
                    )

                    # Step 2: Load model and process frames
                    progress_text.text(
                        "Step 2/3: Loading AI model and detecting smoking..."
                    )
                    model = load_model()

                    # Load frames info
                    frames_info_path = os.path.join(output_dir, "frames_info.json")
                    with open(frames_info_path, "r") as f:
                        frames_data = json.load(f)

                    # Process each frame with progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_frames = len(frames_data)
                    smoking_count = 0
                    start_time = time.time()

                    # Process frames
                    for idx, frame in enumerate(frames_data):
                        image_path = os.path.join(output_dir, frame["path"])
                        if os.path.exists(image_path):
                            frames_data, _, smoking_detected = process_image(
                                model, image_path, frames_data, detect_only=True
                            )
                            if smoking_detected:
                                smoking_count += 1
                                
                        # Update progress
                        progress = (idx + 1) / total_frames
                        progress_bar.progress(progress)
                        
                        # Calculate ETA and FPS
                        elapsed_time = time.time() - start_time
                        if idx > 0:
                            fps_processing = (idx + 1) / elapsed_time
                            frames_remaining = total_frames - (idx + 1)
                            time_remaining = frames_remaining / fps_processing
                            
                            # Format time remaining as MM:SS
                            mins_remaining = int(time_remaining // 60)
                            secs_remaining = int(time_remaining % 60)
                            
                            status_text.text(
                                f"Analyzing: {idx + 1}/{total_frames} frames | "
                                f"Smoking: {smoking_count} frames ({smoking_count/(idx+1)*100:.1f}%) | "
                                f"Speed: {fps_processing:.1f} FPS | "
                                f"ETA: {mins_remaining:02d}:{secs_remaining:02d}"
                            )

                    # Save updated frames info
                    with open(frames_info_path, "w") as f:
                        json.dump(frames_data, f)

                    # Step 3: Add disclaimer to video
                    progress_text.text("Step 3/3: Adding disclaimers to video...")
                    output_path = os.path.join(session_dir, "processed_video.mp4")
                    
                    final_video, smoking_count = process_video_with_disclaimer(
                        video_path=video_path,
                        frames_info_path=frames_info_path,
                        output_path=output_path,
                        disclaimer_image_path=disclaimer_image_path,
                    )

                    # Create a download link for the processed video
                    if os.path.exists(output_path):
                        # Get file size
                        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        
                        st.success(f"Processing complete!")
                        
                        # Create download button
                        with open(output_path, "rb") as f:
                            st.download_button(
                                "Download Processed Video",
                                f,
                                file_name=f"smoking_disclaimer_{timestamp}.mp4",
                                mime="video/mp4",
                            )
                        
                        # Show video preview
                        st.video(output_path)
                        
                        # Show processing summary
                        st.subheader("Processing Summary")
                        st.write(f"• Total frames analyzed: {total_frames}")
                        st.write(f"• Smoking detected in: {smoking_count} frames ({smoking_count/total_frames*100:.1f}%)")
                        st.write(f"• Frames per second analyzed: {frames_per_second}")
                        st.write(f"• Output file size: {file_size_mb:.2f} MB")
                    else:
                        st.error("Error: Failed to create output video")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
