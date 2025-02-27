import streamlit as st
import os
import cv2
import json
import tempfile
from PIL import Image
import numpy as np
from split import split_video_and_create_collages
from process_smoking_detection import load_model, process_image
from tqdm import tqdm


def process_video_with_disclaimer(
    video_path, frames_info_path, output_path, disclaimer_image_path=None
):
    """Add disclaimer to video at smoking timestamps"""
    # Load smoking detection results
    with open(frames_info_path, "r") as f:
        frames_info = json.load(f)
    
    # Create a list of smoking segments
    smoking_segments = []
    for frame in frames_info:
        if "smoking" in frame and frame["smoking"] and "from_sec" in frame and "to_sec" in frame:
            smoking_segments.append((frame["from_sec"], frame["to_sec"]))
    
    if not smoking_segments:
        st.warning("No smoking detected in the video. No disclaimer needed.")
        return
    
    st.info(f"Found {len(smoking_segments)} smoking segments")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info(f"Video info: {total_frames} frames, {fps} FPS, Resolution: {width}x{height}")
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Load and prepare disclaimer image if provided
    disclaimer_img = None
    if disclaimer_image_path:
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
    
    # Set up default text disclaimer
    disclaimer_text = "WARNING: This video contains smoking, which is harmful to health"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Calculate text size
    (text_width, text_height), _ = cv2.getTextSize(
        disclaimer_text, font, font_scale, thickness
    )
    
    # Calculate position for text disclaimer (centered at bottom)
    text_x = (width - text_width) // 2
    text_y = height - 50
    
    # Calculate position for image disclaimer (bottom left)
    if disclaimer_img is not None:
        img_x = 20  # 20 pixels from left
        img_y = height - disclaimer_img.shape[0] - 20  # 20 pixels from bottom
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process the video
    frame_count = 0
    
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
            if disclaimer_img is not None:
                # Add image disclaimer
                for c in range(3):  # RGB channels
                    frame[img_y:img_y + disclaimer_img.shape[0], 
                          img_x:img_x + disclaimer_img.shape[1], c] = \
                        frame[img_y:img_y + disclaimer_img.shape[0], 
                              img_x:img_x + disclaimer_img.shape[1], c] * (1 - alpha) + \
                        disclaimer_img[:, :, c] * alpha
            else:
                # Add text disclaimer
                cv2.rectangle(
                    frame,
                    (text_x - 10, text_y - text_height - 10),
                    (text_x + text_width + 10, text_y + 10),
                    (0, 0, 0),
                    -1,
                )
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
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    # Release resources
    cap.release()
    out.release()
    
    st.success(f"Video with disclaimer saved to: {output_path}")
    return output_path


def main():
    st.title("Smoking Detection & Disclaimer Adder")
    st.write("Upload a video to detect smoking and add disclaimers")
    
    # File uploader for video
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if video_file:
        # Save uploaded video to temp file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        
        # Parameters for frame extraction
        st.subheader("Frame Extraction Settings")
        frames_per_second = st.slider("Frames per second to analyze", 1, 30, 1)
        
        # File uploader for disclaimer image
        disclaimer_image = st.file_uploader("Upload Disclaimer Image (optional)", type=["png", "jpg", "jpeg"])
        disclaimer_image_path = None
        if disclaimer_image:
            disclaimer_image_path = os.path.join(temp_dir, "disclaimer.png")
            # Convert to PNG to preserve transparency
            img = Image.open(disclaimer_image)
            img.save(disclaimer_image_path, "PNG")
        
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                # Create output directories
                output_dir = os.path.join(temp_dir, "output_collages")
                os.makedirs(output_dir, exist_ok=True)
                
                # Step 1: Split video into frames
                st.text("Step 1: Extracting frames...")
                split_video_and_create_collages(
                    video_path=video_path,
                    output_dir=output_dir,
                    num_frames=frames_per_second,
                    collage_grid=None  # Don't create collages
                )
                
                # Step 2: Load model and process frames
                st.text("Step 2: Detecting smoking in frames...")
                model = load_model()
                
                # Load frames info
                with open(os.path.join(output_dir, "frames_info.json"), "r") as f:
                    frames_data = json.load(f)
                
                # Process each frame
                for frame in frames_data:
                    image_path = os.path.join(output_dir, frame["path"])
                    if os.path.exists(image_path):
                        frames_data, _, _ = process_image(
                            model, image_path, frames_data, detect_only=True
                        )
                
                # Save updated frames info
                with open(os.path.join(output_dir, "frames_info.json"), "w") as f:
                    json.dump(frames_data, f, indent=2)
                
                # Step 3: Add disclaimer to video
                st.text("Step 3: Adding disclaimer to video...")
                output_path = os.path.join(temp_dir, "output_video.mp4")
                final_video = process_video_with_disclaimer(
                    video_path=video_path,
                    frames_info_path=os.path.join(output_dir, "frames_info.json"),
                    output_path=output_path,
                    disclaimer_image_path=disclaimer_image_path
                )
                
                # Provide download link
                if os.path.exists(final_video):
                    with open(final_video, "rb") as f:
                        st.download_button(
                            "Download Processed Video",
                            f,
                            file_name="video_with_disclaimer.mp4",
                            mime="video/mp4"
                        )


if __name__ == "__main__":
    main() 