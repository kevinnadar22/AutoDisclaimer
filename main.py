import cv2
import numpy as np
from PIL import Image, ImageDraw

# delete al the images in the current directory
import os
os.system("rm -rf *.jpg")

def extract_and_create_collage(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Extract 5 frames per second
    frames_per_second = 5
    frame_interval = fps // frames_per_second
    total_frames_to_extract = int(duration * frames_per_second)
    
    # Calculate number of frames per collage (6 frames per collage)
    frames_per_collage = 6
    num_collages = total_frames_to_extract // frames_per_collage
    
    for collage_num in range(num_collages):
        frames = []
        start_frame = collage_num * frames_per_collage * frame_interval
        
        # Extract 6 frames for this collage
        for i in range(frames_per_collage):
            frame_pos = start_frame + (i * frame_interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        if len(frames) == frames_per_collage:
            # Create collage
            # Resize frames to consistent size
            target_size = (320, 180)  # You can adjust this size
            resized_frames = [cv2.resize(f, target_size) for f in frames]
            
            # Create a 2x3 grid
            rows, cols = 2, 3
            collage = Image.new('RGB', (target_size[0] * cols, target_size[1] * rows))
            draw = ImageDraw.Draw(collage)
            
            # Place frames and add annotations
            for idx, frame in enumerate(resized_frames):
                frame_img = Image.fromarray(frame)
                x = (idx % cols) * target_size[0]
                y = (idx // cols) * target_size[1]
                collage.paste(frame_img, (x, y))

            # Save the collage
            collage.save(f"collage_{collage_num + 1}.jpg")
    
    cap.release()

# Usage
video_path = "video.mp4"
extract_and_create_collage(video_path)
