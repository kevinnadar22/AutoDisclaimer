import gradio as gr
import os
import json
import time
import shutil
import tempfile
from pathlib import Path
from utils import (
    split_video_and_create_collages,
    load_model,
    process_image,
    process_video_with_disclaimer,
)
from huggingface_hub import login

# Initialize the model globally for reuse
model = None

def initialize_model():
    global model
    if model is None:
        print("Loading model...")
        model = load_model()
    return model

def process_video_file(
    video_file, 
    frames_per_second: int = 1, 
    model_endpoint: str = "point",
    disclaimer_image = None,
    progress=gr.Progress()
):
    """Process a video file and detect smoking scenes"""
    try:
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Save video file
        video_path = os.path.join(temp_dir, "input_video.mp4")
        shutil.copy2(video_file.name, video_path)

        # Save disclaimer image if provided
        disclaimer_path = None
        if disclaimer_image is not None:
            disclaimer_path = os.path.join(temp_dir, "disclaimer.png")
            shutil.copy2(disclaimer_image.name, disclaimer_path)

        # Initialize model
        model = initialize_model()

        # Extract frames
        progress(0.05, desc="Extracting frames...")
        frame_info = split_video_and_create_collages(
            video_path=video_path,
            output_dir=frames_dir,
            num_frames=frames_per_second,
            collage_grid=None,  # None for individual frames
            resize_frames=None,  # Use original frame sizes
        )

        # Load frames info
        frames_info_path = os.path.join(frames_dir, "frames_info.json")
        if not os.path.exists(frames_info_path):
            raise Exception("Failed to extract frames from video")

        with open(frames_info_path, "r") as f:
            frames_data = json.load(f)

        total_frames = len(frames_data)
        smoking_frames = []
        smoking_count = 0

        # Process each frame for smoking detection
        progress(0.3, desc="Detecting smoking...")
        for i, frame_info in enumerate(frames_data):
            frame_path = os.path.join(frames_dir, frame_info["path"])
            
            # Process image with specified model endpoint
            frames_data, _, smoking_detected = process_image(
                model=model,
                image_path=frame_path,
                frames_data=frames_data,
                detect_only=True,
                model_endpoint=model_endpoint,
            )

            if smoking_detected:
                smoking_count += 1
                smoking_frames.append(frame_info["from_sec"])

            # Update progress
            progress((0.3 + (i / total_frames * 0.4)), 
                    desc=f"Analyzing frame {i+1}/{total_frames} | Found {smoking_count} smoking scenes")

        # Add disclaimers to video
        progress(0.7, desc="Adding disclaimers...")
        output_path = os.path.join(temp_dir, "output_video.mp4")
        process_video_with_disclaimer(
            input_video=video_path,
            output_video=output_path,
            smoking_timestamps=smoking_frames,
            disclaimer_image=disclaimer_path,
            disclaimer_text="Warning: This video contains scenes of smoking" if not disclaimer_path else None,
        )

        # Calculate statistics
        smoking_percentage = (smoking_count / total_frames * 100) if total_frames > 0 else 0
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        # Create summary
        summary = f"""
### Processing Summary
- Total frames analyzed: {total_frames}
- Smoking detected in: {smoking_count} frames ({smoking_percentage:.1f}%)
- Frames per second analyzed: {frames_per_second}
- Output file size: {file_size_mb:.2f} MB
        """

        progress(1.0, desc="Processing complete!")
        return output_path, summary

    except Exception as e:
        raise gr.Error(f"Error processing video: {str(e)}")
    finally:
        # Clean up temporary directory
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Smoking Detection & Disclaimer Adder", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # Smoking Detection & Disclaimer Adder
        Upload a video to detect smoking and add disclaimers
        """)

        with gr.Row():
            with gr.Column():
                # Input components
                video_input = gr.Video(
                    label="Upload Video",
                    sources="upload",
                    # type="filepath",
                    interactive=True
                )
                
                with gr.Row():
                    fps_slider = gr.Slider(
                        minimum=1,
                        maximum=24,
                        value=1,
                        step=1,
                        label="Frames per second to analyze",
                        info="Higher values provide more accurate detection but increase processing time"
                    )
                    
                    model_endpoint = gr.Dropdown(
                        choices=["point", "detect", "query"],
                        value="point",
                        label="Detection Method",
                        info="Different methods may have varying accuracy and speed"
                    )

                disclaimer_image = gr.Image(
                    label="Disclaimer Image (optional)",
                    type="filepath",
                    interactive=True
                )

                process_btn = gr.Button(
                    "Process Video", 
                    variant="primary",
                    scale=1
                )

            with gr.Column():
                # Output components
                output_video = gr.Video(
                    label="Processed Video",
                    interactive=False
                )
                output_summary = gr.Markdown(
                    label="Processing Summary"
                )

        # Handle processing
        process_btn.click(
            fn=process_video_file,
            inputs=[
                video_input,
                fps_slider,
                model_endpoint,
                disclaimer_image
            ],
            outputs=[
                output_video,
                output_summary
            ]
        )

    return app

if __name__ == "__main__":
    # Login to Hugging Face
    login("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")
    
    # Launch the app
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 