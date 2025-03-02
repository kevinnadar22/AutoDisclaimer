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
    start_time = time.time()
    try:
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Handle video file path (could be string or file object)
        video_path = os.path.join(temp_dir, "input_video.mp4")
        if isinstance(video_file, str):
            shutil.copy2(video_file, video_path)
        else:
            shutil.copy2(video_file.name, video_path)

        # Handle disclaimer image (could be string or file object)
        disclaimer_path = None
        if disclaimer_image is not None:
            disclaimer_path = os.path.join(temp_dir, "disclaimer.png")
            if isinstance(disclaimer_image, str):
                shutil.copy2(disclaimer_image, disclaimer_path)
            else:
                shutil.copy2(disclaimer_image.name, disclaimer_path)

        # Initialize model
        model_load_start = time.time()
        model = initialize_model()
        model_load_time = time.time() - model_load_start

        # Extract frames
        progress(0.05, desc="Extracting frames...")
        frame_extraction_start = time.time()
        frame_info = split_video_and_create_collages(
            video_path=video_path,
            output_dir=frames_dir,
            num_frames=frames_per_second,
            collage_grid=None,  # None for individual frames
            resize_frames=None,  # Use original frame sizes
        )
        frame_extraction_time = time.time() - frame_extraction_start

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
        detection_start = time.time()
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

        detection_time = time.time() - detection_start

        # Add disclaimers to video
        progress(0.7, desc="Adding disclaimers...")
        disclaimer_start = time.time()
        output_path = os.path.join(temp_dir, "output_video.mp4")
        process_video_with_disclaimer(
            input_video=video_path,
            output_video=output_path,
            smoking_timestamps=smoking_frames,
            disclaimer_image=disclaimer_path,
            disclaimer_text="Warning: This video contains scenes of smoking" if not disclaimer_path else None,
        )
        disclaimer_time = time.time() - disclaimer_start

        # Calculate statistics
        total_time = time.time() - start_time
        smoking_percentage = (smoking_count / total_frames * 100) if total_frames > 0 else 0
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        fps_achieved = total_frames / detection_time if detection_time > 0 else 0

        # Create detailed summary
        summary = f"""
### Processing Summary 📊

#### Detection Results
- Total frames analyzed: **{total_frames}**
- Smoking detected in: **{smoking_count} frames ({smoking_percentage:.1f}%)**
- Frames per second analyzed: **{frames_per_second}**
- Output file size: **{file_size_mb:.2f} MB**

#### Performance Metrics ⚡
- Model loading time: **{model_load_time:.2f}s**
- Frame extraction time: **{frame_extraction_time:.2f}s**
- Detection time: **{detection_time:.2f}s**
  - Average processing speed: **{fps_achieved:.1f} FPS**
- Disclaimer addition time: **{disclaimer_time:.2f}s**
- Total processing time: **{total_time:.2f}s**

#### Configuration Used 🛠️
- Detection method: **{model_endpoint}**
- Input FPS: **{frames_per_second}**
- Disclaimer image: **{'Yes' if disclaimer_image else 'No'}**
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
    # Custom theme
    theme = gr.themes.Soft().set(
        font=["Poppins", "Inter", "system-ui", "sans-serif"],
        font_mono=["IBM Plex Mono", "Consolas", "Monaco", "monospace"],
        
        # Primary colors
        primary_hue="indigo",
        secondary_hue="blue",
        
        # Component colors
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        
        # Spacing and radius
        spacing_md=8,
        radius_md=12,
        
        # Text styles
        text_md=gr.themes.Size(
            font_size="16px",
            line_height="24px"
        ),
        
        # Border styles
        border_color_primary="*primary_300"
    )

    with gr.Blocks(title="Smoking Detection & Disclaimer Adder", theme=theme) as app:
        gr.Markdown("""
        # 🚬 Smoking Detection & Disclaimer Adder
        Upload a video to detect smoking scenes and automatically add disclaimers
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                with gr.Group():
                    gr.Markdown("### 📤 Upload")
                    video_input = gr.Video(
                        label="Upload Video",
                        sources="upload",
                        interactive=True
                    )
                
                with gr.Group():
                    gr.Markdown("### ⚙️ Settings")
                    with gr.Row():
                        fps_slider = gr.Slider(
                            minimum=1,
                            maximum=24,
                            value=1,
                            step=1,
                            label="Frames per second",
                            info="Higher values = better accuracy, longer processing",
                        )
                        
                        model_endpoint = gr.Dropdown(
                            choices=["point", "detect", "query"],
                            value="point",
                            label="Detection Method",
                            info="Different methods have varying accuracy/speed"
                        )

                    disclaimer_image = gr.Image(
                        label="Disclaimer Image (optional)",
                        type="filepath",
                        interactive=True
                    )

                process_btn = gr.Button(
                    "🎬 Process Video", 
                    variant="primary",
                    scale=1,
                    size="lg"
                )

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📊 Results")
                    output_video = gr.Video(
                        label="Processed Video",
                        interactive=False,
                        show_label=True,
                    )
                    output_summary = gr.Markdown()

        # Add examples if you have any
        gr.Examples(
            examples=[
                ["path/to/example1.mp4", 1, "point", None],
                ["path/to/example2.mp4", 2, "detect", "path/to/disclaimer.png"],
            ],
            inputs=[video_input, fps_slider, model_endpoint, disclaimer_image],
            outputs=[output_video, output_summary],
            fn=process_video_file,
            cache_examples=True,
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