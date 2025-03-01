# Automatic Smoking Disclaimer Adder

This application automatically detects smoking scenes in videos and adds appropriate disclaimers. It uses a deep learning model to identify smoking content and overlays a warning message during those specific timestamps.

## Features

- **Smoking Detection**: Uses the Moondream2 vision-language model to detect smoking in video frames
- **Automatic Disclaimer**: Adds a professional-looking disclaimer during smoking scenes only
- **User-Friendly Interface**: Streamlit web UI for easy video processing
- **GPU Acceleration**: Optimized for NVIDIA GPUs for faster processing
- **Customization Options**: Adjust batch size, image dimensions, and disclaimer appearance

## Quick Start

The easiest way to run the application is using the provided setup script:

```bash
# Make the script executable
chmod +x setup_and_run.sh

# Run the setup script
./setup_and_run.sh
```

This will:
1. Check for Docker and NVIDIA GPU availability
2. Build the Docker image with all dependencies
3. Start the Streamlit web application
4. Provide usage instructions

Once running, open your browser and go to: http://localhost:8501

## Manual Setup

If you prefer to set up manually:

### Using Docker (recommended)

```bash
# Build the Docker image
docker-compose build

# Run the application
docker-compose up
```

### Without Docker

1. Install Python 3.9 and dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. **Upload Video**: Select a video file to process
2. **Configure Settings**: Adjust frame extraction rate and batch size if needed
3. **Process Video**: Click the "Process Video" button
4. **Download Result**: Once processing is complete, download the video with disclaimers

## Components

- `process_smoking_detection.py`: Handles smoking detection using the Moondream2 model
- `add_smoking_disclaimer.py`: Adds disclaimers to videos at specific timestamps
- `app.py`: Streamlit web interface
- `Dockerfile` and `docker-compose.yml`: Container configuration

## Performance Tips

- **GPU Memory**: Reduce batch size if you encounter CUDA out of memory errors
- **Processing Speed**: Larger batch sizes generally improve performance on GPUs with sufficient memory
- **Image Size**: Use the max image size option to reduce memory usage for high-resolution videos

## License

[MIT License](LICENSE)

## Acknowledgments

- Uses the [Moondream2](https://huggingface.co/vikhyatk/moondream2) model for smoking detection 