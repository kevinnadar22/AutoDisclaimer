# Auto Disclaimer Adder

This repository contains tools for automatically detecting smoking in videos and adding appropriate disclaimers.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The workflow consists of three main steps:

1. Split the video into frames
2. Detect smoking in the frames
3. Add disclaimers to the original video

### 1. Split Video into Frames

Use the `split.py` script to extract frames from a video:

```bash
python split.py
```

This will extract frames from the video and save them in the `output_collages` directory along with a `frames_info.json` file containing metadata.

### 2. Detect Smoking in Frames

Use the `process_smoking_detection.py` script to detect smoking in the extracted frames:

```bash
python process_smoking_detection.py --input-dir output_collages --output-dir annotated_images
```

This will process each frame, detect smoking, and update the `frames_info.json` file with smoking detection results.

For better performance, you can adjust the batch size and image size:

```bash
python process_smoking_detection.py --batch-size 4 --max-image-size 800
```

### 3. Add Smoking Disclaimers to Video

Use the new `add_smoking_disclaimer.py` script to add disclaimers to the original video based on the smoking detection results:

```bash
python add_smoking_disclaimer.py --video video.mp4 --frames-info output_collages/frames_info.json --output video_with_disclaimer.mp4
```

This will add a disclaimer overlay whenever smoking is detected in the video.

#### Customizing the Disclaimer

You can customize the disclaimer appearance and behavior with various options:

```bash
python add_smoking_disclaimer.py --video video.mp4 \
    --disclaimer-text "WARNING: Smoking is injurious to health" \
    --disclaimer-duration 7.0 \
    --disclaimer-position top \
    --disclaimer-bg-opacity 0.8 \
    --disclaimer-font-scale 1.2
```

Available options:

- `--disclaimer-text`: Text to display as disclaimer
- `--disclaimer-duration`: How long to show the disclaimer after smoking is detected (seconds)
- `--disclaimer-position`: Position of disclaimer (top, bottom, center)
- `--disclaimer-bg-opacity`: Background opacity (0-1)
- `--disclaimer-font-scale`: Font scale for disclaimer text
- `--no-timestamp`: Don't show timestamp of smoking detection

## Advanced Options

### Smoking Detection Options

The `process_smoking_detection.py` script supports several options for optimizing performance:

- `--batch-size`: Number of images to process in each batch (default: 4)
- `--max-image-size`: Maximum image dimension to resize to before processing (default: 1024)
- `--device`: Device to run the model on (cuda, cpu)
- `--aggressive-gc`: Aggressively collect garbage to reduce memory usage
- `--debug-timing`: Print detailed timing information

### Video Splitting Options

The `split.py` script supports several options for customizing frame extraction:

- `--num-frames`: Number of frames to extract per second
- `--resize-frames`: Resize frames to specified dimensions
- `--max-frames`: Maximum number of frames to process

## License

[MIT License](LICENSE) 