# Auto Disclaimer Adder

This project provides tools to automatically detect smoking-related content in video frames and add appropriate disclaimers.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have a valid Hugging Face API token for accessing the models.

## Scripts

### 1. Update Smoking Status

This script processes collage images and updates the JSON file with smoking detection results:

```
python3 update_smoking_status.py
```

### 2. Annotate Smoking Images

This script annotates images where smoking is detected and saves them in a new directory:

```
python3 annotate_smoking_images.py
```

### 3. All-in-One Processing

This script can both detect smoking and annotate images in one go:

```
# Run full detection and annotation
python3 process_smoking_detection.py

# Only detect smoking, don't annotate images
python3 process_smoking_detection.py --detect-only

# Only annotate images based on existing JSON data
python3 process_smoking_detection.py --annotate-only
```

## How It Works

1. The system uses the Moondream model to detect smoking-related content in collage images.
2. It updates a JSON file with smoking detection results for each frame.
3. It can annotate images by drawing red rectangles around frames where smoking is detected.

## Output

- Updated JSON file with smoking detection results: `output_collages/frames_info.json`
- Annotated images: `annotated_collages/` 