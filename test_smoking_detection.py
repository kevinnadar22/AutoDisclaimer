import os
import json
import time
from PIL import Image
from process_smoking_detection import load_model, process_image


def test_smoking_detection(input_dir="output_collages", output_dir="test_results"):
    """Test smoking detection model on a set of images"""
    os.makedirs(output_dir, exist_ok=True)

    # Load frames info
    with open(os.path.join(input_dir, "frames_info.json"), "r") as f:
        frames_data = json.load(f)

    # Get unique image paths
    image_paths = list(set(frame["path"] for frame in frames_data))

    # Load model
    model = load_model()

    # Process each image
    results = []
    for image_name in image_paths:
        image_path = os.path.join(input_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Time the processing
        start_time = time.time()
        frames_data, _ = process_image(model, image_path, frames_data, detect_only=True)
        process_time = time.time() - start_time

        # Check if smoking was detected
        image_frames = [f for f in frames_data if f["path"] == image_name]
        smoking_detected = any(f["smoking"] for f in image_frames)

        results.append(
            {
                "image": image_name,
                "time": process_time,
                "smoking_detected": smoking_detected,
            }
        )

    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}/results.json")
    return results


def main():
    test_smoking_detection()


if __name__ == "__main__":
    main()
