from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    render_template,
    send_from_directory,
)
from flask_cors import CORS
import os
import uuid
import json
import time
import threading
from werkzeug.utils import secure_filename
import shutil
from utils import (
    split_video_and_create_collages,
    load_model,
    process_image,
    process_video_with_disclaimer,
)

# Get absolute paths for static and templates folders
current_dir = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(current_dir, "static")
templates_folder = os.path.join(current_dir, "templates")

# Create Flask app with explicit static and template folders
app = Flask(__name__, static_folder=static_folder, template_folder=templates_folder)
CORS(app)  # Enable CORS for all routes

# Set maximum content length to 200MB (adjust as needed)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB in bytes

# Print debug information about static folder
print(f"Static folder path: {app.static_folder}")
print(f"Static folder exists: {os.path.exists(app.static_folder)}")
if os.path.exists(app.static_folder):
    print(f"Static folder contents: {os.listdir(app.static_folder)}")

# Configuration
UPLOAD_FOLDER = "uploads"
DOWNLOAD_FOLDER = "downloads"
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Store processing status
processing_tasks = {}


def allowed_video_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    )


def allowed_image_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_video():
    # Check if video file is present
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No video file selected"}), 400

    if not allowed_video_file(video_file.filename):
        return jsonify({"error": "Invalid video file format"}), 400

    # Check frames per second parameter
    frames_per_second = request.form.get("frames_per_second", "1")
    try:
        frames_per_second = int(frames_per_second)
        if frames_per_second < 1 or frames_per_second > 24:
            return jsonify({"error": "Frames per second must be between 1 and 24"}), 400
    except ValueError:
        return jsonify({"error": "Invalid frames per second value"}), 400

    # Handle disclaimer image if present
    disclaimer_image_path = None
    if "disclaimer_image" in request.files:
        disclaimer_image = request.files["disclaimer_image"]
        if disclaimer_image.filename != "" and allowed_image_file(
            disclaimer_image.filename
        ):
            disclaimer_filename = secure_filename(disclaimer_image.filename)
            disclaimer_image_path = os.path.join(
                UPLOAD_FOLDER, f"{uuid.uuid4()}_{disclaimer_filename}"
            )
            disclaimer_image.save(disclaimer_image_path)

    # Save video file
    video_filename = secure_filename(video_file.filename)
    task_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{video_filename}")
    video_file.save(video_path)

    # Create task directory in downloads folder
    task_dir = os.path.join(DOWNLOAD_FOLDER, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # Initialize task status
    processing_tasks[task_id] = {
        "status": "initializing",
        "progress": 0,
        "message": "Initializing processing...",
        "video_path": video_path,
        "disclaimer_path": disclaimer_image_path,
        "frames_per_second": frames_per_second,
        "output_path": os.path.join(task_dir, f"processed_{video_filename}"),
        "start_time": time.time(),
        "frames_processed": 0,
        "total_frames": 0,
        "smoking_frames": 0,
    }

    # Start processing in a background thread
    threading.Thread(target=process_video_task, args=(task_id,)).start()

    return jsonify(
        {"task_id": task_id, "status": "started", "message": "Video processing started"}
    )


def process_video_task(task_id):
    """Process the video in the background"""
    task = processing_tasks[task_id]

    try:
        # Load the smoking detection model
        model = load_model()

        # Update status to extracting frames
        task["status"] = "extracting_frames"
        task["progress"] = 5
        task["message"] = "Extracting frames from video..."

        # Create frames directory
        frames_dir = os.path.join(DOWNLOAD_FOLDER, task_id, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Extract frames using split_video_and_create_collages function
        # Note: split_video_and_create_collages(video_path, output_dir, fps=1, collage_grid=(4, 4), resize_frames=None)
        frame_info = split_video_and_create_collages(
            video_path=task["video_path"],
            output_dir=frames_dir,
            fps=task["frames_per_second"],
            collage_grid=None,  # None for individual frames
            resize_frames=None,  # Let's use original frame sizes for better detection
        )

        # Check if frames were extracted successfully
        frames_info_path = os.path.join(frames_dir, "frames_info.json")
        if not os.path.exists(frames_info_path):
            raise Exception("Failed to extract frames from video")

        # Load frames info
        with open(frames_info_path, "r") as f:
            frames_data = json.load(f)

        task["total_frames"] = len(frames_data)
        task["progress"] = 30
        task["message"] = f'Extracted {task["total_frames"]} frames'

        # Update status to detecting smoking
        task["status"] = "detecting_smoking"
        task["message"] = "Detecting smoking in frames..."

        # Process each frame for smoking detection
        smoking_frames = []
        for i, frame_info in enumerate(frames_data):
            frame_path = os.path.join(frames_dir, frame_info["filename"])

            # Note: process_image(model, image_path, confidence_threshold=0.5)
            is_smoking = process_image(
                model=model,
                image_path=frame_path,
                confidence_threshold=0.5,  # Default threshold
            )

            if is_smoking:
                task["smoking_frames"] += 1
                smoking_frames.append(frame_info["timestamp"])

            task["frames_processed"] = i + 1
            task["progress"] = 30 + (i / task["total_frames"]) * 40

            # Calculate processing speed and ETA
            elapsed_time = time.time() - task["start_time"]
            if i > 0:
                fps = task["frames_processed"] / elapsed_time
                frames_remaining = task["total_frames"] - task["frames_processed"]
                time_remaining = frames_remaining / fps if fps > 0 else 0

                task["message"] = (
                    f"Analyzing: {task['frames_processed']}/{task['total_frames']} frames | "
                    f"Smoking: {task['smoking_frames']} frames ({task['smoking_frames']/task['frames_processed']*100:.1f}%) | "
                    f"Speed: {fps:.1f} FPS | "
                    f"ETA: {int(time_remaining//60):02d}:{int(time_remaining%60):02d}"
                )

        # Update status to adding disclaimers
        task["status"] = "adding_disclaimers"
        task["progress"] = 70
        task["message"] = "Adding disclaimers to video..."

        # Note: process_video_with_disclaimer(input_video, output_video, smoking_timestamps, disclaimer_image=None, disclaimer_text=None)
        process_video_with_disclaimer(
            input_video=task["video_path"],
            output_video=task["output_path"],
            smoking_timestamps=smoking_frames,
            disclaimer_image=(
                task["disclaimer_path"] if task["disclaimer_path"] else None
            ),
            disclaimer_text=(
                "Warning: This video contains scenes of smoking"
                if not task["disclaimer_path"]
                else None
            ),
        )

        # Update status to completed
        task["status"] = "completed"
        task["progress"] = 100
        task["message"] = "Processing completed successfully"

        # Calculate file size
        file_size_mb = os.path.getsize(task["output_path"]) / (1024 * 1024)
        task["file_size"] = f"{file_size_mb:.2f} MB"

    except Exception as e:
        # Update status to error
        task["status"] = "error"
        task["progress"] = 0
        task["message"] = f"Error: {str(e)}"
        print(f"Error processing video: {str(e)}")

        # Clean up any temporary files
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)


@app.route("/api/status/<task_id>", methods=["GET"])
def get_status(task_id):
    """Get the status of a processing task"""
    if task_id not in processing_tasks:
        return jsonify({"error": "Task not found"}), 404

    task = processing_tasks[task_id]

    response = {
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
    }

    if task["status"] == "completed":
        response["file_size"] = task.get("file_size", "0 MB")
        response["total_frames"] = task["total_frames"]
        response["smoking_frames"] = task["smoking_frames"]
        response["smoking_percentage"] = (
            f"{task['smoking_frames']/task['total_frames']*100:.1f}"
            if task["total_frames"] > 0
            else "0.0"
        )
        response["frames_per_second"] = task["frames_per_second"]

    return jsonify(response)


@app.route("/api/download/<task_id>", methods=["GET"])
def download_video(task_id):
    """Download the processed video"""
    if task_id not in processing_tasks:
        return jsonify({"error": "Task not found"}), 404

    task = processing_tasks[task_id]

    if task["status"] != "completed":
        return jsonify({"error": "Processing not completed yet"}), 400

    if not os.path.exists(task["output_path"]):
        return jsonify({"error": "Output file not found"}), 404

    return send_file(
        task["output_path"],
        as_attachment=True,
        download_name=os.path.basename(task["output_path"]),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
