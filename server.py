from flask import Flask, request, jsonify, send_file, render_template, url_for
from flask_cors import CORS
import os
import uuid
import json
import time
import threading
from werkzeug.utils import secure_filename
import shutil
from werkzeug.middleware.proxy_fix import ProxyFix
from utils import (
    split_video_and_create_collages,
    load_model,
    process_image,
    process_video_with_disclaimer,
)
from dotenv import load_dotenv

load_dotenv()

# Get absolute paths for static and templates folders
current_dir = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(current_dir, "static")
templates_folder = os.path.join(current_dir, "templates")

# URL prefix configuration (e.g., "/proxy/5000")
URL_PREFIX = os.environ.get("URL_PREFIX", "")

# Create Flask app with explicit static and template folders
app = Flask(__name__, static_folder=static_folder, template_folder=templates_folder)

# Apply URL prefix if set
if URL_PREFIX:
    class PrefixMiddleware:
        def __init__(self, app, prefix):
            self.app = app
            self.prefix = prefix

        def __call__(self, environ, start_response):
            if environ['PATH_INFO'].startswith(self.prefix):
                environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):]
                environ['SCRIPT_NAME'] = self.prefix
                return self.app(environ, start_response)
            else:
                start_response('404', [('Content-Type', 'text/plain')])
                return [b'Not Found']

    app.wsgi_app = PrefixMiddleware(app.wsgi_app, URL_PREFIX)

# Fix for running behind a proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Enable CORS for all routes with support for credentials
CORS(app, supports_credentials=True)

# Set maximum content length to 1000MB (1GB) to handle large video files
app.config["MAX_CONTENT_LENGTH"] = 1000 * 1024 * 1024  # 1GB in bytes

# Override url_for to include the URL prefix for static files
@app.context_processor
def override_url_for():
    def _url_for(endpoint, **kwargs):
        if endpoint == 'static':
            # If URL_PREFIX is set and not already in the url
            if URL_PREFIX and not request.path.startswith(URL_PREFIX):
                kwargs['_external'] = False
                return URL_PREFIX + url_for(endpoint, **kwargs)
        return url_for(endpoint, **kwargs)
    return dict(url_for=_url_for)

# Print debug information about static folder and configuration
print(f"Static folder path: {app.static_folder}")
print(f"Static folder exists: {os.path.exists(app.static_folder)}")
print(f"URL_PREFIX: {URL_PREFIX}")
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
    return render_template("index.html", url_prefix=URL_PREFIX)


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
            num_frames=task["frames_per_second"],
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
                frames_data=frames_data,
                detect_only=True,
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
    app.run(debug=True, host="0.0.0.0", port=80)
