from flask import Flask, request, jsonify, send_file, render_template, Blueprint
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

# URL prefix configuration (e.g., /proxy/5000)
URL_PREFIX = os.getenv("URL_PREFIX", "").rstrip("/")  # Remove trailing slash if present

# Create Flask app with explicit static and template folders
app = Flask(__name__, 
           static_folder=static_folder, 
           template_folder=templates_folder,
           static_url_path=f'{URL_PREFIX}/static' if URL_PREFIX else '/static')

# Create Blueprint for API routes
api = Blueprint('api', __name__, url_prefix=f'{URL_PREFIX}/api' if URL_PREFIX else '/api')

# Fix for running behind a proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Enable CORS for all routes with support for credentials
CORS(app, supports_credentials=True)

# Set maximum content length to 1000MB (1GB)
app.config["MAX_CONTENT_LENGTH"] = 1000 * 1024 * 1024  # 1GB in bytes

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

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "File too large",
        "message": "The file you're trying to upload is too large. Maximum allowed size is 1GB."
    }), 413

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

# Move API routes to Blueprint
@api.route("/process", methods=["POST"])
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

@api.route("/status/<task_id>", methods=["GET"])
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

@api.route("/download/<task_id>", methods=["GET"])
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

# Register the API blueprint
app.register_blueprint(api)

if __name__ == "__main__":
    print(f"Server running with URL prefix: {URL_PREFIX}")
    print(f"API endpoints will be available at: {URL_PREFIX}/api/*")
    print(f"Static files will be served from: {URL_PREFIX}/static/*")
    app.run(debug=True, host="0.0.0.0", port=80)
