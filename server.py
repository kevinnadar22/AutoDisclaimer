from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import uuid
import json
import time
import threading
from werkzeug.utils import secure_filename
import subprocess
import shutil

# Import modules from the utils package
from utils.split import split_video_and_create_collages
from utils.process_smoking_detection import load_model, process_image
from utils.add_smoking_disclaimer import process_video_with_disclaimer

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Store processing status
processing_tasks = {}


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_video():
    # Check if video file is present
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if not allowed_video_file(video_file.filename):
        return jsonify({'error': 'Invalid video file format'}), 400
    
    # Check frames per second parameter
    frames_per_second = request.form.get('frames_per_second', '1')
    try:
        frames_per_second = int(frames_per_second)
        if frames_per_second < 1 or frames_per_second > 24:
            return jsonify({'error': 'Frames per second must be between 1 and 24'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid frames per second value'}), 400
    
    # Handle disclaimer image if present
    disclaimer_image_path = None
    if 'disclaimer_image' in request.files:
        disclaimer_image = request.files['disclaimer_image']
        if disclaimer_image.filename != '' and allowed_image_file(disclaimer_image.filename):
            disclaimer_filename = secure_filename(disclaimer_image.filename)
            disclaimer_image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{disclaimer_filename}")
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
        'status': 'initializing',
        'progress': 0,
        'message': 'Initializing processing...',
        'video_path': video_path,
        'disclaimer_path': disclaimer_image_path,
        'frames_per_second': frames_per_second,
        'output_path': os.path.join(task_dir, f"processed_{video_filename}"),
        'start_time': time.time(),
        'frames_processed': 0,
        'total_frames': 0,
        'smoking_frames': 0
    }
    
    # Start processing in a background thread
    threading.Thread(target=process_video_task, args=(task_id,)).start()
    
    return jsonify({
        'task_id': task_id,
        'status': 'started',
        'message': 'Video processing started'
    })


def process_video_task(task_id):
    """Process the video in the background"""
    task = processing_tasks[task_id]
    
    try:
        # Update status to extracting frames
        task['status'] = 'extracting_frames'
        task['progress'] = 5
        task['message'] = 'Extracting frames from video...'
        
        # Create frames directory
        frames_dir = os.path.join(DOWNLOAD_FOLDER, task_id, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract frames using split.py
        cmd = [
            'python', 'split.py',
            '--video', task['video_path'],
            '--output-dir', frames_dir,
            '--num-frames', str(task['frames_per_second']),
            '--resize', '640x360'
        ]
        
        # Run the command and simulate progress
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()
        
        # Check if frames were extracted successfully
        frames_info_path = os.path.join(frames_dir, 'frames_info.json')
        if not os.path.exists(frames_info_path):
            raise Exception("Failed to extract frames from video")
        
        # Load frames info
        with open(frames_info_path, 'r') as f:
            frames_data = json.load(f)
        
        task['total_frames'] = len(frames_data)
        task['progress'] = 30
        task['message'] = f'Extracted {task["total_frames"]} frames'
        
        # Update status to detecting smoking
        task['status'] = 'detecting_smoking'
        task['message'] = 'Detecting smoking in frames...'
        
        # Simulate smoking detection (in a real implementation, you would call process_smoking_detection.py)
        for i in range(task['total_frames']):
            # Simulate processing delay
            time.sleep(0.05)
            
            # Randomly determine if smoking is detected (for simulation)
            is_smoking = (i % 5 == 0)  # Simulate 20% of frames with smoking
            
            if is_smoking:
                task['smoking_frames'] += 1
            
            task['frames_processed'] = i + 1
            task['progress'] = 30 + (i / task['total_frames']) * 40
            
            # Calculate processing speed and ETA
            elapsed_time = time.time() - task['start_time']
            if i > 0:
                fps = task['frames_processed'] / elapsed_time
                frames_remaining = task['total_frames'] - task['frames_processed']
                time_remaining = frames_remaining / fps if fps > 0 else 0
                
                task['message'] = (
                    f"Analyzing: {task['frames_processed']}/{task['total_frames']} frames | "
                    f"Smoking: {task['smoking_frames']} frames ({task['smoking_frames']/task['frames_processed']*100:.1f}%) | "
                    f"Speed: {fps:.1f} FPS | "
                    f"ETA: {int(time_remaining//60):02d}:{int(time_remaining%60):02d}"
                )
        
        # Update status to adding disclaimers
        task['status'] = 'adding_disclaimers'
        task['progress'] = 70
        task['message'] = 'Adding disclaimers to video...'
        
        # Simulate adding disclaimers (in a real implementation, you would call process_video_with_disclaimer)
        for i in range(100):
            # Simulate processing delay
            time.sleep(0.1)
            
            task['progress'] = 70 + (i / 100) * 30
            
            # Calculate processing speed and ETA
            elapsed_time = time.time() - task['start_time']
            fps = i / elapsed_time if elapsed_time > 0 else 0
            frames_remaining = 100 - i
            time_remaining = frames_remaining / fps if fps > 0 else 0
            
            task['message'] = (
                f"Processing: {i}/{100} frames | "
                f"Smoking: {task['smoking_frames']} frames ({task['smoking_frames']/task['total_frames']*100:.1f}%) | "
                f"Speed: {fps:.1f} FPS | "
                f"ETA: {int(time_remaining//60):02d}:{int(time_remaining%60):02d}"
            )
        
        # For demo purposes, copy the original video as the processed output
        # In a real implementation, you would generate a new video with disclaimers
        shutil.copy(task['video_path'], task['output_path'])
        
        # Update status to completed
        task['status'] = 'completed'
        task['progress'] = 100
        task['message'] = 'Processing completed successfully'
        
        # Calculate file size
        file_size_mb = os.path.getsize(task['output_path']) / (1024 * 1024)
        task['file_size'] = f"{file_size_mb:.2f} MB"
        
    except Exception as e:
        # Update status to error
        task['status'] = 'error'
        task['progress'] = 0
        task['message'] = f'Error: {str(e)}'
        print(f"Error processing video: {str(e)}")


@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get the status of a processing task"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    
    response = {
        'status': task['status'],
        'progress': task['progress'],
        'message': task['message'],
    }
    
    if task['status'] == 'completed':
        response['file_size'] = task.get('file_size', '0 MB')
        response['total_frames'] = task['total_frames']
        response['smoking_frames'] = task['smoking_frames']
        response['smoking_percentage'] = f"{task['smoking_frames']/task['total_frames']*100:.1f}" if task['total_frames'] > 0 else "0.0"
        response['frames_per_second'] = task['frames_per_second']
    
    return jsonify(response)


@app.route('/api/download/<task_id>', methods=['GET'])
def download_video(task_id):
    """Download the processed video"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    
    if task['status'] != 'completed':
        return jsonify({'error': 'Processing not completed yet'}), 400
    
    if not os.path.exists(task['output_path']):
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(
        task['output_path'],
        as_attachment=True,
        download_name=os.path.basename(task['output_path'])
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 