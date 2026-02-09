##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'app.py'
#
# Description:
# Flask web application for the Vision-Based Packing
# Project. Provides upload interface, processing status,
# and results visualization.
#
##############################################

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import threading
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path('data/frames').mkdir(parents=True, exist_ok=True)
Path('data/yolo_frames').mkdir(parents=True, exist_ok=True)
Path('data/videos').mkdir(parents=True, exist_ok=True)

# Processing status storage
processing_status = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_async(video_path, job_id):
    """Process video in background thread"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = 'Extracting frames...'
        
        # Import main processing function
        import sys
        sys.path.insert(0, 'src')
        from main import main as process_pipeline
        
        # Run processing
        process_pipeline(video_path)
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['message'] = 'Processing complete!'
        processing_status[job_id]['completed_at'] = datetime.now().isoformat()
        
        # Load results
        try:
            with open('refined_item_list.txt', 'r', encoding='utf-8') as f:
                items = [line.strip() for line in f if line.strip()]
            processing_status[job_id]['items'] = items
            processing_status[job_id]['item_count'] = len(items)
        except:
            processing_status[job_id]['items'] = []
        
        # Check for output files - FIXED PATHS
        processing_status[job_id]['files'] = {
            'detection_log': os.path.exists('detection_log.csv'),
            'item_list': os.path.exists('refined_item_list.txt'),
            'annotated_video': os.path.exists('data/videos/output_annotated.mp4'),  # FIXED
            'entry_log': os.path.exists('entry_log.json'),
            'box_mappings': os.path.exists('box_mappings.json')
        }
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Error: {str(e)}'
        processing_status[job_id]['error'] = str(e)


@app.route('/')
def index():
    """Main upload page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # Create job ID
    job_id = timestamp
    
    # Initialize processing status
    processing_status[job_id] = {
        'job_id': job_id,
        'filename': filename,
        'status': 'queued',
        'message': 'Video uploaded, waiting to start...',
        'uploaded_at': datetime.now().isoformat(),
        'filepath': filepath
    }
    
    # Start processing in background
    thread = threading.Thread(
        target=process_video_async,
        args=(filepath, job_id)
    )
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': 'Upload successful, processing started'
    })


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_status[job_id])


@app.route('/results/<job_id>')
def get_results(job_id):
    """Get detailed results"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = processing_status[job_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not complete'}), 400
    
    results = {
        'job_id': job_id,
        'items': status.get('items', []),
        'item_count': status.get('item_count', 0),
        'files': status.get('files', {})
    }
    
    # Load detection log if available
    if os.path.exists('detection_log.csv'):
        import csv
        detections = []
        with open('detection_log.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                detections.append(row)
        results['detections'] = detections[:100]  # Limit to first 100
    
    # Load entry log if available
    if os.path.exists('entry_log.json'):
        try:
            with open('entry_log.json', 'r', encoding='utf-8') as f:
                results['entry_log'] = json.load(f)
        except:
            pass
    
    # Load box mappings if available
    if os.path.exists('box_mappings.json'):
        try:
            with open('box_mappings.json', 'r', encoding='utf-8') as f:
                results['box_mappings'] = json.load(f)
        except:
            pass
    
    return jsonify(results)


@app.route('/download/<file_type>')
def download_file(file_type):
    """Download result files"""
    # FIXED: Updated video path
    files = {
        'items': 'refined_item_list.txt',
        'log': 'detection_log.csv',
        'video': 'data/videos/output_annotated.mp4',  # FIXED PATH
        'entry_log': 'entry_log.json',
        'box_mappings': 'box_mappings.json'
    }
    
    if file_type not in files:
        return jsonify({'error': 'Invalid file type'}), 400
    
    filepath = files[file_type]
    
    if not os.path.exists(filepath):
        return jsonify({'error': f'File not found: {filepath}'}), 404
    
    return send_file(filepath, as_attachment=True)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("[INFO] Starting Vision-Based Packing Web Interface")
    print("[INFO] Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)