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
# FIXED: Added encoding='utf-8' to all file reads/writes
#        to prevent Windows charmap errors.
# UPDATED: Added human-in-the-loop exit confirmation endpoints:
#          GET  /exit_confirmations   — returns pending exit events for UI
#          POST /exit_confirm/<id>    — user submits Yes/No answer
# FIXED: Item list parser now skips box header lines ([BOX-001] etc.)
#        so the item count and list only reflect actual detected items.
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
Path('data/exit_crops').mkdir(parents=True, exist_ok=True)

# Processing status storage
processing_status = {}

# Shared exit confirmation queue (populated by ExitDetector, read by routes below)
# Imported lazily inside routes to avoid circular import at startup.
def _get_confirmation_queue():
    try:
        import sys
        sys.path.insert(0, 'src')
        from detection.exit_detector import confirmation_queue
        return confirmation_queue
    except Exception:
        return {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_cleanup():
    """
    Run cleanup.py to delete extracted frames before starting a new job.
    Keeps .gitkeep placeholder files intact.
    """
    try:
        import sys
        import importlib.util
        cleanup_path = Path(__file__).parent / 'cleanup.py'
        spec = importlib.util.spec_from_file_location("cleanup", cleanup_path)
        cleanup_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cleanup_mod)

        cleanup_mod.cleanup_frames("data/frames")
        cleanup_mod.cleanup_frames("data/yolo_frames")
        print("[INFO] Cleanup complete before new job.")
    except Exception as e:
        print(f"[WARNING] Cleanup failed (non-fatal): {e}")


def _parse_item_list(filepath: str) -> list:
    """
    Parse refined_item_list.txt into a flat list of item name strings.

    The file format is:
        [BOX-001]
          item 1: "Apple"
          item 2: "Bottle"

    We skip box header lines (starting with '[') and blank lines,
    and return only the item label strings so the count and display
    are accurate.
    """
    items = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                # Skip box header lines like [BOX-001]
                if stripped.startswith('[') and stripped.endswith(']'):
                    continue
                items.append(stripped)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[WARNING] Could not parse item list: {e}")
    return items


def process_video_async(video_path, job_id):
    """Process video in background thread"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = 'Cleaning up previous run...'

        run_cleanup()

        processing_status[job_id]['message'] = 'Extracting frames...'

        import sys
        sys.path.insert(0, 'src')

        # ---- Windows UTF-8 fix (env vars only, no stdout wrapping in threads) ----
        import os
        os.environ['PYTHONUTF8'] = '1'
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        from main import main as process_pipeline
        process_pipeline(video_path)

        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['message'] = 'Processing complete!'
        processing_status[job_id]['completed_at'] = datetime.now().isoformat()

        # Parse item list — skips box headers so count is accurate
        items = _parse_item_list('refined_item_list.txt')
        processing_status[job_id]['items']      = items
        processing_status[job_id]['item_count'] = len(items)

        # Check for output files
        processing_status[job_id]['files'] = {
            'detection_log':   os.path.exists('detection_log.csv'),
            'item_list':       os.path.exists('refined_item_list.txt'),
            'annotated_video': os.path.exists('data/videos/output_annotated.mp4'),
            'entry_log':       os.path.exists('entry_log.json'),
            'box_mappings':    os.path.exists('box_mappings.json'),
        }

    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Error: {str(e)}'
        processing_status[job_id]['error'] = str(e)
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

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
        'job_id':      job_id,
        'filename':    filename,
        'status':      'queued',
        'message':     'Video uploaded, waiting to start...',
        'uploaded_at': datetime.now().isoformat(),
        'filepath':    filepath,
    }

    # Start processing in background
    thread = threading.Thread(
        target=process_video_async,
        args=(filepath, job_id)
    )
    thread.start()

    return jsonify({
        'job_id': job_id,
        'message': 'Upload successful, processing started',
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
        'job_id':     job_id,
        'items':      status.get('items', []),
        'item_count': status.get('item_count', 0),
        'files':      status.get('files', {}),
    }

    # ---- FIX: encoding='utf-8' on all file reads ----
    if os.path.exists('detection_log.csv'):
        import csv
        detections = []
        try:
            with open('detection_log.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    detections.append(row)
            results['detections'] = detections[:100]
        except Exception as e:
            print(f"[WARNING] Could not read detection log: {e}")

    if os.path.exists('entry_log.json'):
        try:
            with open('entry_log.json', 'r', encoding='utf-8') as f:
                results['entry_log'] = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read entry log: {e}")

    if os.path.exists('box_mappings.json'):
        try:
            with open('box_mappings.json', 'r', encoding='utf-8') as f:
                results['box_mappings'] = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read box mappings: {e}")

    return jsonify(results)


@app.route('/download/<file_type>')
def download_file(file_type):
    """Download result files"""
    files = {
        'items':        'refined_item_list.txt',
        'log':          'detection_log.csv',
        'video':        'data/videos/output_annotated.mp4',
        'entry_log':    'entry_log.json',
        'box_mappings': 'box_mappings.json',
    }

    if file_type not in files:
        return jsonify({'error': 'Invalid file type'}), 400

    filepath = files[file_type]

    if not os.path.exists(filepath):
        return jsonify({'error': f'File not found: {filepath}'}), 404

    return send_file(filepath, as_attachment=True)


# ──────────────────────────────────────────────────────────────────────────────
# Exit confirmation routes (human-in-the-loop)
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/exit_confirmations')
def get_exit_confirmations():
    """
    Returns all pending (unanswered) exit confirmation requests.
    The frontend polls this every 3 seconds while a job is processing.

    Response shape:
    {
      "pending": [
        {
          "confirmation_id": "exit_3_42",
          "label":           "Mug",
          "box_id":          "BOX-001",
          "frame":           42,
          "timestamp":       19.5,
          "image_b64":       "<base64 JPEG of box crop>"
        },
        ...
      ]
    }
    """
    confirmation_queue = _get_confirmation_queue()
    pending = [
        {
            "confirmation_id": cid,
            "label":           entry["label"],
            "box_id":          entry["box_id"],
            "frame":           entry["frame"],
            "timestamp":       round(entry["timestamp"], 2),
            "image_b64":       entry["image_b64"],
        }
        for cid, entry in confirmation_queue.items()
        if entry["answer"] is None
    ]
    return jsonify({"pending": pending})


@app.route('/exit_confirm/<confirmation_id>', methods=['POST'])
def submit_exit_confirmation(confirmation_id):
    """
    User submits their answer for an exit confirmation.

    Request body (JSON):
      { "confirmed": true }   → item was removed
      { "confirmed": false }  → item is still present (false alarm)

    Response:
      { "ok": true, "confirmation_id": "exit_3_42" }
    """
    confirmation_queue = _get_confirmation_queue()

    if confirmation_id not in confirmation_queue:
        return jsonify({"error": "Unknown confirmation ID"}), 404

    data      = request.get_json(silent=True) or {}
    confirmed = bool(data.get("confirmed", False))

    confirmation_queue[confirmation_id]["answer"] = confirmed

    label = confirmation_queue[confirmation_id]["label"]
    box   = confirmation_queue[confirmation_id]["box_id"]
    print(
        f"[HUMAN] Exit {'CONFIRMED' if confirmed else 'REJECTED'}: "
        f"'{label}' from {box} (id={confirmation_id})"
    )

    return jsonify({"ok": True, "confirmation_id": confirmation_id})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status':    'healthy',
        'timestamp': datetime.now().isoformat(),
    })


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    os.environ['PYTHONUTF8']       = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    print("[INFO] Starting Vision-Based Packing Web Interface")
    print("[INFO] Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)