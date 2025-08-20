import os
import logging
import time
import uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, Response
from werkzeug.utils import secure_filename
import cv2
import base64
import numpy as np
from threading import Lock
import json

from config import Config
from utils.detection import run_detection, get_detector
from utils.cleanup import cleanup_old_files, schedule_cleanup

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# Global variables for live detection
camera_lock = Lock()
live_detector = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in Config.ALLOWED_EXTENSIONS

def generate_filename(original_filename):
    """Generate unique filename"""
    ext = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_{unique_id}{ext}"

def get_recent_results(limit=5):
    """Get recent detection results"""
    results_dir = Path(Config.RESULTS_FOLDER)
    if not results_dir.exists():
        return []
    
    files = list(results_dir.glob('detected_*'))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    recent = []
    for file_path in files[:limit]:
        recent.append({
            'filename': file_path.name,
            'timestamp': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'size_kb': round(file_path.stat().st_size / 1024, 1)
        })
    
    return recent

@app.route('/')
def index():
    """Main upload page with recent results"""
    recent_results = get_recent_results()
    return render_template('index.html', recent_results=recent_results)

@app.route('/live')
def live_detection_page():
    """Live camera detection page"""
    return render_template('live.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle file upload and run detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported: {", ".join(Config.ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save uploaded file
        filename = generate_filename(file.filename)
        input_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(input_path)
        
        logger.info(f"File uploaded: {filename}")
        
        # Run detection
        start_time = time.time()
        result = run_detection(input_path)
        
        # Log results
        detection_time = time.time() - start_time
        logger.info(f"Detection completed in {detection_time:.2f}s, found {len(result['detections'])} objects")
        
        # Clean up input file
        os.remove(input_path)
        
        # Return result filename for redirect
        result_filename = Path(result['output_path']).name
        
        if request.headers.get('Content-Type') == 'application/json':
            return jsonify({
                'success': True,
                'result_url': url_for('show_result', filename=result_filename),
                'detections': result['detections'],
                'runtime_ms': result['runtime_ms']
            })
        else:
            return redirect(url_for('show_result', filename=result_filename))
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return jsonify({'error': 'Detection failed. Please try again.'}), 500

@app.route('/live_detect', methods=['POST'])
def live_detect():
    """Handle live camera frame detection"""
    global live_detector
    
    try:
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Get detector (create if needed)
        if live_detector is None:
            live_detector = get_detector()
        
        # Run detection on frame
        with camera_lock:
            results = live_detector.model.predict(frame, save=False, verbose=False)
            result = results[0]
        
        # Extract detections
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                detection = {
                    'class': result.names[int(box.cls.item())],
                    'confidence': round(box.conf.item(), 3),
                    'bbox': [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'frame_width': frame.shape[1],
            'frame_height': frame.shape[0]
        })
        
    except Exception as e:
        logger.error(f"Live detection failed: {e}")
        return jsonify({'error': 'Live detection failed'}), 500

@app.route('/result/<filename>')
def show_result(filename):
    """Display detection results"""
    format_type = request.args.get('format', 'html')
    
    result_path = os.path.join(Config.RESULTS_FOLDER, filename)
    
    if not os.path.exists(result_path):
        if format_type == 'json':
            return jsonify({'error': 'Result not found'}), 404
        return render_template('error.html', message='Result not found'), 404
    
    # Get file info
    file_stat = os.stat(result_path)
    file_info = {
        'filename': filename,
        'size_kb': round(file_stat.st_size / 1024, 1),
        'created': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Try to load detection metadata (if exists)
    metadata_path = result_path.replace('.', '_metadata.') + '.json'
    detections = []
    runtime_ms = 0
    model_name = Config.YOLO_MODEL
    
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                detections = metadata.get('detections', [])
                runtime_ms = metadata.get('runtime_ms', 0)
                model_name = metadata.get('model_name', Config.YOLO_MODEL)
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
    
    if format_type == 'json':
        return jsonify({
            'filename': filename,
            'detections': detections,
            'runtime_ms': runtime_ms,
            'model_name': model_name,
            'file_info': file_info
        })
    
    # Determine if it's a video file
    is_video = Path(filename).suffix.lower() in ['.mp4', '.mov', '.avi']
    
    return render_template('result.html', 
                         filename=filename, 
                         detections=detections,
                         runtime_ms=runtime_ms,
                         model_name=model_name,
                         file_info=file_info,
                         is_video=is_video)

@app.route('/download/<filename>')
def download_result(filename):
    """Download result file"""
    result_path = os.path.join(Config.RESULTS_FOLDER, filename)
    
    if not os.path.exists(result_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(result_path, as_attachment=True)

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    detector = get_detector()
    
    return jsonify({
        'status': 'ready',
        'model': detector.model_name,
        'upload_limit_mb': Config.MAX_CONTENT_LENGTH // (1024 * 1024),
        'allowed_extensions': list(Config.ALLOWED_EXTENSIONS),
        'max_video_duration': Config.MAX_VIDEO_DURATION
    })

@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize detector and cleanup on startup
    logger.info("Starting Flask Object Detection App")
    
    try:
        # Warm up detector
        detector = get_detector()
        logger.info(f"YOLO model '{detector.model_name}' loaded successfully")
        
        # Clean up old files on startup
        upload_deleted = cleanup_old_files(Config.UPLOAD_FOLDER, Config.CLEANUP_AGE_HOURS)
        results_deleted = cleanup_old_files(Config.RESULTS_FOLDER, Config.CLEANUP_AGE_HOURS)
        
        if upload_deleted or results_deleted:
            logger.info(f"Startup cleanup: removed {upload_deleted} upload files, {results_deleted} result files")
        
        # Schedule regular cleanup
        schedule_cleanup(app)
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)