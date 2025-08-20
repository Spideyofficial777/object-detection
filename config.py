import os
from datetime import timedelta

class Config:
    # Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    
    # File upload settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB limit
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.mp4', '.mov', '.avi'}
    
    # Directory settings
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'static/results'
    SAMPLES_FOLDER = 'samples'
    
    # YOLO model settings
    YOLO_MODEL = os.environ.get('YOLO_MODEL', 'yolov8n.pt')
    
    # Video processing limits
    MAX_VIDEO_DURATION = int(os.environ.get('MAX_VIDEO_DURATION', '30'))  # seconds
    
    # File cleanup settings
    CLEANUP_INTERVAL_HOURS = 24
    CLEANUP_AGE_HOURS = 24
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create directories if they don't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
        os.makedirs(Config.SAMPLES_FOLDER, exist_ok=True)