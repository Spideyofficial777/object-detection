# 🎯 YOLO Object Detection Flask App

A production-ready Flask web application for object detection using YOLO (You Only Look Once) models. Features both file upload detection and real-time camera detection capabilities.

## 🌟 Features

### 📁 File Upload Detection
- Upload images (JPG, PNG) or videos (MP4, MOV, AVI)
- Real-time processing with progress indicators
- Drag-and-drop interface
- Download annotated results
- JSON API for detection data

### 📹 Live Camera Detection
- Real-time object detection using webcam
- Live video stream with detection overlays
- Performance statistics (FPS, processing time)
- Detection history and summaries
- Responsive controls (start/stop/pause)

### 🔧 Production Features
- File size limits (50MB) and validation
- Automatic cleanup of old files (24h)
- Error handling and logging
- Responsive design (mobile-friendly)
- API endpoints for integration
- Security measures (filename randomization)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam (for live detection)
- Modern web browser with WebRTC support

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd yolo-detection-app
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python app.py
```

4. **Open your browser:**
Navigate to `http://localhost:5000`

## 📖 Usage

### File Upload Detection
1. Go to the main page (`/`)
2. Drag and drop a file or click to browse
3. Supported formats: JPG, PNG, MP4, MOV, AVI (max 50MB)
4. Click "Detect Objects" to process
5. View results with bounding boxes and confidence scores
6. Download the annotated file or get JSON data

### Live Camera Detection
1. Click "Live Camera Detection" on the main page
2. Click "Start Camera" and allow camera permissions
3. See real-time object detection with overlays
4. Monitor performance statistics
5. Use pause/resume controls as needed

## 🎮 Example cURL Commands

### Upload and detect objects in an image:
```bash
curl -X POST -F "file=@sample_image.jpg" http://localhost:5000/detect
```

### Get detection results as JSON:
```bash
curl "http://localhost:5000/result/detected_sample_image.jpg?format=json"
```

### Check API status:
```bash
curl http://localhost:5000/api/status
```

### Live detection (send base64 image):
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQSkZJRg..."}' \
  http://localhost:5000/live_detect
```

## 🏗️ Architecture

### Core Components
- **app.py**: Main Flask application with routes
- **utils/detection.py**: YOLO detection logic and model management
- **utils/cleanup.py**: File cleanup and maintenance
- **config.py**: Configuration management
- **templates/**: HTML templates with Tailwind CSS
- **static/**: CSS and JavaScript assets

### API Endpoints
- `GET /`: Main upload interface
- `GET /live`: Live camera detection page
- `POST /detect`: Process uploaded files
- `POST /live_detect`: Process camera frames
- `GET /result/<filename>`: View/download results
- `GET /api/status`: API status and configuration

### File Structure
```
yolo-detection-app/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── test_detection.py     # Unit tests
├── README.md            # This file
├── utils/
│   ├── __init__.py
│   ├── detection.py     # YOLO detection logic
│   └── cleanup.py       # File management
├── templates/
│   ├── index.html       # Main upload page
│   ├── live.html        # Live detection page
│   ├── result.html      # Results display
│   └── error.html       # Error page
├── static/
│   └── style.css        # Custom CSS
├── uploads/             # Temporary upload storage
├── static/results/      # Detection results
└── samples/             # Sample test files
```

## ⚙️ Configuration

### Environment Variables
- `YOLO_MODEL`: Model to use (default: `yolov8n.pt`)
- `SECRET_KEY`: Flask secret key
- `MAX_VIDEO_DURATION`: Max video length in seconds (default: 30)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Options
- `yolov8n.pt`: Nano (fastest, least accurate)
- `yolov8s.pt`: Small
- `yolov8m.pt`: Medium  
- `yolov8l.pt`: Large
- `yolov8x.pt`: Extra Large (slowest, most accurate)

### Customization
Edit `config.py` to modify:
- File size limits
- Allowed extensions
- Cleanup intervals
- Storage paths

## 🧪 Testing

Run the unit tests:
```bash
python test_detection.py
```

The tests cover:
- YOLO model initialization
- Image/video detection
- Detection data extraction
- Error handling
- Configuration validation

## 🔐 Security Features

- **File Validation**: Extension and size limits
- **Filename Randomization**: Prevents path traversal
- **Automatic Cleanup**: Regular removal of old files  
- **Error Handling**: Graceful failure modes
- **Input Sanitization**: Safe file processing

## 📱 Browser Support

### File Upload
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

### Live Detection (WebRTC required)
- Chrome 70+
- Firefox 65+
- Safari 14+
- Edge 80+

## 🚨 Troubleshooting

### Common Issues

**Camera not working:**
- Check browser permissions
- Ensure HTTPS for production
- Try different browsers

**Detection too slow:**
- Use smaller YOLO model (yolov8n.pt)
- Reduce video resolution
- Check system resources

**Upload fails:**
- Check file size (max 50MB)
- Verify file format
- Check disk space

### Logs
Check the application logs for detailed error information:
```bash
python app.py 2>&1 | tee app.log
```

## 📊 Performance Tips

1. **Model Selection**: Use `yolov8n.pt` for speed, `yolov8x.pt` for accuracy
2. **Hardware**: GPU acceleration with PyTorch CUDA
3. **Video**: Keep videos under 30 seconds for best experience
4. **Memory**: Monitor memory usage with large files
5. **Cleanup**: Regular file cleanup prevents disk issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📜 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the detection models
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Tailwind CSS](https://tailwindcss.com/) for the UI styling

---

**Happy Detecting! 🎯**