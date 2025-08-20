import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
import json
from pathlib import Path
import numpy as np

# Mock ultralytics YOLO before importing our modules
mock_yolo = Mock()
mock_result = Mock()
mock_result.boxes = Mock()
mock_result.names = {0: 'person', 1: 'bicycle', 2: 'car'}
mock_result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

# Mock box object
mock_box = Mock()
mock_box.cls.item.return_value = 0  # person class
mock_box.conf.item.return_value = 0.85  # confidence
mock_box.xyxy.tolist.return_value = [[100, 150, 200, 300]]  # bbox

mock_result.boxes = [mock_box]
mock_yolo.predict.return_value = [mock_result]

with patch('ultralytics.YOLO', return_value=mock_yolo):
    from utils.detection import run_detection, YOLODetector
    from config import Config

class TestYOLODetection(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.test_dir, 'test_image.jpg')
        self.test_video_path = os.path.join(self.test_dir, 'test_video.mp4')
        
        # Create dummy test files
        # Create a simple test image (1x1 pixel)
        from PIL import Image
        img = Image.new('RGB', (1, 1), color='red')
        img.save(self.test_image_path)
        
        # For video, we'll mock cv2 operations
        Path(self.test_video_path).touch()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('ultralytics.YOLO')
    def test_yolo_detector_initialization(self, mock_yolo_class):
        """Test YOLODetector initialization"""
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance
        
        # Mock warming up prediction
        dummy_result = Mock()
        mock_yolo_instance.predict.return_value = [dummy_result]
        
        detector = YOLODetector('yolov8n.pt')
        
        self.assertEqual(detector.model_name, 'yolov8n.pt')
        self.assertEqual(detector.model, mock_yolo_instance)
        mock_yolo_class.assert_called_once_with('yolov8n.pt')
        mock_yolo_instance.predict.assert_called_once()  # Warm-up call
    
    @patch('cv2.imwrite')
    @patch('ultralytics.YOLO')
    def test_image_detection(self, mock_yolo_class, mock_imwrite):
        """Test image detection functionality"""
        # Setup mocks
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance
        
        # Mock result object
        mock_result = Mock()
        mock_result.names = {0: 'person', 1: 'bicycle'}
        mock_result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock box detection
        mock_box = Mock()
        mock_box.cls.item.return_value = 0  # person
        mock_box.conf.item.return_value = 0.92
        mock_box.xyxy.tolist.return_value = [[10, 20, 100, 200]]
        
        mock_result.boxes = [mock_box]
        mock_yolo_instance.predict.return_value = [mock_result]
        
        # Create detector and run detection
        detector = YOLODetector('yolov8n.pt')
        output_path = os.path.join(self.test_dir, 'output.jpg')
        
        result = detector.detect(self.test_image_path, output_path)
        
        # Verify results
        self.assertEqual(result['output_path'], output_path)
        self.assertEqual(result['input_type'], 'image')
        self.assertEqual(len(result['detections']), 1)
        
        detection = result['detections'][0]
        self.assertEqual(detection['class'], 'person')
        self.assertEqual(detection['confidence'], 0.92)
        self.assertEqual(detection['bbox'], [10, 20, 100, 200])
        
        # Verify model was called correctly
        mock_yolo_instance.predict.assert_called()
        mock_imwrite.assert_called_once()
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    @patch('ultralytics.YOLO')
    def test_video_detection(self, mock_yolo_class, mock_video_writer, mock_video_capture):
        """Test video detection functionality"""
        # Setup video capture mock
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            1: 30.0,  # FPS
            7: 150.0  # Frame count
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        # Setup video writer mock
        mock_writer = Mock()
        mock_video_writer.return_value = mock_writer
        
        # Setup YOLO mock
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance
        
        # Mock result with detections
        mock_result = Mock()
        mock_result.names = {0: 'person', 1: 'car'}
        mock_result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        mock_box = Mock()
        mock_box.cls.item.return_value = 1  # car
        mock_box.conf.item.return_value = 0.78
        mock_box.xyxy.tolist.return_value = [[50, 60, 150, 200]]
        
        mock_result.boxes = [mock_box]
        mock_yolo_instance.predict.return_value = [mock_result, mock_result]  # 2 frames
        
        # Create detector and run detection
        detector = YOLODetector('yolov8n.pt')
        output_path = os.path.join(self.test_dir, 'output.mp4')
        
        result = detector.detect(self.test_video_path, output_path)
        
        # Verify results
        self.assertEqual(result['output_path'], output_path)
        self.assertEqual(result['input_type'], 'video')
        self.assertEqual(result['duration'], 5.0)  # 150 frames / 30 fps
        self.assertEqual(result['total_detections'], 2)  # 2 frames with 1 detection each
        
        # Check detection summary (aggregated)
        self.assertEqual(len(result['detections']), 1)  # 1 unique class
        detection_summary = result['detections'][0]
        self.assertEqual(detection_summary['class'], 'car')
        self.assertEqual(detection_summary['count'], 2)
    
    @patch('os.path.exists')
    @patch('utils.detection.get_detector')
    def test_run_detection_function(self, mock_get_detector, mock_exists):
        """Test the main run_detection function"""
        # Setup mocks
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_get_detector.return_value = mock_detector
        
        expected_result = {
            'output_path': '/fake/path/detected_test.jpg',
            'detections': [
                {'class': 'person', 'confidence': 0.95, 'bbox': [10, 20, 100, 200]}
            ],
            'model_name': 'yolov8n.pt',
            'runtime_ms': 150.5
        }
        mock_detector.detect.return_value = expected_result
        
        # Run detection
        result = run_detection(self.test_image_path)
        
        # Verify
        mock_get_detector.assert_called_once()
        mock_detector.detect.assert_called_once()
        self.assertEqual(result, expected_result)
    
    def test_detection_data_extraction(self):
        """Test detection data extraction from YOLO results"""
        detector = YOLODetector('yolov8n.pt')
        
        # Create mock result with multiple detections
        mock_result = Mock()
        mock_result.names = {0: 'person', 1: 'bicycle', 2: 'car'}
        
        # Mock multiple boxes
        mock_box1 = Mock()
        mock_box1.cls.item.return_value = 0  # person
        mock_box1.conf.item.return_value = 0.95
        mock_box1.xyxy.tolist.return_value = [[10, 20, 100, 200]]
        
        mock_box2 = Mock()
        mock_box2.cls.item.return_value = 2  # car
        mock_box2.conf.item.return_value = 0.87
        mock_box2.xyxy.tolist.return_value = [[200, 100, 400, 300]]
        
        mock_result.boxes = [mock_box1, mock_box2]
        
        detections = detector._extract_detections(mock_result)
        
        self.assertEqual(len(detections), 2)
        
        # Check first detection
        self.assertEqual(detections[0]['class'], 'person')
        self.assertEqual(detections[0]['confidence'], 0.95)
        self.assertEqual(detections[0]['bbox'], [10, 20, 100, 200])
        
        # Check second detection
        self.assertEqual(detections[1]['class'], 'car')
        self.assertEqual(detections[1]['confidence'], 0.87)
        self.assertEqual(detections[1]['bbox'], [200, 100, 400, 300])
    
    def test_detection_summarization(self):
        """Test detection summary for videos"""
        detector = YOLODetector('yolov8n.pt')
        
        # Create mock detections list (multiple frames)
        detections = [
            {'class': 'person', 'confidence': 0.95},
            {'class': 'person', 'confidence': 0.87},
            {'class': 'car', 'confidence': 0.92},
            {'class': 'person', 'confidence': 0.91},
            {'class': 'bicycle', 'confidence': 0.78},
        ]
        
        summary = detector._summarize_detections(detections)
        
        # Should be sorted by count (descending)
        self.assertEqual(len(summary), 3)  # 3 unique classes
        
        # Person appears 3 times
        person_summary = next(s for s in summary if s['class'] == 'person')
        self.assertEqual(person_summary['count'], 3)
        self.assertAlmostEqual(person_summary['confidence'], 0.91, places=2)  # average
        self.assertEqual(person_summary['max_confidence'], 0.95)
        
        # Car appears 1 time
        car_summary = next(s for s in summary if s['class'] == 'car')
        self.assertEqual(car_summary['count'], 1)
        self.assertEqual(car_summary['confidence'], 0.92)
        
        # Bicycle appears 1 time
        bicycle_summary = next(s for s in summary if s['class'] == 'bicycle')
        self.assertEqual(bicycle_summary['count'], 1)
        self.assertEqual(bicycle_summary['confidence'], 0.78)
    
    def test_unsupported_file_format(self):
        """Test error handling for unsupported file formats"""
        detector = YOLODetector('yolov8n.pt')
        
        unsupported_file = os.path.join(self.test_dir, 'test.txt')
        Path(unsupported_file).touch()
        
        output_path = os.path.join(self.test_dir, 'output.txt')
        
        with self.assertRaises(ValueError) as context:
            detector.detect(unsupported_file, output_path)
        
        self.assertIn('Unsupported file format', str(context.exception))
    
    @patch('cv2.VideoCapture')
    def test_video_duration_limit(self, mock_video_capture):
        """Test video duration limit enforcement"""
        # Setup video capture mock for long video
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            1: 30.0,  # FPS
            7: 1200.0  # Frame count (40 seconds at 30 fps)
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        detector = YOLODetector('yolov8n.pt')
        output_path = os.path.join(self.test_dir, 'output.mp4')
        
        with self.assertRaises(ValueError) as context:
            detector.detect(self.test_video_path, output_path)
        
        self.assertIn('exceeds limit', str(context.exception))


class TestConfig(unittest.TestCase):
    """Test configuration settings"""
    
    def test_config_values(self):
        """Test that config has expected values"""
        self.assertEqual(Config.MAX_CONTENT_LENGTH, 50 * 1024 * 1024)  # 50MB
        self.assertEqual(Config.MAX_VIDEO_DURATION, 30)  # 30 seconds
        self.assertIn('.jpg', Config.ALLOWED_EXTENSIONS)
        self.assertIn('.mp4', Config.ALLOWED_EXTENSIONS)
        self.assertEqual(Config.CLEANUP_AGE_HOURS, 24)


if __name__ == '__main__':
    # Create test directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)