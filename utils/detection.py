import os
import time
import logging
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class YOLODetector:
    """YOLO object detection wrapper"""
    
    def __init__(self, model_name: str = 'yolov8n.pt'):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model and warm it up"""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            # Warm up the model with a dummy prediction
            logger.info("Warming up model...")
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.no_grad():
                self.model.predict(dummy_image, verbose=False)
            logger.info("Model loaded and warmed up successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Run object detection on image or video
        
        Args:
            input_path: Path to input file
            output_path: Path to save annotated output
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        try:
            file_ext = Path(input_path).suffix.lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png']:
                return self._detect_image(input_path, output_path, start_time)
            elif file_ext in ['.mp4', '.mov', '.avi']:
                return self._detect_video(input_path, output_path, start_time)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise

    def _detect_image(self, input_path: str, output_path: str, start_time: float) -> Dict[str, Any]:
        """Process image detection"""
        with torch.no_grad():
            results = self.model.predict(input_path, save=False, verbose=False)
            
        # Get the first result (single image)
        result = results[0]
        
        # Save annotated image
        annotated_img = result.plot()
        cv2.imwrite(output_path, annotated_img)
        
        # Extract detection data
        detections = self._extract_detections(result)
        
        runtime_ms = (time.time() - start_time) * 1000
        
        return {
            'output_path': output_path,
            'detections': detections,
            'model_name': self.model_name,
            'runtime_ms': round(runtime_ms, 2),
            'input_type': 'image'
        }
    
    def _detect_video(self, input_path: str, output_path: str, start_time: float) -> Dict[str, Any]:
        """Process video detection"""
        from config import Config
        
        # Check video duration
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if duration > Config.MAX_VIDEO_DURATION:
            raise ValueError(f"Video duration ({duration:.1f}s) exceeds limit ({Config.MAX_VIDEO_DURATION}s)")
        
        # Process video with YOLO
        with torch.no_grad():
            results = self.model.predict(input_path, save=False, verbose=False)
        
        # Save annotated video
        self._save_annotated_video(input_path, output_path, results)
        
        # Aggregate detections from all frames
        all_detections = []
        for result in results:
            all_detections.extend(self._extract_detections(result))
        
        # Get unique detections summary
        detections_summary = self._summarize_detections(all_detections)
        
        runtime_ms = (time.time() - start_time) * 1000
        
        return {
            'output_path': output_path,
            'detections': detections_summary,
            'model_name': self.model_name,
            'runtime_ms': round(runtime_ms, 2),
            'input_type': 'video',
            'duration': round(duration, 2),
            'total_detections': len(all_detections)
        }
    
    def _save_annotated_video(self, input_path: str, output_path: str, results):
        """Save video with annotations"""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for result in results:
                annotated_frame = result.plot()
                out.write(annotated_frame)
        finally:
            cap.release()
            out.release()
    
    def _extract_detections(self, result) -> List[Dict[str, Any]]:
        """Extract detection data from YOLO result"""
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                detection = {
                    'class': result.names[int(box.cls.item())],
                    'confidence': round(box.conf.item(), 3),
                    'bbox': box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        return detections
    
    def _summarize_detections(self, detections: List[Dict]) -> List[Dict[str, Any]]:
        """Summarize detections by class with counts and average confidence"""
        class_summary = {}
        
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            if class_name not in class_summary:
                class_summary[class_name] = {
                    'class': class_name,
                    'count': 0,
                    'total_confidence': 0,
                    'max_confidence': 0
                }
            
            class_summary[class_name]['count'] += 1
            class_summary[class_name]['total_confidence'] += confidence
            class_summary[class_name]['max_confidence'] = max(
                class_summary[class_name]['max_confidence'], confidence
            )
        
        # Calculate average confidence and format results
        summary = []
        for class_data in class_summary.values():
            avg_confidence = class_data['total_confidence'] / class_data['count']
            summary.append({
                'class': class_data['class'],
                'count': class_data['count'],
                'confidence': round(avg_confidence, 3),
                'max_confidence': round(class_data['max_confidence'], 3)
            })
        
        return sorted(summary, key=lambda x: x['count'], reverse=True)


# Global detector instance
detector = None

def get_detector() -> YOLODetector:
    """Get global detector instance"""
    global detector
    if detector is None:
        from config import Config
        detector = YOLODetector(Config.YOLO_MODEL)
    return detector

def run_detection(input_path: str) -> Dict[str, Any]:
    """
    Main detection function - runs object detection on input file
    
    Args:
        input_path: Path to input image or video file
        
    Returns:
        Dictionary containing:
        - output_path: Path to annotated result file
        - detections: List of detected objects with class, confidence, bbox
        - model_name: Name of YOLO model used
        - runtime_ms: Processing time in milliseconds
    """
    # Generate output filename
    input_file = Path(input_path)
    output_filename = f"detected_{input_file.name}"
    
    from config import Config
    output_path = os.path.join(Config.RESULTS_FOLDER, output_filename)
    
    # Run detection
    detector = get_detector()
    result = detector.detect(input_path, output_path)
    
    logger.info(f"Detection completed: {len(result['detections'])} objects found in {result['runtime_ms']:.2f}ms")
    
    return result