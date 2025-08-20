# 🧪 Test Commands and Examples

This file contains example commands and test cases for the YOLO Detection Flask App.

## cURL Command Examples

### 1. Check API Status
```bash
curl -X GET http://localhost:5000/api/status
```

**Expected Response:**
```json
{
  "status": "ready",
  "model": "yolov8n.pt",
  "upload_limit_mb": 50,
  "allowed_extensions": [".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"],
  "max_video_duration": 30
}
```

### 2. Upload Image for Detection
```bash
# Create a test image first (you can use any image file)
curl -X POST \
  -F "file=@path/to/your/image.jpg" \
  http://localhost:5000/detect
```

### 3. Get Detection Results as JSON
```bash
# Replace 'detected_image.jpg' with actual result filename
curl -X GET "http://localhost:5000/result/detected_image.jpg?format=json"
```

**Expected Response:**
```json
{
  "filename": "detected_image.jpg",
  "detections": [
    {
      "class": "person",
      "confidence": 0.89,
      "bbox": [100, 150, 300, 450]
    },
    {
      "class": "bicycle",
      "confidence": 0.76,
      "bbox": [320, 200, 480, 380]
    }
  ],
  "runtime_ms": 245.7,
  "model_name": "yolov8n.pt",
  "file_info": {
    "filename": "detected_image.jpg",
    "size_kb": 156.8,
    "created": "2024-01-15 14:30:22"
  }
}
```

### 4. Live Detection Frame Processing
```bash
# Note: This requires a base64-encoded image
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
  }' \
  http://localhost:5000/live_detect
```

### 5. Download Detection Result
```bash
curl -X GET \
  -o "downloaded_result.jpg" \
  http://localhost:5000/download/detected_image.jpg
```

## Python Test Examples

### Simple Detection Test
```python
import requests
import json

# Test API status
response = requests.get('http://localhost:5000/api/status')
print("API Status:", response.json())

# Test file upload (replace with actual image path)
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/detect', files=files)
    
if response.status_code == 200:
    print("Detection successful!")
    # The response will be a redirect to results page
else:
    print("Detection failed:", response.json())
```

### Live Detection Simulation
```python
import requests
import base64
import cv2
import json

def test_live_detection():
    # Capture a frame (you can also load from file)
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data = f"data:image/jpeg;base64,{img_base64}"
        
        # Send to live detection endpoint
        response = requests.post('http://localhost:5000/live_detect', 
                               json={'image': img_data})
        
        if response.status_code == 200:
            result = response.json()
            print(f"Detected {len(result['detections'])} objects:")
            for detection in result['detections']:
                print(f"  {detection['class']}: {detection['confidence']:.2f}")
        else:
            print("Live detection failed:", response.text)

# Run the test
test_live_detection()
```

## JavaScript Test Examples (Browser Console)

### Upload File via JavaScript
```javascript
// Test file upload via JavaScript
async function testFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            console.log('Upload successful!');
            // Handle redirect or JSON response
        } else {
            const error = await response.json();
            console.error('Upload failed:', error);
        }
    } catch (error) {
        console.error('Network error:', error);
    }
}

// Usage: Select a file input element and call testFileUpload(fileInput.files[0])
```

### Live Detection Test
```javascript
// Test live detection with canvas
async function testLiveDetection() {
    // Get video stream
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();
    
    // Wait for video to be ready
    video.onloadedmetadata = async () => {
        // Create canvas to capture frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        // Capture frame
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to detection endpoint
        try {
            const response = await fetch('/live_detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            console.log('Detection result:', result);
        } catch (error) {
            console.error('Detection failed:', error);
        }
        
        // Stop stream
        stream.getTracks().forEach(track => track.stop());
    };
}

// Run the test
testLiveDetection();
```

## Performance Testing

### Load Testing with Multiple Files
```python
import requests
import concurrent.futures
import time

def upload_file(file_path):
    start_time = time.time()
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5000/detect', files=files)
    
    end_time = time.time()
    return {
        'status_code': response.status_code,
        'processing_time': end_time - start_time,
        'file': file_path
    }

def load_test():
    # List of test files
    test_files = ['test1.jpg', 'test2.jpg', 'test3.jpg']  # Add your test files
    
    # Upload files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(upload_file, file_path) for file_path in test_files]
        results = [future.result() for future in futures]
    
    # Print results
    for result in results:
        print(f"File: {result['file']}")
        print(f"Status: {result['status_code']}")
        print(f"Time: {result['processing_time']:.2f}s")
        print("---")

# Run load test
load_test()
```

### Memory Usage Monitoring
```python
import psutil
import requests
import time

def monitor_detection():
    process = psutil.Process()  # Current process
    
    print("Before detection:")
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    # Upload file for detection
    with open('large_video.mp4', 'rb') as f:
        files = {'file': f}
        start_time = time.time()
        response = requests.post('http://localhost:5000/detect', files=files)
        end_time = time.time()
    
    print("After detection:")
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"Processing time: {end_time - start_time:.2f}s")
    print(f"Status: {response.status_code}")

monitor_detection()
```

## Error Testing

### Test File Size Limit
```bash
# Create a large file (over 50MB)
dd if=/dev/zero of=large_file.jpg bs=1M count=51

# Try to upload it (should fail)
curl -X POST \
  -F "file=@large_file.jpg" \
  http://localhost:5000/detect
```

### Test Unsupported Format
```bash
# Try to upload an unsupported file type
echo "test content" > test.txt
curl -X POST \
  -F "file=@test.txt" \
  http://localhost:5000/detect
```

### Test Invalid JSON for Live Detection
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}' \
  http://localhost:5000/live_detect
```

## Automated Test Script

Create a `test_all.py` file:

```python
#!/usr/bin/env python3
"""
Comprehensive test script for YOLO Detection App
"""

import requests
import json
import time
import os
from pathlib import Path

BASE_URL = 'http://localhost:5000'

def test_api_status():
    """Test API status endpoint"""
    print("Testing API status...")
    response = requests.get(f'{BASE_URL}/api/status')
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert data['status'] == 'ready'
    print("✅ API status test passed")

def test_file_upload():
    """Test file upload functionality"""
    print("Testing file upload...")
    
    # Create a small test image
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    test_file = 'test_image.jpg'
    img.save(test_file)
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f'{BASE_URL}/detect', files=files)
        
        # Should redirect on success
        assert response.status_code in [200, 302]
        print("✅ File upload test passed")
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test with no file
    response = requests.post(f'{BASE_URL}/detect')
    assert response.status_code == 400
    
    # Test with unsupported file
    test_file = 'test.txt'
    with open(test_file, 'w') as f:
        f.write('test content')
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f'{BASE_URL}/detect', files=files)
        
        assert response.status_code == 400
        print("✅ Error handling test passed")
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    """Run all tests"""
    print("Starting comprehensive tests...\n")
    
    try:
        test_api_status()
        test_file_upload()
        test_error_handling()
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
```

Run with:
```bash
python test_all.py
```

These examples provide comprehensive testing scenarios for all aspects of the YOLO Detection Flask App.