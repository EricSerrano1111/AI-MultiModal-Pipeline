import os
import cv2
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ObjectTracker:
    """
    Utilizes YOLOv8 to detect and track static and moving objects within a video frame.
    """
    def __init__(self, model_dir: str):
        # Specify models directory so Ultralytics downloads the weights exactly where wanted
        self.model_path = os.path.join(model_dir, 'yolov8n.pt')
        logging.info(f"Loading YOLO model from {self.model_path}...")
        
        # Initialize YOLO. If yolov8n.pt isn't found at the path, it automatically downloads it.
        self.model = YOLO(self.model_path)

    def analyze_frame(self, frame_path: str) -> list:
        """Runs object detection on a single frame and extracts bounding boxes and labels."""
        image = cv2.imread(frame_path)
        if image is None:
            logging.warning(f"Could not read frame at {frame_path}")
            return []

        # Run inference. verbose=False keeps server logs clean.
        results = self.model(image, verbose=False)
        
        detected_objects = []
        
        # Parse the YOLO Results object
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates: [x_min, y_min, x_max, y_max]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Extract confidence score and class ID
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                # Map the class ID to the human-readable string (e.g., "person", "car")
                class_name = self.model.names[class_id]
                
                detected_objects.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 2),
                    "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
                })
                
        return detected_objects