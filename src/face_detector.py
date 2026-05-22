import os
import cv2
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceAnalyzer:
    """
    Applies a two-stage pipeline: Detects faces via OpenCV Haar Cascades, 
    then processes cropped faces using a local TensorFlow/Keras ResNet-50 model.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Load Stage 1 Detector (OpenCV Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        # Initialize Stage 2 Deep Learning Model (ResNet-50)
        self.model = self._load_keras_model()

    def _load_keras_model(self):
        """Safely loads weights into the ResNet architecture."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ResNet-50 weights not found at: {self.model_path}")
        
        try:
            # Initialize the base ResNet-50 architecture
            logging.info("Initializing ResNet-50 architecture...")
            
            # NOTE: If trained as a feature extractor (without the 1000-class output), 
            # will need to change this to: ResNet50(weights=None, include_top=False, pooling='avg')
            model = ResNet50(weights=None)
            
            # Load the saved weights into the initialized architecture
            model.load_weights(self.model_path)
            logging.info(f"Successfully loaded weights from {self.model_path}.")
            
            return model
            
        except ValueError as ve:
            logging.error(f"Architecture mismatch.. Error: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error loading Keras weights: {e}")
            raise

    def analyze_frame(self, frame_path: str) -> list:
        """Detects faces in a frame and runs them through the Keras ResNet-50 model."""
        image = cv2.imread(frame_path)
        if image is None:
            logging.warning(f"Could not read frame at {frame_path}")
            return []

        # Haar Cascades require grayscale images for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect bounding boxes
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        
        # Pass cropped faces through ResNet-50
        for (x, y, w, h) in faces:
            # Crop the face out of the original BGR image
            face_crop = image[y:y+h, x:x+w]
            
            # Convert BGR to RGB (OpenCV uses BGR, Keras expects RGB)
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # ResNet-50 expects exactly 224x224 pixel inputs
            face_resized = cv2.resize(face_rgb, (224, 224))
            
            # Convert to numpy array and add batch dimension (shape becomes: 1, 224, 224, 3)
            img_array = np.expand_dims(face_resized, axis=0)
            
            # Apply ResNet-specific scaling (zero-centering color channels)
            img_preprocessed = preprocess_input(img_array)
            
            # Run inference
            predictions = self.model.predict(img_preprocessed, verbose=0)
            
            # Extract sample of the output (first 5 values) to keep JSON lightweight
            feature_vector = predictions[0].tolist()[:5] 
            
            results.append({
                "bounding_box": [int(x), int(y), int(w), int(h)],
                "face_features_sample": feature_vector
            })
            
        return results