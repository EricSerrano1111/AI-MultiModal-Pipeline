import os
import sys
import unittest

# Ensure Python can find src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.face_detector import FaceAnalyzer

class TestFaceDetector(unittest.TestCase):
    
    def setUp(self):
        """Set up test variables and initialize model."""
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_path = os.path.join(self.base_dir, 'models', 'face_resnet50.weights.h5')
        self.frames_dir = os.path.join(self.base_dir, 'data', 'processed', 'frames', 'test_video')
        
        # Ensure the required files exist before running deep learning model
        if not os.path.exists(self.model_path):
            self.skipTest(f"Model weights not found at {self.model_path}.")
            
        if not os.path.exists(self.frames_dir) or len(os.listdir(self.frames_dir)) == 0:
            self.skipTest("No extracted frames found. Please run the preprocessor test first.")
            
        # Grab very first frame from the directory to test
        self.test_frame = os.path.join(self.frames_dir, os.listdir(self.frames_dir)[0])
        
        # Initialize deep learning analyzer
        self.analyzer = FaceAnalyzer(model_path=self.model_path)

    def test_face_analysis(self):
        """Pass a frame into the model and verify the output structure."""
        print(f"\nAnalyzing frame: {os.path.basename(self.test_frame)}")
        results = self.analyzer.analyze_frame(self.test_frame)
        
        # Verify the result is list
        self.assertIsInstance(results, list, "Output should be a list of detected faces.")
        
        # If the Haar Cascade found a face in this specific frame, verify the ResNet JSON structure
        if len(results) > 0:
            print(f"Success: Found {len(results)} face(s) in the frame.")
            first_face = results[0]
            
            self.assertIn("bounding_box", first_face, "Missing bounding box data.")
            self.assertIn("face_features_sample", first_face, "Missing ResNet feature data.")
            
            self.assertEqual(len(first_face["bounding_box"]), 4, "Bounding box must be exactly [x, y, w, h].")
            self.assertEqual(len(first_face["face_features_sample"]), 5, "Feature sample must be exactly 5 floats.")
        else:
            print("Notice: No faces detected in this specific frame by the Haar Cascade.")

if __name__ == '__main__':
    unittest.main()