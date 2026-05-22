import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.object_tracker import ObjectTracker

class TestObjectTracker(unittest.TestCase):
    
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.frames_dir = os.path.join(self.base_dir, 'data', 'processed', 'frames', 'test_video')
        
        if not os.path.exists(self.frames_dir) or len(os.listdir(self.frames_dir)) == 0:
            self.skipTest("No extracted frames found. Please run the preprocessor test first.")
            
        self.test_frame = os.path.join(self.frames_dir, os.listdir(self.frames_dir)[0])
        
        # Initialize YOLO tracker
        self.tracker = ObjectTracker(model_dir=self.models_dir)

    def test_object_detection(self):
        print(f"\nAnalyzing frame for objects: {os.path.basename(self.test_frame)}")
        results = self.tracker.analyze_frame(self.test_frame)
        
        self.assertIsInstance(results, list, "Output should be a list of detected objects.")
        
        if len(results) > 0:
            print(f"Success: Found {len(results)} object(s) in the frame..")
            first_object = results[0]
            
            # Print first detected object to terminal to see findings
            print(f"Sample Detection: {first_object['class_name']} ({first_object['confidence'] * 100}%)")
            
            self.assertIn("class_name", first_object)
            self.assertIn("confidence", first_object)
            self.assertIn("bounding_box", first_object)
        else:
            print("Notice: No objects detected in this specific frame..")

if __name__ == '__main__':
    unittest.main()