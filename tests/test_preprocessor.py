import os
import sys
import unittest
import shutil

# Allows test script to import from the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessor import VideoPreprocessor

class TestVideoPreprocessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test variables before each test runs."""
        # Define paths based on our directory structure
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.raw_video_path = os.path.join(self.base_dir, 'data', 'raw', 'test_video.mp4')
        self.audio_out_dir = os.path.join(self.base_dir, 'data', 'processed', 'audio')
        self.frames_out_dir = os.path.join(self.base_dir, 'data', 'processed', 'frames')
        
        # Ensure test video actually exists before testing
        if not os.path.exists(self.raw_video_path):
            self.skipTest(f"Please place a video file at {self.raw_video_path} to run this test.")

        # Initialize class
        self.processor = VideoPreprocessor(
            video_path=self.raw_video_path,
            audio_out_dir=self.audio_out_dir,
            frames_out_dir=self.frames_out_dir
        )

    def test_audio_extraction(self):
        """Test if the audio track is successfully stripped."""
        audio_path = self.processor.extract_audio()
        
        # Verify file was created and is not empty
        self.assertTrue(os.path.exists(audio_path), "Audio file was not created.")
        self.assertGreater(os.path.getsize(audio_path), 0, "Audio file is empty.")

    def test_frame_extraction(self):
        """Test if the video is successfully sliced into frames."""
        frames_dir = self.processor.extract_frames(extract_fps=1)
        
        # Verify directory was created
        self.assertTrue(os.path.exists(frames_dir), "Frames directory was not created.")
        
        # Verify at least one frame was extracted
        extracted_files = os.listdir(frames_dir)
        self.assertGreater(len(extracted_files), 0, "No frames were extracted.")
        
        # Verify the extracted files are images
        self.assertTrue(extracted_files[0].endswith('.jpg'), "Extracted files are not JPEGs.")

    def tearDown(self):
        """Clean up the processed files after tests to keep the directory clean."""
        # Optional: You can remove the generated files here if you want a fresh start every time
        pass

if __name__ == '__main__':
    unittest.main()