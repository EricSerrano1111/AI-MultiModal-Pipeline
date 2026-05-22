import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.speech_analyzer import SpeechAnalyzer

class TestSpeechAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_path = os.path.join(self.base_dir, 'models', 'custom_kws.pth')
        
        # Will test it on a known good "yes" file extracted earlier
        self.test_audio_dir = os.path.join(self.base_dir, 'data', 'training', 'speech_commands', 'yes')
        
        if not os.path.exists(self.model_path):
            self.skipTest(f"Custom model weights not found at {self.model_path}")
            
        if not os.path.exists(self.test_audio_dir) or len(os.listdir(self.test_audio_dir)) == 0:
            self.skipTest("Training audio files not found.")
            
        self.test_audio_path = os.path.join(self.test_audio_dir, os.listdir(self.test_audio_dir)[0])
        
        self.analyzer = SpeechAnalyzer(model_path=self.model_path)

    def test_audio_analysis(self):
        print(f"\nAnalyzing audio file: {os.path.basename(self.test_audio_path)}")
        result = self.analyzer.analyze_audio(self.test_audio_path)
        
        print(f"Prediction: {result.get('detected_keyword')} (Confidence: {result.get('confidence')}%)")
        
        self.assertIn("detected_keyword", result)
        self.assertIn("confidence", result)
        self.assertGreater(result["confidence"], 0.0)

if __name__ == '__main__':
    unittest.main()