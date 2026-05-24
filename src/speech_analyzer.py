import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
import soundfile as sf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Redefine same architecture built notebook
class KeywordSpotterCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(KeywordSpotterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 4, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SpeechAnalyzer:
    """
    Loads custom PyTorch CNN weights and predicts keywords from an audio file.
    """
    def __init__(self, model_path: str):
        self.classes = ["yes", "no", "stop", "go"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize blueprint
        self.model = KeywordSpotterCNN(num_classes=len(self.classes))
        
        # Load weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Custom model weights not found at: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # Set to inference mode (turns off dropout)
        
        # Exact same Spectrogram settings as training
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        ).to(self.device)

    def analyze_audio(self, audio_path: str) -> dict:
        """Processes a 1-second audio clip and predicts the keyword."""
        try:
            # Use soundfile workaround
            audio_data, sample_rate = sf.read(audio_path, dtype='float32')
            
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1) # Stereo to Mono
                
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
            
            # Pad to exactly 16,000 samples
            target_length = 16000
            current_length = waveform.shape[1]
            
            if current_length > target_length:
                waveform = waveform[:, :target_length]
            elif current_length < target_length:
                padding = torch.zeros(1, target_length - current_length)
                waveform = torch.cat((waveform, padding), dim=1)
                
            # Convert to Spectrogram and add Batch dimension: [1, 1, 64, 32]
            waveform = waveform.to(self.device)
            spectrogram = self.mel_transform(waveform).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(spectrogram)
                # Convert raw logits to readable percentages
                probabilities = F.softmax(outputs, dim=1)[0] * 100
                
                # Find highest score
                max_prob, predicted_idx = torch.max(probabilities, 0)
                predicted_word = self.classes[predicted_idx.item()]
                
            return {
                "detected_keyword": predicted_word,
                "confidence": round(max_prob.item(), 2)
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze audio: {e}")
            return {"error": str(e)}