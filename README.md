# AI

```
ai-multimodal-pipeline/
│
├── config/                 # Configuration files for absolute pathing & thresholds
│   └── config.yaml
│
├── data/                   # Data storage
│   ├── raw/                # Original uploaded or sample .mp4 videos
│   └── processed/
│       ├── audio/          # Extracted .wav files from the pre-processor
│       └── frames/         # Extracted .jpg frames for vision models
│   └── training/           # Speech model training dataset
│        └── speech_commands/          
│           ├── validation_list.txt/    
│           ├── testing_list.txt/     
│           └── ...(other word folders)
│
├── models/                         # Pre-trained and custom model weights
│   ├── face_resnet50.weights.h5    # Your existing local ResNet-50 weights
│   ├── object_yolo.pt              # YOLO weights
│   └── speech_ctc.pth              # Your trained custom speech model weights
│
├── notebooks/              # Prototyping and training scripts
│   └── speech_training.ipynb # Notebook to train/fine-tune on Google Speech Commands
│
├── src/                    # Core application source code
│   ├── __init__.py
│   ├── preprocessor.py     # Video/Audio separation and frame extraction
│   ├── face_detector.py    # ResNet-50 inference logic
│   ├── object_tracker.py   # YOLO inference and tracking logic
│   └── speech_analyzer.py  # Spectrogram generation & CTC model architecture
│ 
├── tests/                       # Local testing of application source code
│   ├── __init__.py
│   ├── test_preprocessor.py     # Test Video/Audio separation and frame extraction
│   ├── test_face_detector.py    # Test ResNet-50 inference logic
│   ├── test_object_tracker.py   # Test YOLO inference and tracking logic
│   └── test_speech_analyzer.py  # Test Spectrogram generation & CTC model architecture
│
├── main.py                 # Flask API entry point & Orchestrator
├── requirements.txt        # Python package dependencies
└── README.md               # Setup instructions and documentation
```

