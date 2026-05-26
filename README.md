# Multi-Modal AI Video Orchestrator: ResNet, YOLO, PyTorch

## Overview
An end-to-end, multi-modal machine learning pipeline designed to process raw video input and simultaneously execute speech recognition, facial feature localization, and real-time object tracking. This project orchestrates three distinct deep learning models a custom PyTorch CNN for Keyword Spotting (KWS), a Keras ResNet-50 pipeline for facial detection, and an Ultralytics YOLOv8 for object tracking behind a unified FastAPI backend. The system is fully containerized with a Streamlit web interface, allowing users to upload .mp4 files and receive a consolidated, multi-modal JSON analysis without interacting with the underlying code.

## Getting Started
Deploying a multi-model architecture locally requires a strict dependency environment and a dual-server setup. Follow the steps below to configure your local machine, manage the necessary deep learning weights, and launch both the orchestration API and the frontend Web UI.

### Dependencies:
* Python 3.9+
* Anaconda
* Git

### Environment Setup
If you are using Anaconda, you can recreate the exact environment using the .yml file:
Bash - 
* conda env create -f environment.yml
* conda activate multimodal-env

```
ai-multimodal-pipeline/
│
├── config/                 # Configuration files for absolute pathing & thresholds
│   └── config.yml
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
├── api.py                  # FastAPI orchestration script acts as wrapper for models
├── app.py                  # Frontend UI allow file uploads send to FastAPI server
├── visualize_report.py     # Visual plot results for processed data
│
├── requirements.txt        # Python package dependencies
├── environment.yml         # Configuration file for reproducibility
└── README.md               # Setup instructions and documentation
```

### Model Weights Configuration
Due to GitHub file size limits, the deep learning model weights are not included in this repository. To run the pipeline locally, ensure you have the following weights saved in your models/ directory:
* face_resnet50.weights.h5 (Keras ResNet-50)
* custom_kws.pth (PyTorch CNN)
* Note: YOLOv8 (yolov8n.pt) will download automatically via the Ultralytics library on first run.

Verify your paths match the config/config.yaml file:
model_paths:
  face_resnet: "models/face_resnet50.weights.h5"
  speech_cnn: "models/custom_kws.pth"
  tracker_dir: "models/" 

### Booting the Architecture
This system relies on a dual-server architecture, so you will need two terminal windows running simultaneously.

**Terminal 1: Start the Orchestration API**
Bash:  uvicorn api:app –reload 

**Terminal 2: Start the Web UI**
Bash: streamlit run app.py

## Development & Architecture
This project was developed in three distinct engineering phases to ensure modularity and scalability:

* Phase 1: Prototyping (Jupyter Lab)

    Initial development began in a sandbox environment. The custom PyTorch Convolutional Neural Network (CNN) for Keyword Spotting (KWS) was trained, tested, and validated within Jupyter Notebooks using the Google Speech Commands dataset before being exported as a `.pth` weight file.

    Dataset Scoping: Due to the massive storage footprint and processing overhead of the complete Google Speech Commands dataset, the training pipeline was strategically scoped to classify a targeted subset of highly distinct keywords (`yes`, `no`, `stop`, `go`).

    Audio Preprocessing: Because CNNs are inherently designed for image processing, the raw 1D audio waveforms had to be mathematically transformed into Mel Spectrograms. This step converted audio frequencies into a 2D visual feature map, allowing the CNN to effectively "see" and learn the spatial patterns of human speech. 
    Once validated, the model was exported as a production-ready `.pth` weight file.

![mel spectogram](/images/img1.png)

* Phase 2: Object-Oriented Refactoring

    The prototype logic was decoupled from the Jupyter Notebook environment and refactored into strict, object-oriented Python classes. Specifically, the architecture was divided into the VideoPreprocessor, SpeechAnalyzer, FaceAnalyzer, and ObjectTracker. This modular design ensured that the audio, facial, and object tracking pipelines can operate and be maintained independently. Alongside the class definitions, dedicated test scripts were developed to validate each component in isolation prior to full integration.

    The implementation of these test scripts was a critical engineering step. By testing each class locally, I could independently verify input and output operations, such as ensuring the preprocessor successfully extracted the mel spectrograms before attempting to pass data into the speech model. Isolating these dependencies guaranteed that the underlying deep learning models functioned flawlessly on their own.

    To validate the final output of these integrated models, a dedicated visualization script (visualize_report.py) was also engineered during this phase. This script parses the pipeline's raw JSON output and constructs a high-resolution bar chart mapping the frame-by-frame face and object detections, complete with an overlay of the audio keyword confidence. 

    Developing this data storytelling component proved that the system's output was not only accurate but usable, translating complex multi-modal data into an accessible format before introducing the complexity of the FastAPI orchestration layer in Phase 3.


![Barchart](/images/img2.png)

* Phase 3: API Orchestration & UI

    To transition the project from a set of local scripts into a production-ready application, the final phase focused on system orchestration and user accessibility.
    First, all hardcoded paths, frame rates, and model parameters were abstracted into a central YAML configuration file (config.yaml). Decoupling these settings from the core logic ensures the pipeline is maintainable and scalable, allowing variables to be adjusted without ever modifying the underlying Python code.

    Next, a robust orchestration layer was built using FastAPI (api.py). This backend server script serves to mask the complexity of the individual AI models. It handles incoming HTTP requests, parses the YAML configuration, routes the uploaded video through the object-oriented classes developed in Phase 2, and seamlessly aggregates the audio and visual results into a single, consolidated JSON response.

    Finally, to make the system accessible to end-users, a frontend web application was developed using Streamlit (app.py). This User Interface allows users to simply drag and drop or upload video files directly into their web browser. The UI communicates with the FastAPI backend, retrieves the structured JSON analysis, and renders the speech recognition confidence and visual tracking metrics in a clean, interactive dashboard, successfully completing the end-to-end pipeline.

![UI](/images/img3.png)

![UI](/images/img4.png)

![UI](/images/img5.png)

## Help / Issue Log

**Issue 1: Classname Mismatches During Pipeline Integration Description**

During the integration of the multimodal pipeline, the main orchestration script threw an error when attempting to run the video preprocessing module. LLM Assistance Requested: 

**Resolution:**

 The LLM successfully identified that the class being instantiated in the main orchestration script did not match the class name actually defined inside preprocessor.py. I updated the instantiation logic, and the script executed successfully. 

**Issue 2: FastAPI Server Rejecting .MOV File Uploads Description**

For the final system test, the recorded test video was saved as an Apple .MOV file. When uploaded via the Streamlit UI, the FastAPI server threw an HTTP 400 error stating "Only .mp4 files are supported." LLM Assistance Requested: Described the 400 error and the .MOV file format to the LLM to request a way to bypass the hardcoded .mp4 limitation in the API. 

**Resolution:**

Rather than altering the validation checks in api.py and app.py to dynamically accept multiple MIME types, I decided to convert the source video file from a .MOV format directly to an .MP4. This satisfied the application's strict file requirements and allowed the video to be processed successfully without needing to modify the codebase or restart the Uvicorn server.

**Issue 3: Model Pathing and Missing Weights Description**

Upon the initial run of the API orchestration layer, the system threw a "File Not Found" error when attempting to initialize the ResNet and PyTorch models. 

**Resolution:**

By reviewing the project structure, I realized that although the YAML config file was correctly pointing to the /models directory, the heavy weight files (.h5 and .pth) were missing locally because they had been excluded by the .gitignore file. I manually placed the necessary weight files back into the correct local directory, which resolved the pathing error and allowed the system to initialize successfully. 

## TODO

While the current architecture successfully orchestrates multiple local AI models into a unified pipeline, there are several opportunities for future optimization and scalability:

1. **System Containerization:**

    Wrap the FastAPI server, Streamlit UI, and local model weights into Docker containers. This will eliminate the need for manual Conda environment setup and ensure the dual-server architecture can be deployed anywhere with a single docker-compose command.

2. **Cloud Infrastructure Deployment:** 

    Migrate the local FastAPI backend to a managed cloud service, such as Google Cloud Platform (GCP), Microsoft Azure, or AWS. This would allow the Web UI to be hosted publicly without relying on a local machine's hardware for the heavy deep learning processing.

3. **Expanded Speech Recognition:** 

    Upgrade the custom PyTorch CNN to process the entirety of the Google Speech Commands dataset or integrate a more robust pre-trained model to allow for continuous speech transcription rather than isolated keyword spotting.

4. **Database Integration:** 

    Implement a local NoSQL or PostgreSQL database connection within the FastAPI layer to persistently store the generated JSON reports. This would allow users to log in to the Streamlit UI and view a historical dashboard of past video analyses rather than the data disappearing after the browser session ends. 


## Authors
* Lead Developer – **Eric Serrano**

## Version History
* 0.1 – Initial Release 

## License
The MIT License (MIT)
Copyright (c) 2026 Eric Serrano

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
