import os
import yaml
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Import existing custom modules
from src.preprocessor import VideoPreprocessor
from src.face_detector import FaceAnalyzer
from src.object_tracker import ObjectTracker
from src.speech_analyzer import SpeechAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize API
app = FastAPI(
    title="Multi-Modal AI Pipeline API",
    description="Orchestrates audio and visual deep learning models.",
    version="1.0.0"
)

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load config once when the API starts
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
config = load_config(config_path)

# Initialize Models globally so they don't reload on every single API call
logging.info("Loading heavy models into memory (FastAPI startup)..")
resnet_weights = os.path.join(base_dir, config['model_paths']['face_resnet'])
speech_weights = os.path.join(base_dir, config['model_paths']['speech_cnn'])
tracker_dir = os.path.join(base_dir, config['model_paths']['tracker_dir'])

face_detector = FaceAnalyzer(resnet_weights)
object_tracker = ObjectTracker(tracker_dir)
speech_analyzer = SpeechAnalyzer(speech_weights)

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Receives a video file, runs it through pipeline, and returns the multi-modal analysis.
    """
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")
        
    logging.info(f"Received request to analyze: {file.filename}")
    
    # Save the uploaded file locally so preprocessor can read it
    temp_video_path = os.path.join(base_dir, "data", "raw", file.filename)
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Define dynamic output directories for specific API call
        output_dir = os.path.join(base_dir, 'data', 'processed')
        audio_out_dir = os.path.join(output_dir, 'audio')
        frames_out_dir = os.path.join(output_dir, 'frames')
        
        # Preprocess Video
        preprocessor = VideoPreprocessor(temp_video_path, audio_out_dir, frames_out_dir)
        target_fps = config['pipeline']['target_fps']
        prep_results = preprocessor.process_all(fps=target_fps)
        
        audio_path = prep_results["audio_path"]
        frames_dir = prep_results["frames_directory"]
        
        if not audio_path or not frames_dir:
            raise HTTPException(status_code=500, detail="Video preprocessing failed.")

        # Analyze Audio
        speech_results = speech_analyzer.analyze_audio(audio_path)
        
        # Analyze Visuals
        visual_results = []
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        for frame_name in frames:
            frame_path = os.path.join(frames_dir, frame_name)
            faces = face_detector.analyze_frame(frame_path)
            objects = object_tracker.analyze_frame(frame_path)
            
            visual_results.append({
                "frame": frame_name,
                "faces_detected": len(faces),
                "objects_detected": len(objects),
                "object_details": objects
            })

        # Construct API Response
        final_report = {
            "status": "success",
            "video_file": file.filename,
            "audio_analysis": speech_results,
            "visual_analysis": visual_results
        }
        
        return JSONResponse(content=final_report)

    except Exception as e:
        logging.error(f"API Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))