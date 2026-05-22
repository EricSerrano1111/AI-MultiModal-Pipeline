import os
import json
import logging
import yaml
from src.preprocessor import VideoPreprocessor
from src.face_detector import FaceAnalyzer 
from src.object_tracker import ObjectTracker
from src.speech_analyzer import SpeechAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_config(config_path="config/config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_pipeline():
    logging.info("Starting Multi-Modal AI Pipeline..")
    
    # Load Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    config = load_config(config_path)
    
    # Define Paths from Config
    video_path = os.path.join(base_dir, config['pipeline']['input_video'])
    output_dir = os.path.join(base_dir, 'data', 'processed')
    audio_out_dir = os.path.join(output_dir, 'audio')
    frames_out_dir = os.path.join(output_dir, 'frames')
    
    resnet_weights = os.path.join(base_dir, config['model_paths']['face_resnet'])
    speech_weights = os.path.join(base_dir, config['model_paths']['speech_cnn'])
    tracker_dir = os.path.join(base_dir, config['model_paths']['tracker_dir'])
    
    # Pre-flight check
    if not os.path.exists(video_path):
        logging.error(f"Cannot find video at {video_path}. Please check config.yaml.")
        return

    # Initialize the Modules
    logging.info("Loading models into memory..")
    preprocessor = VideoPreprocessor(video_path, audio_out_dir, frames_out_dir)
    face_detector = FaceAnalyzer(resnet_weights)
    object_tracker = ObjectTracker(tracker_dir)
    speech_analyzer = SpeechAnalyzer(speech_weights)
    
    # Preprocess the Video (Using config FPS)
    logging.info("Phase 1: Preprocessing Video..")
    target_fps = config['pipeline']['target_fps']
    prep_results = preprocessor.process_all(fps=target_fps)
    
    audio_path = prep_results["audio_path"]
    frames_dir = prep_results["frames_directory"]
    
    if not audio_path or not frames_dir:
        logging.error("Preprocessing failed. Exiting.")
        return

    # Analyze Audio
    logging.info("Phase 2: Analyzing Audio (Keyword Spotting)..")
    speech_results = speech_analyzer.analyze_audio(audio_path)
    logging.info(f"Detected Keyword: {speech_results.get('detected_keyword')} ({speech_results.get('confidence')}%)")
    
    # Analyze Visuals
    logging.info("Phase 3: Analyzing Visuals..")
    visual_results = []
    
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    for frame_name in frames:
        frame_path = os.path.join(frames_dir, frame_name)
        logging.info(f"Scanning {frame_name}...")
        
        faces = face_detector.analyze_frame(frame_path)
        objects = object_tracker.analyze_frame(frame_path)
        
        logging.info(f"   -> Found {len(faces)} face(s) and {len(objects)} object(s)")
        
        visual_results.append({
            "frame": frame_name,
            "faces_detected": len(faces),
            "objects_detected": len(objects),
            "object_details": objects
        })

    # Aggregate and Output Report
    logging.info("Phase 4: Generating Report..")
    
    final_report = {
        "video_file": os.path.basename(video_path),
        "pipeline_config": config['pipeline'], # Save the config used for reproducibility
        "audio_analysis": speech_results,
        "visual_analysis": visual_results
    }
    
    report_path = os.path.join(output_dir, 'final_report.json')
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    logging.info(f"Pipeline Complete.. Report saved to: {report_path}")

if __name__ == "__main__":
    run_pipeline()