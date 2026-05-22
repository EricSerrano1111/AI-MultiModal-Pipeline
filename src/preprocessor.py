import os
import cv2
import logging
from moviepy import VideoFileClip

# Configure basic logging for our system
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoPreprocessor:
    """
    Handles the extraction of audio tracks and image frames from raw video files.
    """
    def __init__(self, video_path: str, audio_out_dir: str, frames_out_dir: str):
        self.video_path = video_path
        self.audio_out_dir = audio_out_dir
        
        # Create a subfolder for this specific video's frames
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.frames_out_dir = os.path.join(frames_out_dir, self.video_name)
        
        # Ensure destination directories exist before writing to them
        os.makedirs(self.audio_out_dir, exist_ok=True)
        os.makedirs(self.frames_out_dir, exist_ok=True)

    def extract_audio(self) -> str:
        """Strips the audio from the MP4 and saves it as a .wav file."""
        logging.info(f"Extracting audio from {self.video_name}...")
        audio_output_path = os.path.join(self.audio_out_dir, f"{self.video_name}.wav")
        
        try:
            video_clip = VideoFileClip(self.video_path)
            # We use 16000 Hz, a standard sample rate for speech recognition models
            video_clip.audio.write_audiofile(audio_output_path, fps=16000, logger=None)
            video_clip.close()
            logging.info(f"Audio saved to: {audio_output_path}")
            return audio_output_path
        except Exception as e:
            logging.error(f"Failed to extract audio: {e}")
            raise

    def extract_frames(self, extract_fps: int = 1) -> str:
        """
        Slices the video into individual frames. 
        Defaults to 1 frame per second to prevent overloading downstream models.
        """
        logging.info(f"Extracting frames at {extract_fps} FPS...")
        vid_cap = cv2.VideoCapture(self.video_path)
        
        if not vid_cap.isOpened():
            raise ValueError(f"Error opening video file: {self.video_path}")

        original_fps = round(vid_cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, int(original_fps / extract_fps))
        
        current_frame = 0
        saved_count = 0

        while True:
            success, frame = vid_cap.read()
            if not success:
              break # End of video
                
            # Only save the frame if it matches our interval
            if current_frame % frame_interval == 0:
                frame_filename = os.path.join(self.frames_out_dir, f"frame_{current_frame:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                
            current_frame += 1

        vid_cap.release()
        logging.info(f"Successfully extracted {saved_count} frames to: {self.frames_out_dir}")
        return self.frames_out_dir

    def process_all(self, fps: int = 1) -> dict:
        """Executes full pipeline and returns output paths."""
        audio_path = self.extract_audio()
        frames_path = self.extract_frames(extract_fps=fps)
        
        return {
            "audio_path": audio_path,
            "frames_directory": frames_path
        }