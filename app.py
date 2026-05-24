import streamlit as st
import requests
import json
import time

# Page Config
st.set_page_config(
    page_title="Multi-Modal AI Pipeline",
    page_icon="🧠",
    layout="wide"
)

# Header
st.title("🧠 Multi-Modal Video Analysis Dashboard")
st.markdown("""
Upload a video clip to simultaneously run **Speech Recognition**, **Face Detection**, and **Object Tracking** using local deep learning models.
""")
st.divider()

# Main App Logic
API_URL = "http://127.0.0.1:8000/analyze-video"

uploaded_file = st.file_uploader("Upload an MP4 video clip", type=['mp4'])

if uploaded_file is not None:
    # Display the uploaded video
    st.video(uploaded_file)
    
    if st.button("Analyze Video", type="primary", use_container_width=True):
        
        with st.status("Pipeline Running.. This may take a moment depending on video length.", expanded=True) as status:
            st.write("1. Uploading file to Orchestration API..")
            
            # Send the file to the FastAPI backend
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
                
                st.write("2. Triggering Multi-Modal Pipeline (ResNet, YOLO, PyTorch CNN)..")
                start_time = time.time()
                
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    status.update(label=f"Analysis Complete.. (Took {round(time.time() - start_time, 1)}s)", state="complete", expanded=False)
                    data = response.json()
                    
                    st.divider()
                    st.subheader("Analysis Results")
                    
                    # Render Audio Results
                    st.markdown("### Speech Recognition")
                    col1, col2 = st.columns(2)
                    keyword = data['audio_analysis']['detected_keyword'].upper()
                    confidence = data['audio_analysis']['confidence']
                    
                    col1.metric("Detected Keyword", keyword)
                    col2.metric("Confidence Score", f"{confidence}%")
                    
                    # Render Visual Results
                    st.markdown("### Visual Tracking Summary")
                    
                    # Calculate totals
                    total_frames = len(data['visual_analysis'])
                    max_faces = max([frame['faces_detected'] for frame in data['visual_analysis']]) if total_frames > 0 else 0
                    total_objects = sum([frame['objects_detected'] for frame in data['visual_analysis']])
                    
                    vcol1, vcol2, vcol3 = st.columns(3)
                    vcol1.metric("Frames Analyzed", total_frames)
                    vcol2.metric("Max Faces in Frame", max_faces)
                    vcol3.metric("Total Objects Tracked", total_objects)
                    
                    # Raw JSON Dropdown
                    with st.expander("View Raw JSON Report"):
                        st.json(data)
                        
                else:
                    status.update(label="Pipeline Failed", state="error", expanded=True)
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                status.update(label="Connection Error", state="error", expanded=True)
                st.error("Could not connect to the API. Is your FastAPI server (uvicorn) running on port 8000?")