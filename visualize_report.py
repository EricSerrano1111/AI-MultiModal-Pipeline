import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, 'data', 'processed', 'initial_test_report.json')
output_image_path = os.path.join(base_dir, 'data', 'processed', 'initial_test_results_chart.png')

def generate_plot():
    # Load JSON data
    if not os.path.exists(json_path):
        print(f"Could not find JSON report at {json_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract the data
    frames = [item['frame'] for item in data['visual_analysis']]
    faces = [item['faces_detected'] for item in data['visual_analysis']]
    objects = [item['objects_detected'] for item in data['visual_analysis']]
    
    keyword = data['audio_analysis']['detected_keyword']
    confidence = data['audio_analysis']['confidence']

    # Set up the bar chart
    x = np.arange(len(frames))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Faces and Objects side-by-side
    bars1 = ax.bar(x - width/2, faces, width, label='Faces Detected', color='#2ca02c') # Green
    bars2 = ax.bar(x + width/2, objects, width, label='Objects Detected', color='#1f77b4') # Blue

    # Formatting chart for reporting
    ax.set_ylabel('Detection Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Multi-Modal AI Pipeline Detections\nVideo: {data["video_file"]}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(frames, rotation=45, ha='right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Force the Y-axis to use whole numbers (can't detect 1.5 faces)
    ax.yaxis.get_major_locator().set_params(integer=True)

    # Add Audio Pipeline result as an overlay box
    audio_text = f"🎙️ Audio Analysis:\nKeyword: '{keyword.upper()}'\nConfidence: {confidence}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.95, audio_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Save & display
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300) # 300 DPI is standard for print/PDF reports
    print(f"Chart saved to: {output_image_path}")
    
    # Show plot on screen
    plt.show()

if __name__ == "__main__":
    generate_plot()