"""
Create demo audio file for frontend testing
"""
import numpy as np
import soundfile as sf
import os

def create_demo_audio():
    # Create a simple demo audio file
    sample_rate = 44100
    duration = 30  # 30 seconds
    
    # Generate a simple melody
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a basic melody with some rhythm
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00]  # C-D-E-F-G
    audio = np.zeros(len(t))
    
    for i, freq in enumerate(frequencies):
        start_time = i * duration / len(frequencies)
        end_time = (i + 1) * duration / len(frequencies)
        
        section_mask = (t >= start_time) & (t < end_time)
        audio[section_mask] = 0.3 * np.sin(2 * np.pi * freq * t[section_mask])
    
    # Add some basic rhythm
    beat_interval = 0.5  # seconds
    for beat_time in np.arange(0, duration, beat_interval):
        start_idx = int(beat_time * sample_rate)
        end_idx = min(start_idx + int(0.1 * sample_rate), len(audio))
        audio[start_idx:end_idx] *= 1.5
    
    # Create stereo version
    stereo_audio = np.stack([audio, audio])
    
    # Save to workspace
    output_path = "workspace/mashups/demo_mashup.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, stereo_audio.T, sample_rate)
    
    print(f"Demo audio created: {output_path}")

if __name__ == "__main__":
    create_demo_audio()