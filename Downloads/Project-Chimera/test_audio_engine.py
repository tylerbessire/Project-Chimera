#!/usr/bin/env python3
"""
Quick test for the audio engine to validate it produces expected output.
"""

import numpy as np
import pytest
from professional_audio_engine import ProfessionalAudioEngine, AudioBuffer, ProcessingQuality


def test_audio_engine_basic():
    """Test that the audio engine can process basic audio."""
    
    # Create test audio buffers (4-bar loops at 48kHz)
    sample_rate = 48000
    duration = 8.0  # 8 seconds = 4 bars at 120 BPM
    samples = int(duration * sample_rate)
    
    # Generate test signals
    t = np.linspace(0, duration, samples)
    
    # Track A: 440Hz sine wave (A note)
    audio_a_data = np.array([
        np.sin(2 * np.pi * 440 * t),  # Left channel
        np.sin(2 * np.pi * 440 * t)   # Right channel
    ]) * 0.5  # -6dB
    
    # Track B: 880Hz sine wave (A one octave higher)
    audio_b_data = np.array([
        np.sin(2 * np.pi * 880 * t),  # Left channel
        np.sin(2 * np.pi * 880 * t)   # Right channel
    ]) * 0.707  # -3dB
    
    # Create audio buffers
    audio_a = AudioBuffer(
        audio=audio_a_data,
        sample_rate=sample_rate,
        bit_depth=32,
        channels=2,
        duration=duration,
        metadata={'test': 'track_a'}
    )
    
    audio_b = AudioBuffer(
        audio=audio_b_data,
        sample_rate=sample_rate,
        bit_depth=32,
        channels=2,
        duration=duration,
        metadata={'test': 'track_b'}
    )
    
    # Create simple recipe for testing
    test_recipe = {
        'sections': [
            {
                'section_label': 'Intro',
                'duration': 4,
                'description': 'Track A instrumental only'
            },
            {
                'section_label': 'Mix',
                'duration': 4,
                'description': 'Both tracks blended'
            }
        ]
    }
    
    # Initialize engine
    engine = ProfessionalAudioEngine(ProcessingQuality.PROFESSIONAL)
    
    # Test structure conversion
    structure = engine._convert_recipe_to_structure(test_recipe['sections'], audio_a, audio_b)
    
    # Validate structure
    assert len(structure) == 2
    assert structure[0]['source'] == 'a'  # Should detect instrumental
    assert structure[1]['source'] == 'both'  # Should detect blend
    assert structure[0]['duration'] == 4
    assert structure[1]['duration'] == 4
    
    # Test audio processing
    section_audio = engine._process_mashup_section(audio_a, audio_b, structure[0], 0)
    
    # Validate output
    assert section_audio.shape[0] == 2  # Stereo
    assert section_audio.shape[1] > 0   # Has samples
    
    # Check RMS is in expected range
    rms = np.sqrt(np.mean(section_audio**2))
    assert 0.1 < rms < 0.8  # Should be audible but not clipping
    
    print("✅ Audio engine basic test passed!")
    

def test_bpm_detection():
    """Test BPM detection works."""
    engine = ProfessionalAudioEngine()
    
    # Create 120 BPM test signal (simple click track)
    sample_rate = 48000
    duration = 8.0  # 8 seconds
    samples = int(duration * sample_rate)
    
    # Generate click track at 120 BPM (2 beats per second)
    t = np.linspace(0, duration, samples)
    clicks = np.zeros_like(t)
    
    # Add clicks every 0.5 seconds (120 BPM)
    for beat in np.arange(0, duration, 0.5):
        click_start = int(beat * sample_rate)
        click_end = int((beat + 0.1) * sample_rate)
        if click_end < len(clicks):
            clicks[click_start:click_end] = 1.0
    
    # Make stereo
    audio_data = np.array([clicks, clicks])
    
    # Test BPM detection
    detected_bpm = engine._detect_bpm(audio_data)
    
    # Should detect around 120 BPM (allow some tolerance)
    assert 100 < detected_bpm < 140
    
    print(f"✅ BPM detection test passed! Detected: {detected_bpm:.1f} BPM")


if __name__ == "__main__":
    test_audio_engine_basic()
    test_bpm_detection()
    print("🎉 All audio engine tests passed!")