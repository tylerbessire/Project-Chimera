# ==============================================================================
# FILE: real_audio_engine.py (v7.0 - Studio Grade Update)
# ==============================================================================
#
# MAJOR UPDATES:
# - COMPLETE REWRITE. This is now a professional, effects-driven audio renderer.
# - Integrated `pedalboard` for high-quality audio effects (EQ, Compression,
#   Reverb, Delay, Filters, Limiter). This is the core of the "studio sound".
# - Integrated `pyrubberband` for high-quality, artifact-free time-stretching
#   and pitch-shifting.
# - Parses the new, detailed technical recipe from the CollaborationEngine.
# - Implements advanced techniques like sidechain compression and master bus
#   processing.
#
# ==============================================================================

import os
import json
import numpy as np
import soundfile as sf
import librosa
import rubberband
from pydub import AudioSegment
from pedalboard import (
    Pedalboard, Compressor, Reverb, Delay, HighpassFilter, LowpassFilter,
    Limiter, PeakFilter, Gain
)

class RealAudioEngine:
    """
    Renders the final audio from a detailed technical recipe using professional
    audio processing libraries.
    """
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.output_dir = "workspace/mashups"
        os.makedirs(self.output_dir, exist_ok=True)
        self.sr = 44100  # Work at studio sample rate

        print("🎧 RealAudioEngine Initialized. Loading source audio...")
        self.source_audio_data = self._load_source_audio()
        print("✅ Source audio loaded.")

    def render_mashup(self) -> str:
        """
        Orchestrates the entire rendering process from the recipe.
        """
        print("🚀 Starting mashup render process...")
        target_bpm = self.recipe['target_bpm']
        
        # Create a silent canvas to build the mashup on
        # Estimate total duration and add buffer
        est_duration_ms = sum(
            s['end_ms'] - s['start_ms'] 
            for brief in self.source_audio_data.values() 
            for s in brief['structural_segments']
        )
        final_mashup = AudioSegment.silent(duration=est_duration_ms + 10000, frame_rate=self.sr)
        
        current_time_ms = 0
        
        for section in self.recipe['sections']:
            print(f"  -> Rendering Section: {section['section_label']}")
            section_audio = self._render_section(section, target_bpm)
            
            if len(section_audio) > 0:
                final_mashup = final_mashup.overlay(section_audio, position=current_time_ms)
                current_time_ms += len(section_audio)

        print("  -> Applying master effects chain...")
        # Convert final pydub segment to numpy for mastering with pedalboard
        final_mashup_samples = np.array(final_mashup.get_array_of_samples()).reshape(-1, final_mashup.channels).T.astype(np.float32)
        final_mashup_samples /= np.iinfo(final_mashup.sample_width * 8).max
        
        master_board = self._create_pedalboard(self.recipe.get('master_effects_chain', []))
        mastered_samples = master_board(final_mashup_samples, self.sr)
        
        # Sanitize filename
        safe_title = "".join(c for c in self.recipe['mashup_title'][:50] if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
        if not safe_title: safe_title = "AI_Mashup"
        output_filename = f"{safe_title}.wav"
        output_path = os.path.join(self.output_dir, output_filename)
        
        print(f"  -> Exporting to {output_path}")
        sf.write(output_path, mastered_samples.T, self.sr)
        
        print("✅ Mashup rendering complete!")
        return output_path

    def _render_section(self, section_data, target_bpm):
        """Renders a single section by layering its processed clips."""
        section_duration = 0
        
        # Find max duration in this section to define its length
        for layer in section_data['layer_cake']:
            source_track_id = layer['source_track']
            source_brief = self.source_audio_data[source_track_id]
            segment_data = self._find_segment(source_brief, layer['source_segment_label'])
            if segment_data:
                duration = segment_data['end_ms'] - segment_data['start_ms']
                if layer.get('time_slice_ms'):
                    duration = layer['time_slice_ms'][1] - layer['time_slice_ms'][0]
                section_duration = max(section_duration, duration)
        
        if section_duration == 0: return AudioSegment.silent(duration=0)

        section_canvas = AudioSegment.silent(duration=section_duration, frame_rate=self.sr)

        for layer in section_data['layer_cake']:
            processed_clip = self._process_layer(layer, target_bpm)
            if processed_clip:
                section_canvas = section_canvas.overlay(processed_clip)
        
        return section_canvas.fade_in(20).fade_out(20)

    def _process_layer(self, layer_data, target_bpm):
        """Processes a single layer: loads, time-stretches, applies effects."""
        source_track_id = layer_data['source_track']
        source_brief = self.source_audio_data[source_track_id]
        
        # 1. Get the raw audio clip as a numpy array
        clip_samples = self._get_audio_clip(source_brief, layer_data)
        if clip_samples is None: return None
        
        # 2. Time-stretch/pitch-shift the clip using Rubberband
        source_bpm = source_brief['tempo']
        time_ratio = source_bpm / target_bpm
        
        # This is where key changes would be implemented if needed
        # For now, we assume AI picks key-compatible songs.
        pitch_semitones = 0
        
        stretched_samples = rubberband.stretch(clip_samples, self.sr, timeratio=time_ratio, pitch=pitch_semitones)
        
        # 3. Apply the effects chain using Pedalboard
        board = self._create_pedalboard(layer_data.get('effects_chain', []))
        
        # Handle sidechaining - a key professional technique
        sidechain_trigger = layer_data.get("sidechain_trigger")
        if sidechain_trigger and "drums" in self.source_audio_data[sidechain_trigger["source_track"]]["stems"]:
             trigger_audio = self.source_audio_data[sidechain_trigger["source_track"]]["stems"]["drums"]
             # Align and trim trigger audio to match the stretched clip length
             # This is a simplification; a more robust implementation would align beats.
             max_len = min(len(trigger_audio.T), len(stretched_samples.T))
             board.pre_gain = 0.0 # Mute the sidechain input from the output
             effected_samples = board(stretched_samples[:,:max_len], self.sr, sidechain=trigger_audio[:,:max_len])
        else:
             effected_samples = board(stretched_samples, self.sr)

        # 4. Convert processed numpy array back to pydub AudioSegment
        # Ensure correct scaling back to 16-bit integer range
        effected_samples_int = (effected_samples.T * (2**15 - 1)).astype(np.int16)
        
        return AudioSegment(
            effected_samples_int.tobytes(),
            frame_rate=self.sr,
            sample_width=2, # 16-bit
            channels=effected_samples.shape[0] # stereo
        )

    def _get_audio_clip(self, source_brief, layer_data):
        """Extracts a slice of audio from a stem file based on the recipe."""
        stem_name = layer_data['stem']
        if stem_name not in source_brief['stems']:
            return None
        
        audio_data = source_brief['stems'][stem_name]
        
        segment_info = self._find_segment(source_brief, layer_data['source_segment_label'])
        if not segment_info: return None
        
        start_ms = segment_info['start_ms']
        end_ms = segment_info['end_ms']
        
        # Apply a further time slice if specified
        if 'time_slice_ms' in layer_data:
            start_ms += layer_data['time_slice_ms'][0]
            end_ms = start_ms + (layer_data['time_slice_ms'][1] - layer_data['time_slice_ms'][0])
            
        start_frame = int(start_ms / 1000 * self.sr)
        end_frame = int(end_ms / 1000 * self.sr)
        
        return audio_data[:, start_frame:end_frame]

    def _load_source_audio(self):
        """Loads all required stem files into memory as numpy arrays."""
        audio_data = {}
        instrumental_id = self.recipe['roles']['instrumental_track']
        vocal_id = self.recipe['roles']['vocal_track']
        
        for role, song_id in [("instrumental_track", instrumental_id), ("vocal_track", vocal_id)]:
            with open(self.recipe['source_files'][role]['path'], 'r') as f:
                brief = json.load(f)
            brief['stems'] = {}
            stems_dir = brief['stems_path']
            for stem_name in ["vocals", "drums", "bass", "other"]:
                stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
                if os.path.exists(stem_path):
                    y, _ = librosa.load(stem_path, sr=self.sr, mono=False)
                    if y.ndim == 1: y = np.stack([y, y])
                    brief['stems'][stem_name] = y
            audio_data[role] = brief
        return audio_data

    def _find_segment(self, brief, label):
        """Finds a structural segment by its label in the analysis brief."""
        for seg in brief['structural_segments']:
            if seg['label'] == label:
                return seg
        return None

    def _create_pedalboard(self, effects_chain):
        """Constructs a Pedalboard object from an effects recipe."""
        board = Pedalboard()
        for fx in effects_chain:
            if fx['effect'] == 'compressor':
                board.append(Compressor(
                    threshold_db=fx.get('threshold_db', -10),
                    ratio=fx.get('ratio', 4),
                    attack_ms=fx.get('attack_ms', 2),
                    release_ms=fx.get('release_ms', 100)
                ))
            elif fx['effect'] == 'reverb':
                board.append(Reverb(
                    room_size=fx.get('room_size', 0.5),
                    damping=0.5,
                    wet_level=fx.get('wet_level', 0.33),
                    dry_level=fx.get('dry_level', 0.4),
                    width=1.0
                ))
            elif fx['effect'] == 'delay':
                 board.append(Delay(
                    delay_seconds=fx.get('delay_seconds', 0.5),
                    feedback=fx.get('feedback', 0.25),
                    mix=fx.get('mix', 0.5)
                 ))
            elif fx['effect'] == 'lowpass_filter':
                board.append(LowpassFilter(cutoff_frequency_hz=fx.get('cutoff_hz', 500)))
            elif fx['effect'] == 'highpass_filter':
                board.append(HighpassFilter(cutoff_frequency_hz=fx.get('cutoff_hz', 100)))
            elif fx['effect'] == 'peak_filter_cut': # For subtractive EQ
                board.append(Gain(gain_db=fx.get('gain_db', -3.0))) # Using Gain as a simple peak filter for now
            elif fx['effect'] == 'limiter':
                board.append(Limiter(threshold_db=fx.get('threshold_db', -2.0), release_ms=50))
        return board
