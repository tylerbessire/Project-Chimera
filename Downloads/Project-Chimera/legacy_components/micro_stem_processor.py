import os
import librosa
import numpy as np

class MicroStemProcessor:
    """
    Analyzes audio stems and breaks them down into a meticulous catalog of 
    "molecules" based on onsets, transients, and pitches.
    """
    def __init__(self, stems_directory: str):
        self.stems_dir = stems_directory
        self.stems = {
            "vocals": os.path.join(stems_directory, "vocals.wav"),
            "drums": os.path.join(stems_directory, "drums.wav"),
            "bass": os.path.join(stems_directory, "bass.wav"),
            "other": os.path.join(stems_directory, "other.wav"),
        }

    def get_micro_segment_catalog(self):
        """
        Processes all available stems and returns a comprehensive catalog of audio molecules.
        """
        catalog = {}
        for stem_name, stem_path in self.stems.items():
            if not os.path.exists(stem_path):
                continue
                
            print(f"Performing molecular analysis on {stem_name}...")
            y, sr = librosa.load(stem_path, sr=44100, mono=True)
            
            # Use a more sensitive onset detection
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, units='frames', backtrack=True, pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.2, wait=10
            )
            onset_times_ms = librosa.frames_to_time(onset_frames, sr=sr) * 1000

            stem_molecules = []
            for i in range(len(onset_times_ms)):
                start_ms = onset_times_ms[i]
                end_ms = onset_times_ms[i+1] if i + 1 < len(onset_times_ms) else len(y) / sr * 1000
                
                duration_ms = end_ms - start_ms
                if duration_ms < 50:  # Ignore overly short sounds
                    continue

                # Add more metadata to each molecule for the AI
                y_slice = y[int(start_ms/1000*sr):int(end_ms/1000*sr)]
                rms = np.mean(librosa.feature.rms(y=y_slice))
                pitches, magnitudes = librosa.piptrack(y=y_slice, sr=sr)
                primary_pitch = pitches[magnitudes > np.median(magnitudes)].mean() if np.any(magnitudes) else 0

                molecule_id = f"{os.path.basename(self.stems_dir)}_{stem_name}_{i:04d}"
                stem_molecules.append({
                    "id": molecule_id,
                    "source_stem": stem_name,
                    "start_time_ms": int(start_ms),
                    "end_time_ms": int(end_ms),
                    "duration_ms": int(duration_ms),
                    "rms_energy": float(rms),
                    "primary_pitch_hz": float(primary_pitch) if primary_pitch > 0 else None,
                })
            catalog[stem_name] = stem_molecules
        return catalog
