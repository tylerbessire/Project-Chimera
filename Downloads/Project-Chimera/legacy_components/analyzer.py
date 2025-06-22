# ==============================================================================
# FILE: analyzer.py (v7.3 - Final Demucs 4-Stem Model)
# ==============================================================================
#
# MAJOR UPDATES:
# - Upgraded Demucs call to use the full 4-stem model ('htdemucs_ft'). This
#   provides separate, high-quality stems for vocals, bass, drums, and other,
#   which is a significant quality improvement for the final mashup.
# - Refined the subprocess call for maximum stability across different systems.
# - Kept the robust fallback system: Demucs -> Spleeter -> HPSS.
#
# ==============================================================================

import os
import json
import subprocess
import warnings
import librosa
import numpy as np
import soundfile as sf
import shutil

# --- Make Demucs and Spleeter Optional ---
try:
    import demucs.separate
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    from spleeter.separator import Separator
    SPLEETER_AVAILABLE = True
except ImportError:
    SPLEETER_AVAILABLE = False
# ----------------------------------------

# Suppress known library warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AudioAnalyzer:
    """
    Performs deep analysis. Prioritizes Demucs (4-stem model) for stem separation,
    with Spleeter and HPSS as fallbacks.
    """
    def __init__(self, audio_path: str):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        self.audio_path = os.path.abspath(audio_path) # Use absolute path for robustness
        self.filename = os.path.basename(audio_path)
        
        self.workspace_dir = "workspace"
        self.cache_dir = os.path.join(self.workspace_dir, "analysis_cache")
        self.stems_dir = os.path.join(self.workspace_dir, "stems", os.path.splitext(self.filename)[0])
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.stems_dir, exist_ok=True)

        self.cache_path = os.path.join(self.cache_dir, f"{self.filename}.json")
        
        self.sr = 44100
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr, mono=False)
        if self.y.ndim == 1: self.y = np.stack([self.y, self.y])
        self.y_mono = librosa.to_mono(self.y)

        self.separator_type = None
        if DEMUCS_AVAILABLE:
            self.separator_type = 'demucs'
            print("✅ Demucs found. Using state-of-the-art 4-stem separation.")
        elif SPLEETER_AVAILABLE:
            self.separator_type = 'spleeter'
            print("✅ Spleeter found. Using for high-quality stem separation.")
        else:
            self.separator_type = 'hpss'
            print("⚠️ Demucs/Spleeter not found. Falling back to basic separation.")

    def full_analysis(self, force_reanalyze=False):
        if not force_reanalyze and os.path.exists(self.cache_path):
            print(f"✅ Loading analysis from cache for {self.filename}")
            with open(self.cache_path, 'r') as f: return json.load(f)

        print(f"Performing full analysis for {self.filename}...")
        
        stem_paths = self._separate_stems()
        tempo, beats = self._extract_rhythm()
        key, camelot_key = self._extract_harmony()
        duration_ms = librosa.get_duration(y=self.y, sr=self.sr) * 1000
        structural_segments = self._extract_structure(beats)

        analysis_results = {
            "title": os.path.splitext(self.filename)[0],
            "duration_ms": int(duration_ms),
            "tempo": float(tempo), "estimated_key": key, "camelot_key": camelot_key,
            "beat_map_ms": (librosa.frames_to_time(beats, sr=self.sr) * 1000).astype(int).tolist(),
            "structural_segments": structural_segments, "source_file_path": self.audio_path,
            "stems_path": self.stems_dir, "stem_files": stem_paths
        }

        with open(self.cache_path, 'w') as f:
            json.dump(analysis_results, f, indent=4)
            
        print(f"✅ Analysis complete for {self.filename}")
        return analysis_results

    def _separate_stems(self):
        if self.separator_type == 'demucs':
            return self._separate_stems_demucs()
        elif self.separator_type == 'spleeter':
            try:
                separator = Separator('spleeter:4stems')
                return self._separate_stems_spleeter(separator)
            except Exception:
                return self._separate_stems_hpss()
        else:
            return self._separate_stems_hpss()
    
    def _separate_stems_demucs(self):
        """Uses Demucs' 4-stem model for state-of-the-art source separation."""
        print(f"🔪 Separating stems for {self.filename} using Demucs 4-stem model...")
        try:
            out_dir_root = os.path.join(self.workspace_dir, "demucs_output")
            # Using a list of args for subprocess is more robust than a single string
            cmd = [
                "python3", "-m", "demucs.separate",
                "-o", out_dir_root,
                "-n", "htdemucs_ft", # 4-stem model
                self.audio_path
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            base_filename_no_ext = os.path.splitext(self.filename)[0]
            demucs_output_folder = os.path.join(out_dir_root, "htdemucs_ft", base_filename_no_ext)
            
            stem_paths = {}
            for stem_name in ["vocals", "drums", "bass", "other"]:
                source_path = os.path.join(demucs_output_folder, f"{stem_name}.wav")
                if os.path.exists(source_path):
                    dest_path = os.path.join(self.stems_dir, f"{stem_name}.wav")
                    shutil.move(source_path, dest_path)
                    stem_paths[stem_name] = dest_path

            # Clean up the temporary demucs output directory
            shutil.rmtree(out_dir_root)

            if "vocals" not in stem_paths:
                raise RuntimeError("Demucs did not produce the expected stem files.")

            print("✅ Superior-quality 4-stems separated via Demucs.")
            return stem_paths
        except Exception as e:
            print(f"❌ Demucs stem separation failed: {e}")
            # Check if stderr is available from the exception
            if hasattr(e, 'stderr'):
                print(f"   Demucs Error Output: {e.stderr}")
            print("   Falling back to basic separation for this track.")
            return self._separate_stems_hpss()

    def _separate_stems_spleeter(self, separator):
        """Uses Spleeter for high-quality source separation."""
        print(f"🔪 Separating stems for {self.filename} using Spleeter...")
        try:
            separator.separate_to_file(self.audio_path, self.stems_dir)
            
            output_folder = os.path.join(self.stems_dir, os.path.splitext(self.filename)[0])
            stem_paths = {}
            for stem_name in ["vocals", "drums", "bass", "other"]:
                original_path = os.path.join(output_folder, f"{stem_name}.wav")
                if os.path.exists(original_path):
                    dest_path = os.path.join(self.stems_dir, f"{stem_name}.wav")
                    shutil.move(original_path, dest_path)
                    stem_paths[stem_name] = dest_path
            
            if os.path.exists(output_folder): shutil.rmtree(output_folder)
            print("✅ High-quality stems separated via Spleeter.")
            return stem_paths
        except Exception as e:
            print(f"❌ Spleeter stem separation failed unexpectedly: {e}")
            return self._separate_stems_hpss()

    def _separate_stems_hpss(self):
        """Basic fallback separation using librosa's HPSS."""
        print(f"🔪 Performing basic separation for {self.filename} using HPSS...")
        y_harmonic, y_percussive = librosa.effects.hpss(self.y_mono)
        stem_paths = {}
        for name, data in [("vocals", np.stack([y_harmonic, y_harmonic])), 
                             ("drums", np.stack([y_percussive, y_percussive]))]:
            path = os.path.join(self.stems_dir, f"{name}.wav")
            sf.write(path, data.T, self.sr)
            stem_paths[name] = path
        
        for name in ["bass", "other"]:
            path = os.path.join(self.stems_dir, f"{name}.wav")
            sf.write(path, np.zeros_like(self.y).T, self.sr)
            stem_paths[name] = path

        print("✅ Basic stems separated. Note: Quality is much lower than Demucs.")
        return stem_paths

    def _extract_rhythm(self):
        tempo, beats = librosa.beat.beat_track(y=self.y_mono, sr=self.sr)
        return tempo, beats

    def _extract_harmony(self):
        chroma = librosa.feature.chroma_cqt(y=self.y_mono, sr=self.sr)
        maj_p = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        min_p = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        corrs = [np.corrcoef(np.sum(chroma,1),np.roll(p,i))[0,1] for i in range(12) for p in (maj_p,min_p)]
        idx = np.argmax(corrs)
        key = notes[idx//2] + (' Major' if idx%2==0 else ' Minor')
        return key, self._to_camelot(key)

    def _to_camelot(self, key_str: str):
        note, mode = key_str.split(' ')
        map_maj={'B':'1B','F#':'2B','C#':'2B','D♭':'3B','A♭':'4B','E♭':'5B','B♭':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
        map_min={'A♭':'1A','G#':'1A','E♭':'2A','D#':'2A','B♭':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','D♭':'12A','C#':'12A'}
        return map_maj.get(note) if mode == 'Major' else map_min.get(note, "Unknown")

    def _extract_structure(self, beats):
        chroma = librosa.util.sync(librosa.feature.chroma_cqt(y=self.y_mono,sr=self.sr),beats)
        R=librosa.segment.recurrence_matrix(chroma,width=9,mode='affinity',sym=True)
        bounds=librosa.segment.agglomerative(librosa.segment.recurrence_to_lag(R,pad=False),k=10)
        times=librosa.frames_to_time(beats[bounds],sr=self.sr)
        segs, start_time = [], 0.0
        labels = ["Intro","Verse","Chorus","Verse","Chorus","Bridge","Chorus","Outro"]
        for i, end_time in enumerate(times):
            segs.append({"label":labels[i%len(labels)], "start_ms":int(start_time*1000), "end_ms":int(end_time*1000)})
            start_time = end_time
        end = librosa.get_duration(y=self.y_mono,sr=self.sr)
        if start_time<end: segs.append({"label":labels[len(segs)%len(labels)], "start_ms":int(start_time*1000), "end_ms":int(end*1000)})
        return segs
