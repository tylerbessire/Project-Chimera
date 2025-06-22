
"""Professional Audio Engine – patched with real BPM/key alignment (2025‑06‑20)
Requirements:
    pip install librosa pyrubberband soundfile demucs sox
Demucs must be available on $PATH for stem separation.
"""

import os, subprocess, tempfile, pathlib, json, uuid
import numpy as np
import soundfile as sf
import librosa
import pyrubberband as pyrb
from pydub import AudioSegment


class ProfessionalAudioEngine:
    """Render a two‑track mash‑up with proper tempo & key matching."""

    def __init__(self, sr: int = 48000):
        self.target_sr = sr

    # ---------- PUBLIC API ---------- #
    def create_professional_mashup(self, song_a, song_b, analysis, out_path):
        # 1. Split stems so we get clean vocals/inst
        stems_a = self._split_stems(song_a)
        stems_b = self._split_stems(song_b)

        # 2. Detect BPM + key on vocals
        bpm_a = self._detect_bpm(stems_a["vocals"])
        bpm_b = self._detect_bpm(stems_b["vocals"])

        key_a = self._detect_key(stems_a["vocals"])
        key_b = self._detect_key(stems_b["vocals"])

        print(f"Detected BPM: A={bpm_a:.1f}, B={bpm_b:.1f}; Keys: A={key_a}, B={key_b}")

        # 3. Choose mash tempo (mid‑point) + common key (use key B if a semitone or less)
        target_bpm = (bpm_a + bpm_b) / 2
        target_key = key_b if abs(key_a - key_b) <= 1 else key_a

        # 4. Time‑stretch & pitch‑shift both songs
        a_proc = self._process_track(stems_a, bpm_a, target_bpm, key_a, target_key)
        b_proc = self._process_track(stems_b, bpm_b, target_bpm, key_b, target_key)

        # 5. Build arrangement – simple AB overlay for demo
        bed = a_proc["instrumental"].overlay(b_proc["instrumental"], position=0, gain_during_overlay=-2)
        vocals = a_proc["vocals"].overlay(b_proc["vocals"], position=0)

        mix = bed.overlay(vocals)
        mix = mix.normalize()

        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        mix.export(out_path, format="wav")
        return out_path

    # ---------- UTILITIES ---------- #
    def _split_stems(self, wav_path):
        """Return dict with 'vocals' and 'instrumental' AudioSegments."""
        out_dir = pathlib.Path(tempfile.gettempdir()) / "demucs" / uuid.uuid4().hex
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["demucs", "--two-stems", "vocals", wav_path, "-o", str(out_dir)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        stem_folder = next(out_dir.glob("*/"))  # first subfolder
        vocals_path = stem_folder / "vocals.wav"
        other_path = stem_folder / "no_vocals.wav"

        return {
            "vocals": AudioSegment.from_wav(vocals_path),
            "instrumental": AudioSegment.from_wav(other_path)
        }

    def _detect_bpm(self, seg: AudioSegment):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg.export(tmp.name, format="wav")
            y, sr = librosa.load(tmp.name, mono=True)
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        return bpm

    def _detect_key(self, seg: AudioSegment):
        # crude: estimate pitch histogram, choose strongest semitone (0=C)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg.export(tmp.name, format="wav")
            y, sr = librosa.load(tmp.name, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        return int(np.argmax(chroma.mean(axis=1)))

    def _process_track(self, stems, orig_bpm, tgt_bpm, orig_key, tgt_key):
        ratio = tgt_bpm / orig_bpm
        semitone_shift = tgt_key - orig_key

        out = {}
        for name, seg in stems.items():
            # write to tmp
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                seg.export(tmp.name, format="wav")
                y, sr = librosa.load(tmp.name, sr=None)
            # stretch
            y = pyrb.time_stretch(y, sr, ratio)
            # pitch
            if semitone_shift:
                y = pyrb.pitch_shift(y, sr, n_steps=semitone_shift)
            # back to AudioSegment
            tmp_out = tmp.name.replace(".wav", "_proc.wav")
            sf.write(tmp_out, y, sr)
            out[name] = AudioSegment.from_wav(tmp_out)
        return out
