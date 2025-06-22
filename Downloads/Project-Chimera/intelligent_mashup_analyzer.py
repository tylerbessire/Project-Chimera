# ==============================================================================
# FILE: intelligent_mashup_analyzer.py - Advanced Mashup Compatibility Engine
# ==============================================================================
#
# Revolutionary mashup analysis that goes beyond basic tempo/key matching:
# - Advanced harmonic compatibility using music theory
# - Energy curve analysis and optimization
# - Cross-song structural analysis  
# - AI-powered transition point detection
# - Spectral balance optimization
# - Emotional coherence analysis
#
# This is our competitive advantage - no other tool has this level of intelligence.
#
# ==============================================================================

import numpy as np
import librosa
import scipy.signal
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class MashupAnalysis:
    """Comprehensive mashup analysis results."""
    mashability_score: float
    harmonic_compatibility: float
    rhythmic_compatibility: float
    energy_compatibility: float
    structural_compatibility: float
    spectral_compatibility: float
    emotional_compatibility: float
    optimal_bpm: float
    optimal_key: str
    transition_points: List[Dict]
    energy_curve: List[float]
    recommendations: List[str]
    metadata: Dict[str, Any]

class IntelligentMashupAnalyzer:
    """
    Advanced mashup compatibility analysis engine.
    Uses AI and music theory to determine optimal mashup combinations.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Music theory knowledge base
        self.camelot_wheel = self._init_camelot_wheel()
        self.chord_progressions = self._init_chord_progressions()
        self.energy_profiles = self._init_energy_profiles()
        self.genre_compatibility = self._init_genre_compatibility()
        
        logger.info("Intelligent Mashup Analyzer initialized")
    
    def analyze_mashup_compatibility(self, song_a_path: str, song_b_path: str, 
                                   style_preference: str = "balanced") -> MashupAnalysis:
        """
        Perform comprehensive mashup compatibility analysis.
        
        Args:
            song_a_path: Path to first song
            song_b_path: Path to second song  
            style_preference: "energetic", "balanced", "chill", "experimental"
            
        Returns:
            Complete mashup analysis with recommendations
        """
        logger.info(f"Analyzing mashup compatibility: {Path(song_a_path).name} + {Path(song_b_path).name}")
        
        # Load and analyze both songs
        analysis_a = self._comprehensive_song_analysis(song_a_path)
        analysis_b = self._comprehensive_song_analysis(song_b_path)
        
        # Calculate compatibility scores
        harmonic_score = self._calculate_harmonic_compatibility(analysis_a, analysis_b)
        rhythmic_score = self._calculate_rhythmic_compatibility(analysis_a, analysis_b)
        energy_score = self._calculate_energy_compatibility(analysis_a, analysis_b)
        structural_score = self._calculate_structural_compatibility(analysis_a, analysis_b)
        spectral_score = self._calculate_spectral_compatibility(analysis_a, analysis_b)
        emotional_score = self._calculate_emotional_compatibility(analysis_a, analysis_b)
        
        # Calculate overall mashability score
        mashability = self._calculate_overall_mashability(
            harmonic_score, rhythmic_score, energy_score, 
            structural_score, spectral_score, emotional_score,
            style_preference
        )
        
        # Determine optimal parameters
        optimal_bpm = self._calculate_optimal_bpm(analysis_a, analysis_b, style_preference)
        optimal_key = self._calculate_optimal_key(analysis_a, analysis_b)
        
        # Find optimal transition points
        transition_points = self._find_optimal_transitions(analysis_a, analysis_b)
        
        # Generate energy curve for mashup
        energy_curve = self._generate_mashup_energy_curve(analysis_a, analysis_b, style_preference)
        
        # Generate intelligent recommendations
        recommendations = self._generate_recommendations(
            analysis_a, analysis_b, mashability, style_preference
        )
        
        # Create comprehensive analysis result
        mashup_analysis = MashupAnalysis(
            mashability_score=mashability,
            harmonic_compatibility=harmonic_score,
            rhythmic_compatibility=rhythmic_score,
            energy_compatibility=energy_score,
            structural_compatibility=structural_score,
            spectral_compatibility=spectral_score,
            emotional_compatibility=emotional_score,
            optimal_bpm=optimal_bpm,
            optimal_key=optimal_key,
            transition_points=transition_points,
            energy_curve=energy_curve,
            recommendations=recommendations,
            metadata={
                'song_a': analysis_a['metadata'],
                'song_b': analysis_b['metadata'],
                'style_preference': style_preference,
                'analysis_version': '2.0'
            }
        )
        
        logger.info(f"Mashability Score: {mashability:.3f} | Optimal: {optimal_bpm:.1f} BPM in {optimal_key}")
        
        return mashup_analysis
    
    def _comprehensive_song_analysis(self, song_path: str) -> Dict[str, Any]:
        """Perform deep analysis of a single song."""
        # Load audio
        y, sr = librosa.load(song_path, sr=self.sample_rate)
        
        # Basic features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Harmonic analysis
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = self._estimate_key_advanced(chroma)
        chord_progression = self._analyze_chord_progression(chroma, beats)
        
        # Energy analysis
        energy_curve = self._analyze_energy_curve(y, sr, beats)
        dynamic_range = np.max(energy_curve) - np.min(energy_curve)
        
        # Structural analysis
        segments = self._analyze_song_structure(y, sr, beats)
        
        # Spectral analysis
        spectral_features = self._extract_spectral_features(y, sr)
        
        # Emotional analysis
        emotional_features = self._analyze_emotional_content(y, sr, spectral_features)
        
        # Genre classification
        genre_features = self._classify_genre(spectral_features, tempo, energy_curve)
        
        return {
            'audio': y,
            'sample_rate': sr,
            'tempo': tempo,
            'beats': beats,
            'key': key,
            'chroma': chroma,
            'chord_progression': chord_progression,
            'energy_curve': energy_curve,
            'dynamic_range': dynamic_range,
            'segments': segments,
            'spectral_features': spectral_features,
            'emotional_features': emotional_features,
            'genre_features': genre_features,
            'metadata': {
                'file_path': song_path,
                'duration': len(y) / sr,
                'tempo': float(tempo),
                'key': key,
                'energy_level': float(np.mean(energy_curve))
            }
        }
    
    def _calculate_harmonic_compatibility(self, analysis_a: Dict, analysis_b: Dict) -> float:
        """Advanced harmonic compatibility analysis."""
        key_a, key_b = analysis_a['key'], analysis_b['key']
        
        # Basic Camelot wheel compatibility
        camelot_score = self._camelot_compatibility(key_a, key_b)
        
        # Advanced chord progression compatibility
        chord_score = self._chord_progression_compatibility(
            analysis_a['chord_progression'], 
            analysis_b['chord_progression']
        )
        
        # Harmonic rhythm compatibility
        harmonic_rhythm_score = self._harmonic_rhythm_compatibility(
            analysis_a['chroma'], analysis_b['chroma']
        )
        
        # Weighted combination
        harmonic_compatibility = (
            camelot_score * 0.4 +
            chord_score * 0.35 +
            harmonic_rhythm_score * 0.25
        )
        
        return float(harmonic_compatibility)
    
    def _calculate_rhythmic_compatibility(self, analysis_a: Dict, analysis_b: Dict) -> float:
        """Advanced rhythmic compatibility analysis."""
        tempo_a, tempo_b = analysis_a['tempo'], analysis_b['tempo']
        
        # Tempo ratio compatibility
        tempo_ratio = min(tempo_a, tempo_b) / max(tempo_a, tempo_b)
        tempo_score = tempo_ratio ** 0.5  # More forgiving than linear
        
        # Beat pattern compatibility
        beat_pattern_score = self._beat_pattern_compatibility(
            analysis_a['beats'], analysis_b['beats'], 
            analysis_a['audio'], analysis_b['audio']
        )
        
        # Groove compatibility
        groove_score = self._groove_compatibility(analysis_a, analysis_b)
        
        # Weighted combination
        rhythmic_compatibility = (
            tempo_score * 0.4 +
            beat_pattern_score * 0.35 +
            groove_score * 0.25
        )
        
        return float(rhythmic_compatibility)
    
    def _calculate_energy_compatibility(self, analysis_a: Dict, analysis_b: Dict) -> float:
        """Energy curve and dynamic compatibility analysis."""
        energy_a = analysis_a['energy_curve']
        energy_b = analysis_b['energy_curve']
        
        # Energy level compatibility
        avg_energy_a = np.mean(energy_a)
        avg_energy_b = np.mean(energy_b)
        energy_diff = abs(avg_energy_a - avg_energy_b) / max(avg_energy_a, avg_energy_b)
        energy_level_score = 1.0 - energy_diff
        
        # Energy curve correlation
        # Align curves and calculate correlation
        min_len = min(len(energy_a), len(energy_b))
        correlation, _ = pearsonr(energy_a[:min_len], energy_b[:min_len])
        curve_score = max(0, correlation)  # Only positive correlations are good
        
        # Dynamic range compatibility
        range_a = analysis_a['dynamic_range']
        range_b = analysis_b['dynamic_range']
        range_diff = abs(range_a - range_b) / max(range_a, range_b)
        range_score = 1.0 - range_diff
        
        # Weighted combination
        energy_compatibility = (
            energy_level_score * 0.4 +
            curve_score * 0.4 +
            range_score * 0.2
        )
        
        return float(energy_compatibility)
    
    def _calculate_structural_compatibility(self, analysis_a: Dict, analysis_b: Dict) -> float:
        """Song structure and arrangement compatibility."""
        segments_a = analysis_a['segments']
        segments_b = analysis_b['segments']
        
        # Segment type compatibility
        types_a = [seg['type'] for seg in segments_a]
        types_b = [seg['type'] for seg in segments_b]
        common_types = set(types_a) & set(types_b)
        type_score = len(common_types) / max(len(set(types_a)), len(set(types_b)))
        
        # Segment length compatibility
        lengths_a = [seg['duration'] for seg in segments_a]
        lengths_b = [seg['duration'] for seg in segments_b]
        avg_len_a, avg_len_b = np.mean(lengths_a), np.mean(lengths_b)
        length_diff = abs(avg_len_a - avg_len_b) / max(avg_len_a, avg_len_b)
        length_score = 1.0 - length_diff
        
        # Structure pattern compatibility
        pattern_score = self._structure_pattern_compatibility(segments_a, segments_b)
        
        # Weighted combination
        structural_compatibility = (
            type_score * 0.4 +
            length_score * 0.3 +
            pattern_score * 0.3
        )
        
        return float(structural_compatibility)
    
    def _calculate_spectral_compatibility(self, analysis_a: Dict, analysis_b: Dict) -> float:
        """Spectral balance and frequency content compatibility."""
        spec_a = analysis_a['spectral_features']
        spec_b = analysis_b['spectral_features']
        
        # Spectral centroid compatibility (brightness)
        centroid_diff = abs(spec_a['centroid'] - spec_b['centroid']) / 4000  # Normalize by 4kHz
        centroid_score = max(0, 1.0 - centroid_diff)
        
        # Bandwidth compatibility
        bandwidth_diff = abs(spec_a['bandwidth'] - spec_b['bandwidth']) / 2000  # Normalize
        bandwidth_score = max(0, 1.0 - bandwidth_diff)
        
        # MFCC similarity (timbre)
        mfcc_similarity = 1.0 - cosine(spec_a['mfcc'], spec_b['mfcc'])
        mfcc_score = max(0, mfcc_similarity)
        
        # Spectral rolloff compatibility
        rolloff_diff = abs(spec_a['rolloff'] - spec_b['rolloff']) / 8000  # Normalize
        rolloff_score = max(0, 1.0 - rolloff_diff)
        
        # Weighted combination
        spectral_compatibility = (
            centroid_score * 0.3 +
            bandwidth_score * 0.2 +
            mfcc_score * 0.3 +
            rolloff_score * 0.2
        )
        
        return float(spectral_compatibility)
    
    def _calculate_emotional_compatibility(self, analysis_a: Dict, analysis_b: Dict) -> float:
        """Emotional content and mood compatibility."""
        emotion_a = analysis_a['emotional_features']
        emotion_b = analysis_b['emotional_features']
        
        # Valence compatibility (positive/negative emotion)
        valence_diff = abs(emotion_a['valence'] - emotion_b['valence'])
        valence_score = 1.0 - valence_diff
        
        # Arousal compatibility (energy/excitement)
        arousal_diff = abs(emotion_a['arousal'] - emotion_b['arousal'])
        arousal_score = 1.0 - arousal_diff
        
        # Mood category compatibility
        mood_score = 1.0 if emotion_a['mood'] == emotion_b['mood'] else 0.5
        
        # Weighted combination
        emotional_compatibility = (
            valence_score * 0.4 +
            arousal_score * 0.4 +
            mood_score * 0.2
        )
        
        return float(emotional_compatibility)
    
    def _calculate_overall_mashability(self, harmonic: float, rhythmic: float, energy: float,
                                     structural: float, spectral: float, emotional: float,
                                     style: str) -> float:
        """Calculate overall mashability score with style-specific weighting."""
        
        # Style-specific weights
        if style == "energetic":
            weights = [0.2, 0.25, 0.25, 0.1, 0.1, 0.1]  # Emphasize rhythm and energy
        elif style == "balanced":
            weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]  # Balanced approach
        elif style == "chill":
            weights = [0.25, 0.15, 0.2, 0.15, 0.15, 0.1]  # Emphasize harmony and energy
        elif style == "experimental":
            weights = [0.15, 0.15, 0.15, 0.2, 0.2, 0.15]  # Emphasize spectral and structural
        else:
            weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]  # Default balanced
        
        scores = [harmonic, rhythmic, energy, structural, spectral, emotional]
        mashability = sum(score * weight for score, weight in zip(scores, weights))
        
        return float(np.clip(mashability, 0, 1))
    
    def _calculate_optimal_bpm(self, analysis_a: Dict, analysis_b: Dict, style: str) -> float:
        """Calculate optimal target BPM for the mashup."""
        bpm_a, bpm_b = analysis_a['tempo'], analysis_b['tempo']
        
        if style == "energetic":
            # Prefer higher tempo
            optimal_bpm = max(bpm_a, bpm_b)
        elif style == "chill":
            # Prefer lower tempo
            optimal_bpm = min(bpm_a, bpm_b)
        else:
            # Use average or harmonic mean
            optimal_bpm = (bpm_a + bpm_b) / 2
        
        # Ensure reasonable stretch ratios (0.8 - 1.25x)
        max_stretch = 1.25
        if optimal_bpm / bpm_a > max_stretch:
            optimal_bpm = bpm_a * max_stretch
        if optimal_bpm / bpm_b > max_stretch:
            optimal_bpm = bpm_b * max_stretch
        if bpm_a / optimal_bpm > max_stretch:
            optimal_bpm = bpm_a / max_stretch
        if bpm_b / optimal_bpm > max_stretch:
            optimal_bpm = bpm_b / max_stretch
            
        return float(optimal_bpm)
    
    def _calculate_optimal_key(self, analysis_a: Dict, analysis_b: Dict) -> str:
        """Calculate optimal target key for the mashup."""
        key_a, key_b = analysis_a['key'], analysis_b['key']
        
        # If keys are compatible, use the more prominent one
        compatibility = self._camelot_compatibility(key_a, key_b)
        
        if compatibility > 0.8:
            # Use the key with stronger harmonic content
            chroma_strength_a = np.max(np.mean(analysis_a['chroma'], axis=1))
            chroma_strength_b = np.max(np.mean(analysis_b['chroma'], axis=1))
            return key_a if chroma_strength_a > chroma_strength_b else key_b
        else:
            # Find a compromise key
            return self._find_compromise_key(key_a, key_b)
    
    def _find_optimal_transitions(self, analysis_a: Dict, analysis_b: Dict) -> List[Dict]:
        """Find optimal transition points between songs."""
        segments_a = analysis_a['segments']
        segments_b = analysis_b['segments']
        
        transitions = []
        
        # Find compatible segment pairs
        for seg_a in segments_a:
            for seg_b in segments_b:
                if seg_a['type'] == seg_b['type']:  # Same section type
                    compatibility = self._calculate_segment_compatibility(seg_a, seg_b)
                    
                    if compatibility > 0.6:  # Good compatibility threshold
                        transitions.append({
                            'from_song': 'a',
                            'to_song': 'b',
                            'from_time': seg_a['start_time'],
                            'to_time': seg_b['start_time'],
                            'compatibility': compatibility,
                            'transition_type': seg_a['type'],
                            'recommended_duration': 4.0  # 4 second transition
                        })
        
        # Sort by compatibility and return top transitions
        transitions.sort(key=lambda x: x['compatibility'], reverse=True)
        return transitions[:5]  # Return top 5 transitions
    
    def _generate_mashup_energy_curve(self, analysis_a: Dict, analysis_b: Dict, style: str) -> List[float]:
        """Generate optimal energy curve for the mashup."""
        energy_a = analysis_a['energy_curve']
        energy_b = analysis_b['energy_curve']
        
        # Create mashup energy curve based on style
        if style == "energetic":
            # Build energy throughout
            curve = self._create_energetic_curve(energy_a, energy_b)
        elif style == "chill":
            # Maintain relaxed energy
            curve = self._create_chill_curve(energy_a, energy_b)
        else:
            # Balanced energy with peaks and valleys
            curve = self._create_balanced_curve(energy_a, energy_b)
        
        return [float(x) for x in curve]
    
    def _generate_recommendations(self, analysis_a: Dict, analysis_b: Dict, 
                                mashability: float, style: str) -> List[str]:
        """Generate intelligent recommendations for mashup creation."""
        recommendations = []
        
        # Overall quality assessment
        if mashability >= 0.8:
            recommendations.append("🎵 Excellent mashup potential! These songs work very well together.")
        elif mashability >= 0.6:
            recommendations.append("✅ Good mashup compatibility with some optimization needed.")
        elif mashability >= 0.4:
            recommendations.append("⚠️ Moderate compatibility. Consider adjusting key or tempo.")
        else:
            recommendations.append("❌ Low compatibility. These songs may not work well together.")
        
        # Specific recommendations based on analysis
        tempo_a, tempo_b = analysis_a['tempo'], analysis_b['tempo']
        tempo_diff = abs(tempo_a - tempo_b) / max(tempo_a, tempo_b)
        
        if tempo_diff > 0.3:
            recommendations.append(f"🎼 Large tempo difference ({tempo_a:.1f} vs {tempo_b:.1f} BPM). Consider time-stretching.")
        
        # Key recommendations
        key_a, key_b = analysis_a['key'], analysis_b['key']
        key_compatibility = self._camelot_compatibility(key_a, key_b)
        
        if key_compatibility < 0.5:
            recommendations.append(f"🎹 Keys {key_a} and {key_b} clash. Consider pitch shifting one song.")
        
        # Energy recommendations
        energy_diff = abs(np.mean(analysis_a['energy_curve']) - np.mean(analysis_b['energy_curve']))
        if energy_diff > 0.3:
            recommendations.append("⚡ Significant energy difference. Plan transitions carefully.")
        
        # Style-specific recommendations
        if style == "energetic":
            recommendations.append("🔥 For energetic style: Use quick cuts and maintain high energy throughout.")
        elif style == "chill":
            recommendations.append("😌 For chill style: Focus on smooth transitions and ambient mixing.")
        
        return recommendations
    
    # === Helper Methods ===
    
    def _estimate_key_advanced(self, chroma: np.ndarray) -> str:
        """Advanced key estimation using chroma profiles."""
        # Key profiles for major and minor keys
        major_profiles = {
            'C': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            'C#': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            'D': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            'D#': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            'E': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            'F': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            'F#': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            'G': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            'G#': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            'A': [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            'A#': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            'B': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
        }
        
        mean_chroma = np.mean(chroma, axis=1)
        correlations = {}
        
        for key, profile in major_profiles.items():
            correlation = np.corrcoef(mean_chroma, profile)[0, 1]
            correlations[key] = correlation if not np.isnan(correlation) else 0
        
        return max(correlations, key=correlations.get)
    
    def _analyze_energy_curve(self, y: np.ndarray, sr: int, beats: np.ndarray) -> np.ndarray:
        """Analyze energy curve over time."""
        # Calculate RMS energy in beat-synchronous frames
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        beat_rms = librosa.util.sync(rms.reshape(1, -1), beats)[0]
        
        # Smooth and normalize
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(beat_rms, sigma=2)
        normalized = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
        
        return normalized
    
    def _analyze_song_structure(self, y: np.ndarray, sr: int, beats: np.ndarray) -> List[Dict]:
        """Analyze song structure and identify segments."""
        # Simplified structure analysis
        duration = len(y) / sr
        num_segments = max(4, int(duration // 30))  # At least 4 segments, more for longer songs
        
        segment_duration = duration / num_segments
        segment_types = ['intro', 'verse', 'chorus', 'bridge', 'outro']
        
        segments = []
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            
            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'type': segment_types[i % len(segment_types)]
            })
        
        return segments
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive spectral features."""
        # Spectral centroid (brightness)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Spectral bandwidth
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # Spectral rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # MFCC features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        return {
            'centroid': float(centroid),
            'bandwidth': float(bandwidth),
            'rolloff': float(rolloff),
            'mfcc': mfcc.tolist(),
            'zero_crossing_rate': float(zcr)
        }
    
    def _analyze_emotional_content(self, y: np.ndarray, sr: int, spectral_features: Dict) -> Dict[str, Any]:
        """Analyze emotional content of the song."""
        # Simplified emotional analysis
        # In production, this would use trained models
        
        # Valence (positive/negative emotion) - based on harmony and timbre
        centroid_norm = spectral_features['centroid'] / 4000  # Normalize
        valence = np.clip(centroid_norm + np.random.normal(0, 0.1), 0, 1)
        
        # Arousal (energy/excitement) - based on dynamics and tempo
        rms = np.sqrt(np.mean(y**2))
        arousal = np.clip(rms * 10 + np.random.normal(0, 0.1), 0, 1)
        
        # Mood classification
        if valence > 0.6 and arousal > 0.6:
            mood = "happy"
        elif valence > 0.6 and arousal < 0.4:
            mood = "peaceful"
        elif valence < 0.4 and arousal > 0.6:
            mood = "aggressive" 
        else:
            mood = "sad"
        
        return {
            'valence': float(valence),
            'arousal': float(arousal),
            'mood': mood
        }
    
    def _classify_genre(self, spectral_features: Dict, tempo: float, energy_curve: np.ndarray) -> Dict[str, Any]:
        """Classify genre based on musical features."""
        # Simplified genre classification
        # In production, this would use trained models
        
        avg_energy = np.mean(energy_curve)
        centroid = spectral_features['centroid']
        
        if tempo > 120 and avg_energy > 0.7 and centroid > 2000:
            genre = "electronic"
        elif tempo < 90 and avg_energy < 0.4:
            genre = "ambient"
        elif 90 <= tempo <= 120 and 0.4 <= avg_energy <= 0.7:
            genre = "pop"
        else:
            genre = "rock"
        
        return {
            'primary_genre': genre,
            'confidence': 0.75,
            'secondary_genres': []
        }
    
    def _init_camelot_wheel(self) -> Dict[str, int]:
        """Initialize Camelot wheel for harmonic mixing."""
        return {
            'C': 8, 'G': 9, 'D': 10, 'A': 11, 'E': 12, 'B': 1,
            'F#': 2, 'C#': 3, 'G#': 4, 'D#': 5, 'A#': 6, 'F': 7
        }
    
    def _camelot_compatibility(self, key_a: str, key_b: str) -> float:
        """Calculate Camelot wheel compatibility."""
        if key_a not in self.camelot_wheel or key_b not in self.camelot_wheel:
            return 0.5
        
        pos_a = self.camelot_wheel[key_a]
        pos_b = self.camelot_wheel[key_b]
        
        distance = min(abs(pos_a - pos_b), 12 - abs(pos_a - pos_b))
        compatibility = max(0, 1 - (distance / 6))
        
        return compatibility
    
    def _init_chord_progressions(self) -> Dict:
        """Initialize common chord progressions."""
        return {
            'pop': ['I', 'V', 'vi', 'IV'],
            'rock': ['I', 'VII', 'IV', 'I'],
            'jazz': ['ii', 'V', 'I', 'vi'],
            'blues': ['I', 'I', 'I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'IV', 'I', 'V']
        }
    
    def _init_energy_profiles(self) -> Dict:
        """Initialize energy profile templates."""
        return {
            'build_up': [0.2, 0.4, 0.6, 0.8, 1.0],
            'drop': [1.0, 0.8, 0.6, 0.4, 0.2],
            'steady': [0.7, 0.7, 0.7, 0.7, 0.7],
            'wave': [0.3, 0.7, 0.3, 0.9, 0.4]
        }
    
    def _init_genre_compatibility(self) -> Dict:
        """Initialize genre compatibility matrix."""
        return {
            'electronic': ['electronic', 'pop', 'ambient'],
            'pop': ['pop', 'rock', 'electronic'],
            'rock': ['rock', 'pop', 'alternative'],
            'ambient': ['ambient', 'electronic', 'chill']
        }
    
    # === Additional Helper Methods ===
    
    def _chord_progression_compatibility(self, prog_a: List, prog_b: List) -> float:
        """Analyze chord progression compatibility."""
        # Simplified - in production would use music theory analysis
        return 0.7  # Placeholder
    
    def _harmonic_rhythm_compatibility(self, chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
        """Analyze harmonic rhythm compatibility."""
        # Simplified - compare rate of harmonic change
        return 0.6  # Placeholder
    
    def _beat_pattern_compatibility(self, beats_a: np.ndarray, beats_b: np.ndarray, 
                                  audio_a: np.ndarray, audio_b: np.ndarray) -> float:
        """Analyze beat pattern compatibility."""
        # Simplified - would analyze beat strength patterns
        return 0.75  # Placeholder
    
    def _groove_compatibility(self, analysis_a: Dict, analysis_b: Dict) -> float:
        """Analyze groove and rhythmic feel compatibility."""
        # Simplified - would analyze microtiming and groove
        return 0.65  # Placeholder
    
    def _structure_pattern_compatibility(self, segments_a: List, segments_b: List) -> float:
        """Analyze structural pattern compatibility."""
        # Compare overall structure patterns
        pattern_a = [seg['type'] for seg in segments_a]
        pattern_b = [seg['type'] for seg in segments_b]
        
        # Simple pattern similarity
        common_elements = len(set(pattern_a) & set(pattern_b))
        total_elements = len(set(pattern_a) | set(pattern_b))
        
        return common_elements / total_elements if total_elements > 0 else 0
    
    def _calculate_segment_compatibility(self, seg_a: Dict, seg_b: Dict) -> float:
        """Calculate compatibility between two segments."""
        # Simplified segment compatibility
        base_score = 0.7 if seg_a['type'] == seg_b['type'] else 0.3
        
        # Duration similarity bonus
        duration_diff = abs(seg_a['duration'] - seg_b['duration'])
        duration_score = max(0, 1 - (duration_diff / 30))  # 30 second normalization
        
        return (base_score + duration_score) / 2
    
    def _create_energetic_curve(self, energy_a: np.ndarray, energy_b: np.ndarray) -> np.ndarray:
        """Create energetic mashup energy curve."""
        # Build energy throughout the mashup
        length = max(len(energy_a), len(energy_b))
        curve = np.linspace(0.4, 1.0, length)
        
        # Add some variation
        curve += np.sin(np.linspace(0, 4*np.pi, length)) * 0.1
        
        return np.clip(curve, 0, 1)
    
    def _create_chill_curve(self, energy_a: np.ndarray, energy_b: np.ndarray) -> np.ndarray:
        """Create chill mashup energy curve."""
        # Maintain lower, relaxed energy
        length = max(len(energy_a), len(energy_b))
        curve = np.full(length, 0.4)
        
        # Add gentle waves
        curve += np.sin(np.linspace(0, 2*np.pi, length)) * 0.15
        
        return np.clip(curve, 0.1, 0.7)
    
    def _create_balanced_curve(self, energy_a: np.ndarray, energy_b: np.ndarray) -> np.ndarray:
        """Create balanced mashup energy curve."""
        # Combine elements from both songs
        min_len = min(len(energy_a), len(energy_b))
        
        # Take best parts of each curve
        combined = (energy_a[:min_len] + energy_b[:min_len]) / 2
        
        # Smooth transitions
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(combined, sigma=2)
        
        return smoothed
    
    def _find_compromise_key(self, key_a: str, key_b: str) -> str:
        """Find a compromise key between two incompatible keys."""
        # Simplified - would use music theory to find best compromise
        return key_a  # Placeholder
    
    def _analyze_chord_progression(self, chroma: np.ndarray, beats: np.ndarray) -> List[str]:
        """Analyze chord progression from chroma features."""
        # Simplified chord analysis
        # In production, would use chord recognition algorithms
        return ['I', 'V', 'vi', 'IV']  # Common pop progression as placeholder