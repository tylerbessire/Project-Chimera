# ==============================================================================
# FILE: song_library.py (v7.0 - Studio Grade Update)
# ==============================================================================
#
# MAJOR UPDATES:
# - Fully integrated with the new `AudioAnalyzer`.
# - Added `find_compatible_songs` function which implements Camelot wheel mixing
#   rules to suggest harmonically compatible tracks.
# - Manages a structured library where each song has its own folder with
#   source audio, stems, and analysis data.
#
# ==============================================================================

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from analyzer import AudioAnalyzer

class SongLibrary:
    """Manages the song library with analysis data and compatibility checks."""

    def __init__(self, workspace_dir="workspace"):
        self.base_dir = Path(workspace_dir)
        self.songs_dir = self.base_dir / "songs"
        self.source_dir = self.base_dir / "audio_sources"
        os.makedirs(self.songs_dir, exist_ok=True)
        os.makedirs(self.source_dir, exist_ok=True)
        
        self.camelot_compatibility_map = self._build_camelot_map()

    def list_all_songs(self) -> List[Dict]:
        """Lists all songs that have been successfully analyzed."""
        songs = []
        if not self.songs_dir.exists():
            return songs
        for song_id_dir in self.songs_dir.iterdir():
            if song_id_dir.is_dir():
                info_path = song_id_dir / "analysis.json"
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        analysis_data = json.load(f)
                        # Extract title from nested structure
                        if 'song_info' in analysis_data and 'title' in analysis_data['song_info']:
                            analysis_data['title'] = analysis_data['song_info']['title']
                        songs.append(analysis_data)
        return sorted(songs, key=lambda x: x.get('title', ''))

    def add_song(self, audio_file_path: str, force_reanalyze=False) -> Optional[Dict]:
        """Adds a new song to the library, analyzes it, and stores the results."""
        try:
            filename = os.path.basename(audio_file_path)
            song_id = os.path.splitext(filename)[0]
            song_dir = self.songs_dir / song_id
            
            # If we're not forcing reanalysis, check if it already exists
            if not force_reanalyze and song_dir.exists():
                print(f"Song '{song_id}' already exists in the library.")
                return self.get_song_info(song_id)

            os.makedirs(song_dir, exist_ok=True)
            
            # Copy source audio to a centralized location if it's not already there
            source_path = self.source_dir / filename
            if not source_path.exists():
                 shutil.copy2(audio_file_path, source_path)
            
            # Perform the analysis
            analyzer = AudioAnalyzer(str(source_path))
            analysis_data = analyzer.full_analysis(force_reanalyze=force_reanalyze)
            
            # Save the analysis data in the song's directory
            with open(song_dir / "analysis.json", 'w') as f:
                json.dump(analysis_data, f, indent=4)
                
            return analysis_data
        except Exception as e:
            print(f"❌ Error adding song {audio_file_path}: {e}")
            return None

    def get_song_info(self, song_id: str) -> Optional[Dict]:
        """Retrieves the analysis info for a specific song."""
        info_path = self.songs_dir / song_id / "analysis.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        return None

    def find_song_by_query(self, query: str) -> Optional[Dict]:
        """Find a song by fuzzy matching with clean file path resolution."""
        query_lower = query.lower().strip()
        
        # Get all songs
        all_songs = self.list_all_songs()
        if not all_songs:
            return None
        
        # Create mapping for common search terms
        search_mappings = {
            "mr brightside": "brightside",
            "the killers": "brightside", 
            "levels": "levels",
            "avicii": "levels",
            "goodbye my lover": "goodbye",
            "james blunt": "goodbye"
        }
        
        # Try direct mapping first
        for search_term, key in search_mappings.items():
            if search_term in query_lower:
                for song in all_songs:
                    title = song.get('title', '').lower()
                    if key in title:
                        return self._prepare_song_result(song)
        
        # Try word matching
        query_words = query_lower.split()
        best_match = None
        max_score = 0
        
        for song in all_songs:
            title = song.get('title', '').lower()
            score = sum(1 for word in query_words if word in title)
            
            if score > max_score and score >= 1:  # At least one word match
                max_score = score
                best_match = song
        
        if best_match:
            return self._prepare_song_result(best_match)
            
        return None
    
    def _prepare_song_result(self, song: Dict) -> Dict:
        """Prepare song result with correct file path."""
        # Find the actual audio file
        if 'song_info' in song and 'source_file' in song['song_info']:
            source_path = song['song_info']['source_file']
            if os.path.exists(source_path):
                song['source_file_path'] = source_path
                return song
        
        # Try common locations
        title = song.get('title', '')
        possible_files = [
            f"{title}.wav",
            f"{title}.mp3",
            "audio.wav",
            "audio.mp3"
        ]
        
        # Check audio_sources directory
        for filename in possible_files:
            path = self.source_dir / filename
            if path.exists():
                song['source_file_path'] = str(path)
                return song
        
        # Last resort - find any audio file in source_dir that might match
        if self.source_dir.exists():
            for file in self.source_dir.iterdir():
                if file.suffix.lower() in ['.wav', '.mp3'] and any(word in file.stem.lower() for word in title.lower().split()):
                    song['source_file_path'] = str(file)
                    return song
        
        return None

    def find_compatible_songs(self, source_song_id: str) -> List[Dict]:
        """
        Finds songs in the library that are harmonically compatible with the
        source song based on the Camelot wheel rules.
        """
        source_song_info = self.get_song_info(source_song_id)
        if not source_song_info or 'camelot_key' not in source_song_info:
            return []
            
        source_camelot = source_song_info['camelot_key']
        source_tempo = source_song_info.get('tempo', 120)
        compatible_keys = self.camelot_compatibility_map.get(source_camelot, [])
        
        compatible_songs = []
        all_songs = self.list_all_songs()
        
        for song in all_songs:
            if song['title'] == source_song_id:
                continue
            
            target_camelot = song.get('camelot_key')
            target_tempo = song.get('tempo', 120)

            # Check for key compatibility and reasonable tempo difference (+/- 8%)
            if target_camelot in compatible_keys and abs(target_tempo - source_tempo) / source_tempo < 0.08:
                compatible_songs.append(song)
                
        return compatible_songs
        
    def _build_camelot_map(self):
        """Creates the compatibility lookup table for the Camelot wheel."""
        map = {}
        for i in range(1, 13):
            # Minor Keys ('A')
            num = i
            prev_num = 12 if num == 1 else num - 1
            next_num = 1 if num == 12 else num + 1
            map[f"{num}A"] = [f"{num}A", f"{num}B", f"{prev_num}A", f"{next_num}A"]
            
            # Major Keys ('B')
            map[f"{num}B"] = [f"{num}B", f"{num}A", f"{prev_num}B", f"{next_num}B"]
        return map
