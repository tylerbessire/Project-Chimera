"""
Project Chimera - Advanced Audio Acquisition Module
==================================================

Lead Engineer: Claude
Architecture: PSE-Enhanced Audio Acquisition System

This module implements professional-grade audio acquisition capabilities
using yt-dlp for robust download and thefuzz for intelligent search.

Key Features:
- YouTube search with ranking and relevance scoring
- High-quality audio extraction (best available format -> WAV)
- Fuzzy matching for typo correction and search optimization
- Workspace organization and file management
- Error handling and download verification
"""

import os
import yt_dlp
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from thefuzz import fuzz, process
import json
import re
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAcquisitionEngine:
    """
    Professional audio acquisition system for Project Chimera.
    Handles search, download, and organization of audio sources.
    """
    
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.audio_sources_dir = self.workspace_dir / "audio_sources"
        self.cache_dir = self.workspace_dir / "cache"
        
        # Create directories
        self.audio_sources_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure yt-dlp with professional settings
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.audio_sources_dir / '%(title)s_%(id)s.%(ext)s'),
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': '0',  # Best quality
            'quiet': True,
            'no_warnings': True,
        }
        
        logger.info(f"Audio Acquisition Engine initialized - Workspace: {self.workspace_dir}")
    
    def search_for_audio(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for audio content with intelligent fuzzy matching.
        
        Args:
            query: Search query (artist, song, etc.)
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with metadata and relevance scores
        """
        logger.info(f"Searching for audio: '{query}'")
        
        # Enhance query with fuzzy matching improvements
        enhanced_query = self._enhance_search_query(query)
        
        # Perform YouTube search
        search_query = f"ytsearch{max_results}:{enhanced_query}"
        
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            try:
                search_results = ydl.extract_info(search_query, download=False)
                
                if not search_results or 'entries' not in search_results:
                    return []
                
                # Process and rank results
                processed_results = []
                for entry in search_results['entries']:
                    if entry:
                        result = self._process_search_result(entry, query)
                        processed_results.append(result)
                
                # Sort by relevance score
                processed_results.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                logger.info(f"Found {len(processed_results)} results for '{query}'")
                return processed_results
                
            except Exception as e:
                logger.error(f"Search failed for '{query}': {e}")
                return []
    
    def download_audio(self, video_id: str, custom_filename: Optional[str] = None) -> Optional[str]:
        """
        Download audio from YouTube and convert to high-quality WAV.
        
        Args:
            video_id: YouTube video ID or URL
            custom_filename: Optional custom filename
            
        Returns:
            Path to downloaded WAV file, or None if failed
        """
        logger.info(f"Downloading audio: {video_id}")

        # Check if audio already exists in workspace
        if custom_filename:
            safe_name = self._sanitize_filename(custom_filename) + ".wav"
            existing = self.audio_sources_dir / safe_name
            if existing.exists():
                logger.info(f"Audio already present: {existing}")
                return str(existing)
        else:
            existing_files = list(self.audio_sources_dir.glob(f"*{video_id}*.wav"))
            if existing_files:
                logger.info(f"Audio already present: {existing_files[0]}")
                return str(existing_files[0])

        # Prepare download configuration
        if custom_filename:
            # Clean filename for filesystem safety
            safe_filename = self._sanitize_filename(custom_filename)
            output_template = str(self.audio_sources_dir / f"{safe_filename}.%(ext)s")
        else:
            output_template = str(self.audio_sources_dir / '%(title)s_%(id)s.%(ext)s')
        
        download_opts = {
            **self.ydl_opts,
            'outtmpl': output_template,
        }
        
        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                # Extract info first to get metadata
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                if not info:
                    logger.error(f"Could not extract info for {video_id}")
                    return None
                
                # Download the audio
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                
                # Find the downloaded file
                expected_filename = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
                expected_path = Path(expected_filename)
                
                # Verify the download and convert if necessary
                return self._verify_and_convert_download(expected_path, info)
                
        except Exception as e:
            logger.error(f"Download failed for {video_id}: {e}")
            return None
    
    def _enhance_search_query(self, query: str) -> str:
        """
        Enhance search query using fuzzy matching and common corrections.
        """
        # Common artist/song corrections database
        common_corrections = {
            'beet': 'beat',
            'wrapp': 'rap',
            'hiphop': 'hip hop',
            'edm': 'electronic dance music',
            'dnb': 'drum and bass',
        }
        
        # Apply common corrections
        enhanced = query.lower()
        for mistake, correction in common_corrections.items():
            enhanced = enhanced.replace(mistake, correction)
        
        # Remove extra spaces and clean up
        enhanced = ' '.join(enhanced.split())
        
        return enhanced
    
    def _process_search_result(self, entry: Dict, original_query: str) -> Dict:
        """
        Process and score a search result for relevance.
        """
        title = entry.get('title', '')
        uploader = entry.get('uploader', '')
        duration = entry.get('duration', 0)
        view_count = entry.get('view_count', 0)
        
        # Calculate relevance score using fuzzy matching
        title_score = fuzz.partial_ratio(original_query.lower(), title.lower())
        uploader_score = fuzz.partial_ratio(original_query.lower(), uploader.lower())
        
        # Combine scores with weighting
        relevance_score = (title_score * 0.7 + uploader_score * 0.3)
        
        # Boost score for reasonable duration (2-8 minutes for typical songs)
        if 120 <= duration <= 480:
            relevance_score *= 1.1
        
        # Slight boost for higher view counts (indicates popularity/quality)
        if view_count > 100000:
            relevance_score *= 1.05
        
        return {
            'id': entry.get('id'),
            'title': title,
            'uploader': uploader,
            'duration': duration,
            'duration_string': self._format_duration(duration),
            'view_count': view_count,
            'thumbnail': entry.get('thumbnail'),
            'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
            'relevance_score': relevance_score
        }
    
    def _verify_and_convert_download(self, expected_path: Path, info: Dict) -> Optional[str]:
        """
        Verify download succeeded and convert to WAV if necessary.
        """
        # Look for the actual downloaded file
        possible_extensions = ['.wav', '.webm', '.m4a', '.mp3', '.ogg']
        actual_file = None
        
        for ext in possible_extensions:
            test_path = expected_path.with_suffix(ext)
            if test_path.exists():
                actual_file = test_path
                break
        
        if not actual_file:
            logger.error(f"Downloaded file not found at expected location: {expected_path}")
            return None
        
        # If already WAV, we're done
        if actual_file.suffix.lower() == '.wav':
            logger.info(f"Audio downloaded successfully: {actual_file}")
            return str(actual_file)
        
        # Convert to WAV using ffmpeg
        wav_path = actual_file.with_suffix('.wav')
        return self._convert_to_wav(actual_file, wav_path)
    
    def _convert_to_wav(self, input_path: Path, output_path: Path) -> Optional[str]:
        """
        Convert audio file to high-quality WAV using ffmpeg.
        """
        try:
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '44100',          # 44.1kHz sample rate
                '-ac', '2',              # Stereo
                '-y',                    # Overwrite output
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Remove original file to save space
                input_path.unlink()
                logger.info(f"Converted to WAV: {output_path}")
                return str(output_path)
            else:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                return str(input_path)  # Return original if conversion fails
                
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return str(input_path)
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for filesystem safety.
        """
        # Remove invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Limit length
        return sanitized[:100]
    
    def _format_duration(self, seconds: int) -> str:
        """
        Format duration in seconds to MM:SS format.
        """
        if not seconds:
            return "Unknown"
        
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:02d}"
    
    def get_download_history(self) -> List[Dict]:
        """
        Get list of previously downloaded audio files.
        """
        history = []
        
        for audio_file in self.audio_sources_dir.glob("*.wav"):
            file_info = {
                'filename': audio_file.name,
                'path': str(audio_file),
                'size_mb': round(audio_file.stat().st_size / (1024 * 1024), 2),
                'created': audio_file.stat().st_ctime
            }
            history.append(file_info)
        
        # Sort by creation time (newest first)
        history.sort(key=lambda x: x['created'], reverse=True)
        
        return history
    
    def cleanup_workspace(self, keep_recent: int = 10):
        """
        Clean up workspace, keeping only recent downloads.
        """
        history = self.get_download_history()
        
        if len(history) > keep_recent:
            files_to_remove = history[keep_recent:]
            
            for file_info in files_to_remove:
                try:
                    Path(file_info['path']).unlink()
                    logger.info(f"Cleaned up: {file_info['filename']}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_info['filename']}: {e}")


# Convenience functions for direct API use
def search_audio(query: str, max_results: int = 5) -> List[Dict]:
    """Convenience function for audio search."""
    engine = AudioAcquisitionEngine()
    return engine.search_for_audio(query, max_results)

def download_audio(video_id: str, custom_filename: Optional[str] = None) -> Optional[str]:
    """Convenience function for audio download."""
    engine = AudioAcquisitionEngine()
    return engine.download_audio(video_id, custom_filename)
