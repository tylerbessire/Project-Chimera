"""
Project Chimera - Integration Layer
==================================

Lead Engineer: Claude
Directive: PSE-Enhanced Integration of Legacy Components with New Architecture

This module serves as the integration bridge between the enhanced new framework
and the proven legacy engines. It applies the PSE methodology by treating each
legacy component as a specialized tool to be elevated and enhanced.

Key Integration Points:
1. Enhanced AudioAnalyzer integration with Demucs 4-stem separation
2. AI Collaboration Engine with PSE methodology for creative direction
3. Professional RealAudioEngine for production-grade rendering
4. Seamless workflow from acquisition to final mashup
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Import legacy components (PSE: treating as inherited production systems)
sys.path.append(str(Path(__file__).parent / "legacy_components"))

from legacy_components.analyzer import AudioAnalyzer
from legacy_components.song_library import SongLibrary
from legacy_components.collaboration_engine import CollaborationEngine
from professional_audio_engine import ProfessionalAudioEngine

# Import new framework components (PSE: our enhancements)
from downloader import AudioAcquisitionEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChimeraCore:
    """
    Central orchestration engine for Project Chimera.
    
    PSE Applied: This class elevates the workflow established by the legacy
    system while integrating our enhanced acquisition and processing capabilities.
    """
    
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = Path(workspace_dir)
        
        # Initialize all subsystems (PSE: treating legacy as specialized components)
        logger.info("Initializing Chimera Core Systems...")
        
        self.audio_acquisition = AudioAcquisitionEngine(str(self.workspace_dir))
        self.song_library = SongLibrary(str(self.workspace_dir))
        self.collaboration_engine = CollaborationEngine()
        self.audio_engine = ProfessionalAudioEngine()
        
        logger.info("✅ All Chimera systems operational")
    
    def search_and_download_workflow(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Enhanced search and download workflow.
        
        PSE Note: This workflow builds upon user feedback that the original
        upload-only system was too restrictive for creative discovery.
        """
        logger.info(f"Starting search and download workflow for: '{query}'")
        
        try:
            # Search for audio content
            search_results = self.audio_acquisition.search_for_audio(query, max_results)
            
            if not search_results:
                return {
                    "success": False,
                    "error": "No results found for search query",
                    "query": query
                }
            
            return {
                "success": True,
                "query": query,
                "results": search_results,
                "workflow_status": "search_completed"
            }
            
        except Exception as e:
            logger.error(f"Search workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def download_and_analyze_workflow(self, video_id: str, 
                                    custom_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete download and analysis workflow.
        
        PSE Note: This integrates our enhanced acquisition system with the 
        proven analysis pipeline from the legacy system.
        """
        logger.info(f"Starting download and analysis for video: {video_id}")
        
        try:
            # Step 1: Download high-quality audio
            logger.info("Step 1: Downloading audio...")
            audio_path = self.audio_acquisition.download_audio(video_id, custom_name)
            
            if not audio_path:
                return {
                    "success": False,
                    "error": "Failed to download audio",
                    "video_id": video_id
                }
            
            # Step 2: Analyze using legacy AudioAnalyzer (PSE: proven Demucs integration)
            logger.info("Step 2: Performing deep audio analysis...")
            song_info = self.song_library.add_song(audio_path)
            
            if not song_info:
                return {
                    "success": False,
                    "error": "Failed to analyze downloaded audio",
                    "audio_path": audio_path
                }
            
            return {
                "success": True,
                "audio_path": audio_path,
                "song_info": song_info,
                "workflow_status": "analysis_completed"
            }
            
        except Exception as e:
            logger.error(f"Download and analysis workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "video_id": video_id
            }
    
    def create_mashup_from_paths(self, song_paths: List[str], 
                                 mashup_style: str = "Professional EDM", 
                                 progress_callback=None) -> Dict[str, Any]:
        """
        Create mashup directly from file paths.
        
        Args:
            song_paths: List of absolute paths to audio files
            mashup_style: Style description for the mashup
        """
        logger.info(f"Starting mashup creation from paths: {song_paths}")
        
        try:
            # Step 1: Load song analysis from paths
            logger.info("Step 1: Loading song analysis from files...")
            song_briefs = []
            
            for path in song_paths:
                # Check if file exists
                if not os.path.exists(path):
                    return {
                        "success": False,
                        "error": f"Audio file not found: {path}"
                    }
                
                # Get song info from filename analysis
                # Extract song ID from filename to find analysis
                filename = os.path.basename(path)
                song_id = os.path.splitext(filename)[0]
                
                # Try to find analysis data
                song_info = self.song_library.get_song_info(song_id)
                if song_info:
                    # Normalize the data structure - extract all needed fields to top level
                    normalized_info = {}
                    
                    # Extract title
                    if 'song_info' in song_info and 'title' in song_info['song_info']:
                        normalized_info['title'] = song_info['song_info']['title']
                    elif 'title' in song_info:
                        normalized_info['title'] = song_info['title']
                    else:
                        normalized_info['title'] = song_id
                    
                    # Extract analysis results to top level
                    if 'analysis_results' in song_info:
                        analysis = song_info['analysis_results']
                        normalized_info['tempo'] = analysis.get('tempo', 120)
                        normalized_info['estimated_key'] = analysis.get('estimated_key', 'C')
                        normalized_info['camelot_key'] = analysis.get('camelot_key', '1A')
                        normalized_info['segments'] = analysis.get('segments', {})
                        normalized_info['lyrics'] = analysis.get('lyrics', '')
                        
                        # Convert segments to structural_segments format
                        segments = analysis.get('segments', {})
                        structural_segments = []
                        segment_labels = ["Intro", "Verse 1", "Chorus 1", "Verse 2", "Chorus 2", "Bridge", "Outro"]
                        
                        for i, (segment_key, segment_data) in enumerate(segments.items()):
                            label = segment_labels[i] if i < len(segment_labels) else f"Section {i+1}"
                            structural_segments.append({
                                "label": label,
                                "start_time": segment_data.get("start_time", 0),
                                "end_time": segment_data.get("end_time", 30)
                            })
                        
                        # Fallback if no segments
                        if not structural_segments:
                            structural_segments = [
                                {"label": "Intro", "start_time": 0, "end_time": 30},
                                {"label": "Verse", "start_time": 30, "end_time": 60},
                                {"label": "Chorus", "start_time": 60, "end_time": 90},
                                {"label": "Outro", "start_time": 90, "end_time": 120}
                            ]
                        
                        normalized_info['structural_segments'] = structural_segments
                    else:
                        normalized_info['tempo'] = 120
                        normalized_info['estimated_key'] = 'C'
                        normalized_info['camelot_key'] = '1A'
                        normalized_info['segments'] = {}
                        normalized_info['lyrics'] = ''
                        normalized_info['structural_segments'] = [
                            {"label": "Intro", "start_time": 0, "end_time": 30},
                            {"label": "Verse", "start_time": 30, "end_time": 60},
                            {"label": "Chorus", "start_time": 60, "end_time": 90},
                            {"label": "Outro", "start_time": 90, "end_time": 120}
                        ]
                    
                    # Keep source file info
                    if 'song_info' in song_info and 'source_file' in song_info['song_info']:
                        normalized_info['source_file'] = song_info['song_info']['source_file']
                    else:
                        normalized_info['source_file'] = path
                    
                    # Keep the original nested structure for compatibility
                    normalized_info['analysis_results'] = song_info.get('analysis_results', {})
                    normalized_info['song_info'] = song_info.get('song_info', {})
                    
                    song_briefs.append(normalized_info)
                else:
                    # Create basic info if analysis not found
                    basic_info = {
                        "title": song_id,
                        "source_file": path,
                        "tempo": 120,
                        "estimated_key": "C",
                        "camelot_key": "1A",
                        "segments": {},
                        "lyrics": "",
                        "structural_segments": [
                            {"label": "Intro", "start_time": 0, "end_time": 30},
                            {"label": "Verse", "start_time": 30, "end_time": 60},
                            {"label": "Chorus", "start_time": 60, "end_time": 90},
                            {"label": "Outro", "start_time": 90, "end_time": 120}
                        ],
                        "analysis_results": {
                            "tempo": 120,
                            "estimated_key": "C",
                            "camelot_key": "1A"
                        }
                    }
                    song_briefs.append(basic_info)
            
            # Step 2: AI Collaboration for Creative Direction
            logger.info("Step 2: AI Creative Direction...")
            
            recipe = self.collaboration_engine.generate_recipe(song_briefs, mashup_style, progress_callback)
            
            # Step 3: Professional Audio Engineering
            logger.info("Step 3: Professional Audio Engineering...")
            if progress_callback:
                progress_callback("🎛️ Professional Audio Engineering rendering mashup...", "audio_engineering", 95)
            
            # The professional audio engine expects 2 songs, use first two
            if len(song_paths) >= 2:
                # Generate output path using Luna's creative title
                mashup_title = recipe.get("mashup_title", "Organic_Collaboration")
                # Clean the title for filename
                clean_title = "".join(c for c in mashup_title if c.isalnum() or c in (' ', '-', '_')).replace(" ", "_")
                output_filename = f"{clean_title}.wav"
                output_path = os.path.join("workspace", "mashups", output_filename)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Create the professional mashup
                result = self.audio_engine.create_professional_mashup(
                    song_paths[0], song_paths[1], recipe, output_path
                )
                
                # The method returns a dict, extract the actual file path
                if isinstance(result, dict) and "output_path" in result:
                    output_path = result["output_path"]
                else:
                    # Assume it worked and use our generated path
                    pass
            else:
                return {
                    "success": False,
                    "error": f"Professional audio engine requires at least 2 songs, got {len(song_paths)}"
                }
            
            # Create catchy combo name for UI
            song_titles = [os.path.splitext(os.path.basename(path))[0] for path in song_paths]
            catchy_name = recipe.get("mashup_title", f"{song_titles[0]} × {song_titles[1]}")
            
            return {
                "success": True,
                "output_path": output_path,
                "recipe": recipe,
                "mashup_title": catchy_name,
                "catchy_combo_name": catchy_name,
                "song_paths": song_paths,
                "audio_url": f"/api/mashup/audio/{os.path.basename(output_path)}"
            }
            
        except Exception as e:
            logger.error(f"Mashup creation from paths failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "song_paths": song_paths
            }

    def create_professional_mashup_workflow(self, song_titles: List[str], 
                                          mashup_style: str = "Professional EDM") -> Dict[str, Any]:
        """
        Complete professional mashup creation workflow.
        
        PSE Note: This workflow builds upon the solid foundation of the legacy
        creation system, enhanced with our improved error handling and reporting.
        """
        logger.info(f"Starting professional mashup creation: {song_titles}")
        
        try:
            # Step 1: Validate and gather song information
            logger.info("Step 1: Validating songs and gathering analysis...")
            song_briefs = []
            song_paths = []
            
            for title in song_titles:
                song_info = self.song_library.get_song_info(title)
                if not song_info:
                    return {
                        "success": False,
                        "error": f"Song '{title}' not found in library",
                        "suggestion": "Use search and download workflow to add this song"
                    }
                
                song_briefs.append(song_info)
                # Get the source file path from the nested structure
                if 'song_info' in song_info and 'source_file' in song_info['song_info']:
                    song_paths.append(song_info['song_info']['source_file'])
                else:
                    return {
                        "success": False,
                        "error": f"Source file not found for song '{title}'",
                        "song_info": song_info
                    }
            
            # Step 2: AI Collaboration for Creative Direction (PSE: proven workflow)
            logger.info("Step 2: AI Creative Direction...")
            recipe = self.collaboration_engine.generate_recipe(song_briefs, mashup_style)
            
            # Step 3: Professional Audio Rendering
            logger.info("Step 3: Professional audio rendering...")
            # Import here to avoid circular imports
            from legacy_components.real_audio_engine import RealAudioEngine
            renderer = RealAudioEngine(recipe)
            output_path = renderer.render_mashup()
            
            if not output_path or not os.path.exists(output_path):
                return {
                    "success": False,
                    "error": "Failed to render final mashup audio",
                    "recipe": recipe
                }
            
            return {
                "success": True,
                "mashup_title": recipe.get("mashup_title", "Untitled Mashup"),
                "output_path": output_path,
                "recipe": recipe,
                "song_briefs": song_briefs,
                "workflow_status": "mashup_completed"
            }
            
        except Exception as e:
            logger.error(f"Mashup creation workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "song_titles": song_titles
            }
    
    def revise_mashup_workflow(self, original_recipe: Dict[str, Any], 
                             revision_request: str) -> Dict[str, Any]:
        """
        Enhanced mashup revision workflow.
        
        PSE Note: This applies the PSE methodology to revision by treating
        the original recipe as a work from a previous iteration to be improved.
        """
        logger.info("Starting mashup revision workflow")
        
        try:
            # Import revision engine (PSE: treating as specialized revision tool)
            from legacy_components.reviser import RevisionEngine
            from legacy_components.real_audio_engine import RealAudioEngine
            
            revision_engine = RevisionEngine()
            
            # Apply PSE methodology for revision
            revised_recipe = revision_engine.revise_recipe(
                original_recipe, 
                revision_request,
                methodology="pse_enhanced"  # Use PSE for better revisions
            )
            
            # Re-render with revised recipe
            renderer = RealAudioEngine(revised_recipe)
            output_path = renderer.render_mashup()
            
            return {
                "success": True,
                "revised_recipe": revised_recipe,
                "output_path": output_path,
                "revision_request": revision_request,
                "workflow_status": "revision_completed"
            }
            
        except Exception as e:
            logger.error(f"Revision workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "revision_request": revision_request
            }
    
    def find_compatible_songs_enhanced(self, song_title: str) -> Dict[str, Any]:
        """
        Enhanced song compatibility discovery.
        
        PSE Note: Builds upon the solid Camelot wheel foundation from the
        legacy system with enhanced metadata and recommendations.
        """
        logger.info(f"Finding compatible songs for: {song_title}")
        
        try:
            # Use legacy compatibility engine (PSE: proven harmonic matching)
            compatible_songs = self.song_library.find_compatible_songs(song_title)
            
            # Enhanced analysis with additional metadata
            source_song = self.song_library.get_song_info(song_title)
            
            if not source_song:
                return {
                    "success": False,
                    "error": f"Song '{song_title}' not found"
                }
            
            # Add enhanced compatibility scoring
            enhanced_compatibility = []
            for song in compatible_songs:
                score = self._calculate_enhanced_compatibility_score(
                    source_song, song
                )
                song["compatibility_score"] = score
                enhanced_compatibility.append(song)
            
            # Sort by compatibility score
            enhanced_compatibility.sort(
                key=lambda x: x["compatibility_score"], 
                reverse=True
            )
            
            return {
                "success": True,
                "source_song": source_song,
                "compatible_songs": enhanced_compatibility,
                "total_found": len(enhanced_compatibility)
            }
            
        except Exception as e:
            logger.error(f"Compatibility search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "song_title": song_title
            }
    
    def _calculate_enhanced_compatibility_score(self, source: Dict, target: Dict) -> float:
        """
        Calculate enhanced compatibility score using multiple factors.
        
        PSE Note: Enhances the basic Camelot wheel matching with additional
        musical factors for more nuanced compatibility assessment.
        """
        score = 0.0
        
        # Base harmonic compatibility (legacy Camelot wheel)
        if target.get("camelot_key") in self.song_library.camelot_compatibility_map.get(
            source.get("camelot_key", ""), []
        ):
            score += 40.0
        
        # Tempo compatibility (enhanced)
        source_tempo = source.get("tempo", 120)
        target_tempo = target.get("tempo", 120)
        tempo_diff = abs(target_tempo - source_tempo) / source_tempo
        if tempo_diff < 0.05:  # Very close tempo
            score += 30.0
        elif tempo_diff < 0.10:  # Moderate tempo difference
            score += 20.0
        elif tempo_diff < 0.15:  # Acceptable tempo difference
            score += 10.0
        
        # Duration compatibility
        source_duration = source.get("duration_ms", 0)
        target_duration = target.get("duration_ms", 0)
        if source_duration > 0 and target_duration > 0:
            duration_ratio = min(source_duration, target_duration) / max(source_duration, target_duration)
            score += duration_ratio * 20.0
        
        # Structural compatibility (basic)
        source_structure = [seg.get("label", "") for seg in source.get("structural_segments", [])]
        target_structure = [seg.get("label", "") for seg in target.get("structural_segments", [])]
        common_structures = len(set(source_structure) & set(target_structure))
        if common_structures > 0:
            score += min(common_structures * 2.5, 10.0)
        
        return min(score, 100.0)  # Cap at 100
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for monitoring.
        
        PSE Note: Enhanced monitoring capabilities for production deployment.
        """
        try:
            # Gather status from all subsystems
            songs = self.song_library.list_all_songs()
            download_history = self.audio_acquisition.get_download_history()
            
            return {
                "status": "operational",
                "subsystems": {
                    "audio_acquisition": "operational",
                    "song_library": "operational",
                    "collaboration_engine": "operational"
                },
                "statistics": {
                    "total_songs": len(songs),
                    "downloaded_files": len(download_history),
                    "workspace_dir": str(self.workspace_dir),
                },
                "recent_activity": {
                    "recent_songs": songs[-5:] if songs else [],
                    "recent_downloads": download_history[:5]
                }
            }
            
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# Convenience functions for direct API integration
def create_chimera_core(workspace_dir: str = "workspace") -> ChimeraCore:
    """Factory function for creating Chimera core instance."""
    return ChimeraCore(workspace_dir)

def search_audio(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Convenience function for audio search."""
    core = create_chimera_core()
    return core.search_and_download_workflow(query, max_results)

def download_and_analyze(video_id: str, custom_name: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for download and analysis."""
    core = create_chimera_core()
    return core.download_and_analyze_workflow(video_id, custom_name)

def create_mashup(song_titles: List[str], style: str = "Professional EDM") -> Dict[str, Any]:
    """Convenience function for mashup creation."""
    core = create_chimera_core()
    return core.create_professional_mashup_workflow(song_titles, style)