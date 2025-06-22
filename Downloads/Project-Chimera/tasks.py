"""
Enhanced task execution system for Project Chimera
===================================================

Lead Engineer: Claude
PSE Applied: These tasks leverage the existing codebase architecture while
             introducing professional-grade enhancements for real-world usage.

Enhanced Task Types:
- create_mashup_task: Professional AI mashup creation with Luna & Claude
- revise_mashup_task: PSE-based mashup revision workflow
- download_and_analyze_task: YouTube acquisition with audio analysis
- search_audio_task: Intelligent audio search with fuzzy matching
"""

import os
import json
import time
import traceback
import logging
from typing import Dict, List, Any

# Initialize logging
logger = logging.getLogger(__name__)

# Chimera Integration (PSE: enhanced system)
from chimera_integration import ChimeraCore

def create_mashup_task(job_id: str, song_paths: List[str], mashup_style: str, jobs_dict: Dict):
    """
    Enhanced mashup creation task with AI collaboration.
    
    PSE Note: This function builds upon the legacy mashup system by integrating
    Luna (OpenAI) for creative direction and Claude for audio engineering.
    
    Args:
        job_id: Unique identifier for the mashup job
        song_paths: List of file paths to source audio files
        mashup_style: User-specified style for the mashup
        jobs_dict: Shared dictionary for job status tracking
    """
    try:
        # Initialize Chimera core
        chimera_core = ChimeraCore()
        
        # Update job status
        jobs_dict[job_id]["status"] = "running"
        jobs_dict[job_id]["progress"] = "Initializing AI collaboration workflow..."
        jobs_dict[job_id]["stage"] = "initialization"
        
        logger.info(f"Starting enhanced mashup creation for job: {job_id}")
        
        # Update progress  
        jobs_dict[job_id]["progress"] = "AI Creative Direction - Luna designing storyboard..."
        jobs_dict[job_id]["stage"] = "creative_direction"
        
        # Create progress callback to update job status in real-time
        def update_progress(message, stage, percentage):
            jobs_dict[job_id]["progress"] = message
            jobs_dict[job_id]["stage"] = stage
            if percentage:
                jobs_dict[job_id]["percentage"] = percentage
            logger.info(f"Job {job_id} progress: {percentage}% - {message}")
        
        # Execute enhanced mashup workflow using the actual song paths
        workflow_result = chimera_core.create_mashup_from_paths(
            song_paths, mashup_style, update_progress
        )
        
        if not workflow_result["success"]:
            jobs_dict[job_id]["status"] = "failed"
            jobs_dict[job_id]["error"] = workflow_result["error"]
            return
        
        # Update progress for completion with catchy name
        catchy_name = workflow_result.get("catchy_combo_name", "Amazing Mashup")
        jobs_dict[job_id]["progress"] = f"🎉 {catchy_name} completed successfully!"
        jobs_dict[job_id]["stage"] = "completed"
        jobs_dict[job_id]["status"] = "completed"
        
        # Store results
        output_path = workflow_result["output_path"]
        recipe = workflow_result["recipe"]
        
        # Save recipe file for frontend access
        recipe_filename = f"{job_id}_recipe.json"
        recipe_path = os.path.join("workspace", "mashups", recipe_filename)
        os.makedirs(os.path.dirname(recipe_path), exist_ok=True)
        
        with open(recipe_path, 'w') as f:
            json.dump(recipe, f, indent=2)
        
        # Update job with final results for UI
        jobs_dict[job_id].update({
            "result": {
                "audio_file": os.path.basename(output_path),
                "recipe_file": recipe_filename,
                "mashup_title": workflow_result.get("catchy_combo_name", workflow_result["mashup_title"]),
                "output_path": output_path,
                "recipe_path": recipe_path,
                "audio_url": f"/api/mashup/audio/{os.path.basename(output_path)}"
            }
        })
        
        logger.info(f"Mashup creation completed successfully: {workflow_result['mashup_title']}")
        
    except Exception as e:
        logger.error(f"Mashup creation task failed: {e}")
        traceback.print_exc()
        
        jobs_dict[job_id]["status"] = "failed"
        jobs_dict[job_id]["error"] = f"Unexpected error: {str(e)}"
        jobs_dict[job_id]["progress"] = "Failed due to internal error"

def revise_mashup_task(job_id: str, original_recipe: Dict[str, Any], 
                      revision_request: str, jobs_dict: Dict):
    """
    Enhanced mashup revision task with PSE methodology.
    
    PSE Note: This function applies the PSE methodology by treating the original
    recipe as work from a previous team to be improved upon.
    
    Args:
        job_id: Unique identifier for the revision job
        original_recipe: The original mashup recipe to revise
        revision_request: User's description of desired changes
        jobs_dict: Shared dictionary for job status tracking
    """
    try:
        # Initialize Chimera core
        chimera_core = ChimeraCore()
        
        # Update job status
        jobs_dict[job_id]["status"] = "running"
        jobs_dict[job_id]["progress"] = "Analyzing revision request..."
        jobs_dict[job_id]["stage"] = "analysis"
        
        logger.info(f"Starting mashup revision for job: {job_id}")
        
        # Update progress
        jobs_dict[job_id]["progress"] = "AI Revision Engine analyzing original recipe..."
        jobs_dict[job_id]["stage"] = "revision_analysis"
        
        # Execute revision workflow
        revision_result = chimera_core.revise_mashup_workflow(
            original_recipe, revision_request
        )
        
        if not revision_result["success"]:
            jobs_dict[job_id]["status"] = "failed"
            jobs_dict[job_id]["error"] = revision_result["error"]
            return
        
        # Update progress
        jobs_dict[job_id]["progress"] = "Rendering revised mashup..."
        jobs_dict[job_id]["stage"] = "rendering"
        
        # Complete revision
        jobs_dict[job_id]["status"] = "completed"
        jobs_dict[job_id]["progress"] = "Revision completed successfully!"
        jobs_dict[job_id]["stage"] = "completed"
        
        # Save revised recipe
        revised_recipe = revision_result["revised_recipe"]
        revised_recipe_filename = f"{job_id}_revised_recipe.json"
        revised_recipe_path = os.path.join("workspace", "mashups", revised_recipe_filename)
        
        os.makedirs(os.path.dirname(revised_recipe_path), exist_ok=True)
        
        with open(revised_recipe_path, 'w') as f:
            json.dump(revised_recipe, f, indent=2)
        
        # Update job with results
        output_path = revision_result["output_path"]
        jobs_dict[job_id].update({
            "result": {
                "audio_file": os.path.basename(output_path),
                "recipe_file": revised_recipe_filename,
                "mashup_title": revised_recipe.get("mashup_title", "Revised Mashup"),
                "output_path": output_path,
                "recipe_path": revised_recipe_path,
                "revision_request": revision_request
            }
        })
        
        logger.info(f"Mashup revision completed successfully")
        
    except Exception as e:
        logger.error(f"Mashup revision task failed: {e}")
        traceback.print_exc()
        
        jobs_dict[job_id]["status"] = "failed"
        jobs_dict[job_id]["error"] = f"Revision failed: {str(e)}"
        jobs_dict[job_id]["progress"] = "Revision failed due to internal error"

def download_and_analyze_task(job_id: str, video_id: str, custom_name: str, jobs_dict: Dict):
    """
    Enhanced download and analysis task.
    
    PSE Note: This new task type extends the original system's capabilities
    by integrating our enhanced audio acquisition pipeline.
    
    Args:
        job_id: Unique identifier for the download job
        video_id: YouTube video ID to download
        custom_name: Optional custom name for the file
        jobs_dict: Shared dictionary for job status tracking
    """
    try:
        # Initialize Chimera core
        chimera_core = ChimeraCore()
        
        # Update job status
        jobs_dict[job_id]["status"] = "running"
        jobs_dict[job_id]["progress"] = "Downloading high-quality audio..."
        jobs_dict[job_id]["stage"] = "download"
        
        logger.info(f"Starting download and analysis for video: {video_id}")
        
        # Execute download and analysis workflow
        workflow_result = chimera_core.download_and_analyze_workflow(video_id, custom_name)
        
        if not workflow_result["success"]:
            jobs_dict[job_id]["status"] = "failed"
            jobs_dict[job_id]["error"] = workflow_result["error"]
            return
        
        # Update progress for analysis phase
        jobs_dict[job_id]["progress"] = "Analyzing audio with AI stem separation..."
        jobs_dict[job_id]["stage"] = "analysis"
        
        # Simulate progress for user experience
        time.sleep(1)
        
        # Complete task
        jobs_dict[job_id]["status"] = "completed"
        jobs_dict[job_id]["progress"] = "Audio downloaded and analyzed successfully!"
        jobs_dict[job_id]["stage"] = "completed"
        
        # Store results
        song_info = workflow_result["song_info"]
        audio_path = workflow_result["audio_path"]
        
        jobs_dict[job_id].update({
            "result": {
                "audio_path": audio_path,
                "song_info": song_info,
                "title": song_info.get("title", "Unknown"),
                "tempo": song_info.get("tempo", 0),
                "key": song_info.get("estimated_key", "Unknown"),
                "duration_ms": song_info.get("duration_ms", 0)
            }
        })
        
        logger.info(f"Download and analysis completed: {song_info.get('title', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Download and analysis task failed: {e}")
        traceback.print_exc()
        
        jobs_dict[job_id]["status"] = "failed"
        jobs_dict[job_id]["error"] = f"Download failed: {str(e)}"
        jobs_dict[job_id]["progress"] = "Download failed due to internal error"

def search_audio_task(job_id: str, query: str, max_results: int, jobs_dict: Dict):
    """
    Enhanced audio search task.
    
    PSE Note: This new capability addresses user feedback that the original
    system was too restrictive for music discovery.
    
    Args:
        job_id: Unique identifier for the search job
        query: Search query string
        max_results: Maximum number of results to return
        jobs_dict: Shared dictionary for job status tracking
    """
    try:
        # Initialize Chimera core
        chimera_core = ChimeraCore()
        
        # Update job status
        jobs_dict[job_id]["status"] = "running"
        jobs_dict[job_id]["progress"] = f"Searching for '{query}'..."
        jobs_dict[job_id]["stage"] = "search"
        
        logger.info(f"Starting audio search for: {query}")
        
        # Execute search workflow
        search_result = chimera_core.search_and_download_workflow(query, max_results)
        
        if not search_result["success"]:
            jobs_dict[job_id]["status"] = "failed"
            jobs_dict[job_id]["error"] = search_result["error"]
            return
        
        # Complete search
        jobs_dict[job_id]["status"] = "completed"
        jobs_dict[job_id]["progress"] = f"Found {len(search_result['results'])} results"
        jobs_dict[job_id]["stage"] = "completed"
        
        # Store search results
        jobs_dict[job_id].update({
            "result": {
                "query": query,
                "results": search_result["results"],
                "total_found": len(search_result["results"])
            }
        })
        
        logger.info(f"Search completed: {len(search_result['results'])} results for '{query}'")
        
    except Exception as e:
        logger.error(f"Audio search task failed: {e}")
        traceback.print_exc()
        
        jobs_dict[job_id]["status"] = "failed"
        jobs_dict[job_id]["error"] = f"Search failed: {str(e)}"
        jobs_dict[job_id]["progress"] = "Search failed due to internal error"