"""
Project Chimera - Final Production API Server
============================================

Lead Engineer: Claude
Directive: Final integration of the production-ready ChimeraCore engine with the Flask API.

PSE Applied: This server is the final, elevated version, replacing all simulated
logic with the real, professional-grade components. It is now a fully functional
system ready for production use.

Key Enhancements:
- REMOVED all simulated mashup creation logic.
- The `/api/create` endpoint now correctly uses the real `create_mashup_task`.
- Full integration with the ChimeraCore for a seamless, production-ready workflow.
"""

import os
import uuid
import traceback
import logging
from threading import Thread
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Chimera Integration Layer (PSE: our enhanced system)
from chimera_integration import ChimeraCore
from tasks import (
    create_mashup_task,
    revise_mashup_task,
    download_and_analyze_task,
    search_audio_task
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize Chimera Core System (PSE: unified integration)
JOBS = {}
CHIMERA_CORE = ChimeraCore()

# --- API Endpoints ---

@app.route('/api/search', methods=['POST'])
def search_audio():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Please provide a 'query' field"}), 400
    
    query = data['query']
    max_results = data.get('max_results', 5)
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "running",
        "progress": f"Searching for '{query}'...",
        "stage": "search",
        "query": query
    }
    thread = Thread(target=search_audio_task, args=(job_id, query, max_results, JOBS))
    thread.start()
    return jsonify({"job_id": job_id}), 202

@app.route('/api/download_and_analyze', methods=['POST'])
def download_and_analyze():
    data = request.get_json()
    if not data or 'video_id' not in data:
        return jsonify({"error": "Please provide 'video_id'"}), 400
    
    video_id = data['video_id']
    custom_name = data.get('custom_name')
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "running",
        "progress": "Starting download...",
        "stage": "download",
        "video_id": video_id
    }
    thread = Thread(target=download_and_analyze_task, args=(job_id, video_id, custom_name, JOBS))
    thread.start()
    return jsonify({"job_id": job_id}), 202

@app.route('/api/songs', methods=['GET'])
def get_songs():
    songs = CHIMERA_CORE.song_library.list_all_songs()
    return jsonify({"songs": songs, "total_count": len(songs)})

@app.route('/api/mashup/create', methods=['POST'])
def create_mashup():
    """
    FINAL VERSION: This endpoint now correctly handles requests from the frontend
    and triggers the REAL mashup creation task.
    """
    data = request.get_json()
    if not data or 'songs' not in data or len(data['songs']) < 2:
        return jsonify({"error": "Please provide a 'songs' list with at least two song queries."}), 400

    song_queries = [s.get('query') for s in data['songs'] if s.get('query')]
    user_suggestions = data.get('user_suggestions')
    
    if len(song_queries) < 2:
        return jsonify({"error": "Please provide at least two valid song queries."}), 400

    song_paths = []
    for title in song_queries:
        # Try fuzzy matching for better user experience
        logger.info(f"Looking for song: '{title}'")
        info = CHIMERA_CORE.song_library.find_song_by_query(title)
        logger.info(f"Found song info: {info}")
        if not info:
            # List available songs for debugging
            available_songs = CHIMERA_CORE.song_library.list_all_songs()
            available_titles = [s.get('title', 'No title') for s in available_songs]
            logger.error(f"Song '{title}' not found. Available songs: {available_titles}")
            return jsonify({"error": f"Song '{title}' not found in library. Available songs: {available_titles}"}), 404
        song_paths.append(info['source_file_path'])

    mashup_style = data.get('style', 'energetic')
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "queued", 
        "progress": "Queued for professional mashup creation...",
        "stage": "initialization"
    }
    
    # Call the REAL mashup creation task
    thread = Thread(target=create_mashup_task, args=(job_id, song_paths, mashup_style, JOBS))
    thread.start()
    
    return jsonify({"job_id": job_id}), 202

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route('/api/mashup/status/<job_id>', methods=['GET'])
def get_mashup_status(job_id):
    """Frontend-compatible mashup status endpoint."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    # Convert to frontend-expected format
    response = dict(job)
    response["job_id"] = job_id
    
    # Map status names to frontend expectations
    if job.get("status") == "completed":
        response["status"] = "complete"
    elif job.get("status") == "failed":
        response["status"] = "error"
    else:
        response["status"] = job.get("status", "processing")
    
    return jsonify(response)

@app.route('/api/mashup/audio/<filename>', methods=['GET'])
def get_mashup_audio(filename):
    return send_from_directory('workspace/mashups', filename)
    
@app.route('/api/mashup/recipe/<filename>', methods=['GET'])
def get_mashup_recipe(filename):
    return send_from_directory('workspace/mashups', filename)

@app.route('/api/mashup/revise', methods=['POST'])
def revise_mashup():
    data = request.get_json()
    if not data or 'original_recipe' not in data or 'revision_request' not in data:
        return jsonify({
            "error": "Please provide 'original_recipe' and 'revision_request' fields"
        }), 400
    
    original_recipe = data['original_recipe']
    revision_request = data['revision_request']
    
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "running",
        "progress": "Analyzing revision request...",
        "stage": "revision_analysis",
        "revision_request": revision_request
    }
    
    thread = Thread(target=revise_mashup_task, args=(job_id, original_recipe, revision_request, JOBS))
    thread.start()
    
    return jsonify({"job_id": job_id}), 202
    
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        system_status = CHIMERA_CORE.get_system_status()
        active_jobs = len([j for j in JOBS.values() if j["status"] == "running"])
        system_status["statistics"]["active_jobs"] = active_jobs
        return jsonify(system_status), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.errorhandler(500)
def handle_500(e):
    logger.error(f"Internal Server Error: {e}")
    traceback.print_exc()
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Not Found", "message": "The requested resource was not found"}), 404

if __name__ == '__main__':
    print("Initializing Project Chimera (Production Mode)...")
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, port=port, host='0.0.0.0')