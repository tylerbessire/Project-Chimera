# ==============================================================================
# FILE: app.py (v7.0 - Studio Grade Update)
# ==============================================================================
#
# MAJOR UPDATES:
# - Simplified endpoints to reflect the new, more powerful workflow.
# - Added song library management endpoints (`/songs`, `/songs/compatible`).
# - The `/create` endpoint now takes file paths directly, abstracting away
#   the complex analysis and rendering pipeline into a single background task.
# - Improved job status reporting with more granular progress steps.
#
# ==============================================================================

import os
import uuid
import traceback
from threading import Thread
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from tasks import create_mashup_task, revise_mashup_task
from song_library import SongLibrary

# Load environment variables from a .env file if present
load_dotenv()

app = Flask(__name__)
CORS(app)

# In-memory job tracking
JOBS = {}
SONG_LIBRARY = SongLibrary()

# --- API Endpoints ---

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """Handles uploading audio files to the source directory."""
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
        if file:
            filepath = os.path.join(SONG_LIBRARY.source_dir, file.filename)
            file.save(filepath)
            
            # Add to library which also triggers analysis
            SONG_LIBRARY.add_song(filepath)
            uploaded_files.append(file.filename)
            
    return jsonify({
        "message": f"Successfully uploaded and analyzed {len(uploaded_files)} files.",
        "files": uploaded_files
    }), 201

@app.route('/api/songs', methods=['GET'])
def get_songs():
    """Returns a list of all songs in the library."""
    return jsonify(SONG_LIBRARY.list_all_songs())

@app.route('/api/songs/<song_id>/compatible', methods=['GET'])
def get_compatible_songs(song_id):
    """Finds and returns songs compatible with the given song_id."""
    compatible = SONG_LIBRARY.find_compatible_songs(song_id)
    return jsonify(compatible)


@app.route('/api/create', methods=['POST'])
def create_mashup():
    """
    Kicks off a new mashup creation job.
    Expects a JSON body with a list of song titles to mash up.
    """
    data = request.get_json()
    if not data or 'song_titles' not in data or len(data['song_titles']) < 2:
        return jsonify({"error": "Please provide a 'song_titles' list with at least two song titles."}), 400

    song_titles = data['song_titles']
    song_paths = []
    for title in song_titles:
        info = SONG_LIBRARY.get_song_info(title)
        if not info:
            return jsonify({"error": f"Song '{title}' not found in library."}), 404
        song_paths.append(info['source_file_path'])

    mashup_style = data.get('style', 'Energetic EDM (like Kill mR Dj)')
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "progress": "Waiting to start..."}
    
    thread = Thread(target=create_mashup_task, args=(job_id, song_paths, mashup_style, JOBS))
    thread.start()
    
    return jsonify({"job_id": job_id}), 202

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Returns the status of a background job."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route('/api/mashup/audio/<filename>', methods=['GET'])
def get_mashup_audio(filename):
    """Serves the final rendered mashup audio file."""
    return send_from_directory('workspace/mashups', filename)

@app.route('/api/mashup/recipe/<filename>', methods=['GET'])
def get_mashup_recipe(filename):
    """Serves the recipe JSON file for a mashup."""
    return send_from_directory('workspace/mashups', filename)


@app.errorhandler(500)
def handle_500(e):
    traceback.print_exc()
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == '__main__':
    # On first run, scan the source audio directory and add any songs to the library
    print("Scanning for existing songs...")
    for filename in os.listdir(SONG_LIBRARY.source_dir):
        if filename.lower().endswith(('.wav', '.mp3', '.aiff', '.flac')):
             filepath = os.path.join(SONG_LIBRARY.source_dir, filename)
             SONG_LIBRARY.add_song(filepath)

    print("\n🎵 === STUDIO-GRADE MASHUP v7.0 ===")
    print("   - AI Creative Director (Luna/OpenAI)")
    print("   - AI Audio Engineer (Claude/Anthropic)")
    print("   - Professional Audio Effects by Pedalboard")
    print("   - High-Quality Time Stretching by Rubberband")
    print("   - AI Stem Separation by Spleeter")
    print("========================================\n")
    app.run(debug=True, port=5001)

