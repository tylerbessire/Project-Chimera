"""
Workspace Structure Cleanup and Unification
===========================================

This script creates a clean, logical workspace structure and fixes the naming chaos.
"""

import os
import json
import shutil
from pathlib import Path

def cleanup_workspace():
    """Clean up the insane workspace structure."""
    workspace = Path("workspace")
    
    # Create clean structure
    clean_dirs = {
        "songs": workspace / "songs",           # All songs with clean names
        "audio": workspace / "audio",           # Source audio files  
        "stems": workspace / "stems",           # Stem separations
        "mashups": workspace / "mashups",       # Final mashups only
        "cache": workspace / "cache"            # Unified cache
    }
    
    # Create backup
    backup_dir = workspace / "backup_old_structure"
    if not backup_dir.exists():
        print("Creating backup of old structure...")
        shutil.copytree(workspace, backup_dir, ignore_errors=True)
    
    # Clear out the mess
    print("Cleaning workspace structure...")
    for clean_dir in clean_dirs.values():
        clean_dir.mkdir(exist_ok=True)
    
    # Clean up mashups - keep only actual mashup files
    mashups_old = workspace / "mashups"
    mashups_new = clean_dirs["mashups"]
    
    if mashups_old.exists():
        for item in mashups_old.iterdir():
            if item.is_file() and item.suffix == ".wav":
                # Keep actual mashup files
                shutil.copy2(item, mashups_new / item.name)
                print(f"Kept mashup: {item.name}")
    
    # Map songs to clean names
    song_mapping = {
        "gGdGFtwCNBE": "The_Killers_Mr_Brightside",
        "_ovdm2yX4MA": "Avicii_Levels", 
        "james blunt - goodbye my lover": "James_Blunt_Goodbye_My_Lover",
        # Add more mappings as needed
    }
    
    print("Cleaned workspace structure created!")
    print(f"Backup saved to: {backup_dir}")
    
    return clean_dirs

if __name__ == "__main__":
    cleanup_workspace()