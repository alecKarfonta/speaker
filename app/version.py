"""
Version information for the Speaker TTS API
"""

# Application version
app_version = "1.0.0"

# Application name
app_name = "Speaker TTS API"

# Application description
app_description = "Text-to-Speech API using XTTS v2 voice cloning"

# Build information
build_date = "2024-06-22"
build_commit = "unknown"

def get_version_info():
    """Get complete version information"""
    return {
        "version": app_version,
        "name": app_name,
        "description": app_description,
        "build_date": build_date,
        "build_commit": build_commit
    } 