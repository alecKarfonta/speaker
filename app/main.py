import os
import random
import logging
import numpy as np
import io
import soundfile as sf

from pydantic import BaseModel
from pydantic import BaseModel, Field, constr
from fastapi import UploadFile, File
from fastapi import FastAPI, HTTPException

from xtts_service_v2 import TTSService
from log_util import ColoredFormatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter and handler
formatter = ColoredFormatter()
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Add handler to logger
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)
logger.addHandler(handler)

# TTS Service - Now using Coqui TTS with XTTS v2 voice cloning
tts_service = TTSService(logger=logger)

app = FastAPI(
    title="TTS API", 
    debug=True,
    description="High-quality Text-to-Speech API with XTTS v2 voice cloning"
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the TTS API", "model": tts_service.model_name}


@app.get("/voices")
async def get_voices():
    """Get list of available voices for cloning"""
    return {"voices": tts_service.get_voices()}


@app.post("/voices")
async def add_voice(voice_name: str, file: UploadFile = File(...)):
    """Add a new voice by uploading an audio file"""
    # Validate voice name (alphanumeric and underscores only)
    if not voice_name.replace('_', '').isalnum():
        raise HTTPException(
            status_code=400,
            detail="Voice name must contain only letters, numbers, and underscores"
        )
    
    # Validate file type
    if not file.filename or not (file.filename.lower().endswith('.wav') or file.filename.lower().endswith('.mp3')):
        raise HTTPException(
            status_code=400,
            detail="Only .wav and .mp3 files are supported"
        )

    voice_dir = f"data/voices/{voice_name}"
    
    # Create voice directory if it doesn't exist
    os.makedirs(voice_dir, exist_ok=True)
    
    # Find next available index
    voice_index = 1
    file_extension = os.path.splitext(file.filename)[1]
    voice_file_name = f"{voice_name}_{str(voice_index).zfill(2)}{file_extension}"
    while os.path.exists(os.path.join(voice_dir, voice_file_name)):
        voice_index += 1
        voice_file_name = f"{voice_name}_{str(voice_index).zfill(2)}{file_extension}"

    # Save file to disk
    local_save_path = os.path.join(voice_dir, voice_file_name)
    logger.info(f"Saving voice file to {local_save_path}")
    
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
            
        with open(local_save_path, "wb") as f:
            f.write(content)
            
        # Reload voices to include the new one
        tts_service.load_voices()
        
        return {"message": f"Voice '{voice_name}' uploaded successfully", "file": voice_file_name}
        
    except Exception as e:
        logger.error(f"Error saving voice file: {str(e)}", exc_info=True)
        # Clean up partial file if it exists
        if os.path.exists(local_save_path):
            os.remove(local_save_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save voice file: {str(e)}"
        )


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to convert to speech")
    voice_name: str = Field(..., description="Name of the voice to use for cloning")
    language: str = Field("en", pattern="^[a-z]{2}$", description="Two-letter language code")
    temperature: float = Field(0.8, ge=0.1, le=1.0, description="Temperature parameter for randomness")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p sampling parameter")
    # Support emotion tags for OpenAudio
    emotion: str = Field("", description="Emotion tag like (happy), (sad), (angry), etc.")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")

class TTSResponse(BaseModel):
    wav: bytes
    sample_rate: int

from fastapi.responses import Response

# TTS Generation
@app.post("/tts")
async def generate_speech(request: TTSRequest):
    try:
        # Validate voice exists
        available_voices = tts_service.get_voices()
        if request.voice_name not in available_voices:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{request.voice_name}' not found. Available voices: {', '.join(available_voices)}"
            )

        # Prepare text with emotion tags if provided
        text = request.text
        if request.emotion:
            # Add emotion tag to the beginning of the text
            emotion_tag = request.emotion if request.emotion.startswith('(') else f"({request.emotion})"
            text = f"{emotion_tag} {text}"

        logger.info(
            f"Generating speech: text='{text[:50]}...', "
            f"voice={request.voice_name}, language={request.language}"
        )

        audio, sample_rate = tts_service.generate_speech(
            text=text,
            voice_name=request.voice_name,
            language=request.language
        )

        # Debug logging
        logger.debug(f"generate_speech(): {type(audio) = }")    
        if len(audio) > 100:
            random_start_index = random.randint(0, len(audio) - 100)
            logger.debug(f"generate_speech(): {audio[random_start_index:random_start_index + 100] = }")

        if audio is None or len(audio) == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate audio"
            )
        
        # Convert audio to bytes
        audio_bytes = np.array(audio, dtype=np.float32).tobytes()

        logger.debug(f"Successfully generated audio of length: {len(audio_bytes)} bytes")
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Content-Length": str(len(audio_bytes)),
                "X-Sample-Rate": str(sample_rate),
                "X-Audio-Duration": str(len(audio)/sample_rate),
                "X-Model": tts_service.model_name,
                "X-Voice": request.voice_name
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "TTS",
        "version": version.app_version,
        "model": tts_service.model_name
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
