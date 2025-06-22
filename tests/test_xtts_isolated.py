import logging
import soundfile as sf
import sys
import os
import numpy as np

# Add project root and app directory to path to import TTSService and log_util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from app.xtts_service import TTSService

def main():
    """
    A simple script to test the TTSService in isolation.
    """
    # 1. Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("test_xtts_isolated")
    logger.info("Starting XTTS service test...")

    # 2. Initialize TTSService
    try:
        logger.info("Initializing TTSService...")
        tts_service = TTSService(logger=logger)
        tts_service.load_voices()
        logger.info("TTSService initialized and voices loaded.")
    except Exception as e:
        logger.error(f"Failed to initialize TTSService: {e}", exc_info=True)
        return

    # 3. Define test parameters
    test_text = "Hello, this is a test of the text to speech system. I should be able to generate this audio without any issues."
    
    available_voices = tts_service.get_voices()
    if not available_voices:
        logger.error("No voices available. Make sure you have voices in the 'data/voices' directory.")
        return

    voice_to_use = available_voices[0]
    output_filename = f"test_output_{voice_to_use}.wav"
    language = "en"

    logger.info(f"Available voices: {available_voices}")
    logger.info(f"Using voice: '{voice_to_use}'")
    logger.info(f"Test text: '{test_text}'")
    logger.info(f"Output file in tests/ directory: '{output_filename}'")

    # 4. Generate speech
    try:
        logger.info("Generating speech...")
        audio_data, sample_rate = tts_service.generate_speech(
            text=test_text,
            voice_name=voice_to_use,
            language=language
        )
        logger.info(f"Speech generated. Audio data type: {type(audio_data)}, sample rate: {sample_rate}")
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)
        logger.info(f"Audio data shape: {audio_data.shape}")

    except Exception as e:
        logger.error(f"Failed to generate speech: {e}", exc_info=True)
        return

    # 5. Save the output audio
    try:
        output_path = os.path.join(os.path.dirname(__file__), output_filename)
        logger.info(f"Saving audio to {output_path}...")
        sf.write(output_path, audio_data, sample_rate)
        logger.info(f"Successfully generated speech and saved to '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main() 