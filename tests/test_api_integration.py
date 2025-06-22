import requests
import soundfile as sf
import io
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api_integration_test")

def test_tts_api():
    """
    Tests the /tts API endpoint to ensure it returns a valid WAV file.
    Assumes the TTS API server is running.
    """
    # 1. Define API endpoint and test data
    api_url = "http://127.0.0.1:8000/tts"
    test_data = {
        "text": "This is an integration test to confirm the API returns a valid wave file.",
        "voice_name": "biden",  # Using a standard voice
        "language": "en"
    }
    output_filename = "test_api_output.wav"
    output_path = os.path.join(os.path.dirname(__file__), output_filename)

    logger.info(f"Testing TTS API at {api_url}...")
    logger.info(f"Request data: {test_data}")

    # 2. Send request to the API
    try:
        response = requests.post(api_url, json=test_data, timeout=120) # 2 minute timeout for model loading
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to the TTS API: {e}")
        logger.error("Please ensure the TTS API server is running before executing this test.")
        return

    # 3. Check the response
    if response.headers.get('content-type') != 'audio/wav':
        logger.error(f"Unexpected content type: {response.headers.get('content-type')}")
        logger.error(f"Response content: {response.text}")
        return

    audio_bytes = response.content
    if not audio_bytes:
        logger.error("API returned an empty response.")
        return

    logger.info(f"Received {len(audio_bytes)} bytes of audio data.")

    # 4. Verify the audio data is a valid WAV file and save it
    try:
        logger.info(f"Verifying and saving audio to {output_path}...")
        # Use soundfile to read the bytes, which validates the WAV format
        with io.BytesIO(audio_bytes) as buffer:
            data, samplerate = sf.read(buffer)
        
        # If successful, write the original bytes to a file for inspection
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
            
        logger.info(f"Successfully verified and saved WAV file to '{output_path}'")
        logger.info(f"Audio details - Sample Rate: {samplerate}, Samples: {len(data)}")

    except Exception as e:
        logger.error(f"The returned audio is not a valid WAV file: {e}", exc_info=True)
        # Save the invalid response for debugging
        invalid_output_path = os.path.join(os.path.dirname(__file__), "invalid_api_response.bin")
        with open(invalid_output_path, "wb") as f:
            f.write(audio_bytes)
        logger.error(f"Invalid response saved to {invalid_output_path}")

if __name__ == "__main__":
    test_tts_api() 