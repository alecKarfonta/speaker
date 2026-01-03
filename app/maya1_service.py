import os
import soundfile as sf
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import yaml
import torch
import random
import tempfile
import time
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC


class Maya1TTSService:
    """Maya1 Text-to-Speech Service Implementation"""
    
    # Token definitions from Maya1 documentation
    CODE_START_TOKEN_ID = 128257
    CODE_END_TOKEN_ID = 128258
    CODE_TOKEN_OFFSET = 128266
    SNAC_MIN_ID = 128266
    SNAC_MAX_ID = 156937
    SNAC_TOKENS_PER_FRAME = 7
    
    SOH_ID = 128259
    EOH_ID = 128260
    SOA_ID = 128261
    BOS_ID = 128000
    TEXT_EOT_ID = 128009
    
    def __init__(self, logger: Optional[logging.Logger] = None, model_name: str = "maya-research/maya1"):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if not isinstance(self.logger, logging.Logger):
            raise TypeError("Logger must be an instance of logging.Logger")
        
        self.logger.debug(f"({logger = }, {model_name = })")
        self.config = self.load_config()
        self.logger.debug(f"(): Config = {self.config}")
        
        # Model configuration
        self.model_name = self._determine_model_name(model_name)
        self.logger.debug(f"(): Model name: {self.model_name}")
        
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        
        # Supported emotions from Maya1 documentation
        self.supported_emotions = [
            'laugh', 'laugh_harder', 'sigh', 'whisper', 'angry', 'giggle', 
            'chuckle', 'gasp', 'cry', 'snort', 'scream', 'rage', 'sob',
            'excited', 'nervous', 'confused', 'surprised', 'disappointed',
            'relieved', 'proud', 'embarrassed'
        ]
        
        self.logger.debug(f"Initializing Maya1TTSService with model: {self.model_name}")
        self.initialize_model()
    
    def _determine_model_name(self, model_name: str) -> str:
        """Determine the model name from various sources"""
        if model_name:
            return model_name
        elif os.getenv("MAYA1_MODEL_NAME"):
            return str(os.getenv("MAYA1_MODEL_NAME"))
        elif self.config.get("model_name"):
            return self.config["model_name"]
        else:
            return "maya-research/maya1"  # Default Maya1 model
    
    def load_config(self, config_path: str = "maya1_config.yaml") -> dict:
        """Load the config file with defaults"""
        self.logger.debug(f"Loading config from {config_path}")
        
        # Default configuration for Maya1
        default_config = {
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "device": "auto",
            "torch_dtype": "bfloat16",
            "snac_model": "hubertsiuzdak/snac_24khz",
            "sample_rate": 24000,
            "max_audio_duration": 30.0,
            "expected_wpm": 150  # Words per minute for duration estimation
        }
        
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return default_config
            
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            # Merge with defaults
            default_config.update(config)
            return default_config
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse config file: {e}, using defaults")
            return default_config
    
    def _set_seeds(self, seed: int = 42):
        """Set deterministic seeds"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def initialize_model(self):
        """Initialize Maya1 model and tokenizer"""
        self.logger.info("Starting Maya1 model initialization")
        start_time = time.time()
        
        try:
            # Load tokenizer
            self.logger.info("Loading Maya1 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Determine torch dtype
            torch_dtype = getattr(torch, self.config.get("torch_dtype", "bfloat16"))
            
            # Load model
            self.logger.info("Loading Maya1 model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.config.get("device", "auto"),
                trust_remote_code=True
            )
            
            # Load SNAC model for audio decoding
            self.logger.info("Loading SNAC audio codec...")
            snac_model_name = self.config.get("snac_model", "hubertsiuzdak/snac_24khz")
            self.snac_model = SNAC.from_pretrained(snac_model_name).eval()
            if torch.cuda.is_available():
                self.snac_model = self.snac_model.cuda()
            
            init_time = time.time() - start_time
            self.logger.info(f"✅ Maya1 model initialized successfully in {init_time:.2f} seconds")
            self.logger.info(f"Model: {self.model_name}")
            self.logger.info(f"Parameters: ~3B")
            self.logger.info(f"Device: {self.model.device if hasattr(self.model, 'device') else 'Unknown'}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Maya1 model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def build_prompt(self, description: str, text: str) -> str:
        """Build formatted prompt for Maya1"""
        soh_token = self.tokenizer.decode([self.SOH_ID])
        eoh_token = self.tokenizer.decode([self.EOH_ID])
        soa_token = self.tokenizer.decode([self.SOA_ID])
        sos_token = self.tokenizer.decode([self.CODE_START_TOKEN_ID])
        eot_token = self.tokenizer.decode([self.TEXT_EOT_ID])
        bos_token = self.tokenizer.bos_token
        
        formatted_text = f'<description="{description}"> {text}'
        
        prompt = (
            soh_token + bos_token + formatted_text + eot_token +
            eoh_token + soa_token + sos_token
        )
        
        return prompt
    
    def extract_snac_codes(self, token_ids: list) -> list:
        """Extract SNAC codes from generated tokens"""
        try:
            eos_idx = token_ids.index(self.CODE_END_TOKEN_ID)
        except ValueError:
            eos_idx = len(token_ids)
        
        snac_codes = [
            token_id for token_id in token_ids[:eos_idx]
            if self.SNAC_MIN_ID <= token_id <= self.SNAC_MAX_ID
        ]
        
        return snac_codes
    
    def unpack_snac_from_7(self, snac_tokens: list) -> list:
        """Unpack 7-token SNAC frames to 3 hierarchical levels"""
        if snac_tokens and snac_tokens[-1] == self.CODE_END_TOKEN_ID:
            snac_tokens = snac_tokens[:-1]
        
        frames = len(snac_tokens) // self.SNAC_TOKENS_PER_FRAME
        snac_tokens = snac_tokens[:frames * self.SNAC_TOKENS_PER_FRAME]
        
        if frames == 0:
            return [[], [], []]
        
        l1, l2, l3 = [], [], []
        
        for i in range(frames):
            slots = snac_tokens[i*7:(i+1)*7]
            l1.append((slots[0] - self.CODE_TOKEN_OFFSET) % 4096)
            l2.extend([
                (slots[1] - self.CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - self.CODE_TOKEN_OFFSET) % 4096,
            ])
            l3.extend([
                (slots[2] - self.CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - self.CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - self.CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - self.CODE_TOKEN_OFFSET) % 4096,
            ])
        
        return [l1, l2, l3]
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate expected audio duration for text"""
        word_count = len(text.split())
        wpm = self.config.get("expected_wpm", 150)
        duration = (word_count / wpm) * 60 + 1.0  # Add 1s buffer
        return min(duration, self.config.get("max_audio_duration", 30.0))
    
    def _validate_and_limit_audio(self, wav: np.ndarray, text: str, sample_rate: int = 24000) -> np.ndarray:
        """Validate and limit audio duration"""
        if wav.size == 0:
            raise ValueError("Generated audio is empty")
        
        duration = len(wav) / sample_rate
        expected_duration = self._estimate_duration(text)
        max_duration = max(expected_duration * 3, 30.0)  # Allow 3x expected or min 30s
        
        if duration > max_duration:
            self.logger.warning(f"Audio too long ({duration:.2f}s), expected ~{expected_duration:.2f}s")
        
        return wav
    
    def generate_speech(
        self,
        text: str,
        description: str,
        max_new_tokens: int = -1,
        temperature: float = -1,
        top_k: int = -1,
        top_p: float = -1
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text with voice description"""
        
        # Validate inputs
        if not text.strip():
            raise ValueError("Text cannot be empty")
        if not description.strip():
            raise ValueError("Voice description cannot be empty")
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        # Set parameters from config if not provided
        max_new_tokens = max_new_tokens if max_new_tokens != -1 else self.config.get("max_new_tokens", 2048)
        temperature = temperature if temperature != -1 else self.config.get("temperature", 0.7)
        top_k = top_k if top_k != -1 else self.config.get("top_k", 50)
        top_p = top_p if top_p != -1 else self.config.get("top_p", 0.9)
        
        self.logger.debug(f"Generating speech: '{text[:50]}...' with voice: '{description[:50]}...'")
        
        # Build prompt
        prompt = self.build_prompt(description, text)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.CODE_END_TOKEN_ID
            )
        
        # Extract generated tokens (remove input tokens)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:].tolist()
        
        # Extract SNAC codes
        snac_codes = self.extract_snac_codes(generated_tokens)
        
        if not snac_codes:
            raise RuntimeError("No valid SNAC codes generated")
        
        # Unpack SNAC codes
        snac_levels = self.unpack_snac_from_7(snac_codes)
        
        # Convert to tensors with proper shape for SNAC decoder
        # SNAC expects tensors with shape (batch_size, sequence_length)
        snac_tensors = []
        for i, level in enumerate(snac_levels):
            if len(level) == 0:
                # Create empty tensor with proper shape (1, 0)
                tensor = torch.zeros((1, 0), dtype=torch.long)
            else:
                # Reshape to (1, sequence_length) for batch dimension
                tensor = torch.tensor(level, dtype=torch.long).unsqueeze(0)
            
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            snac_tensors.append(tensor)
            self.logger.debug(f"Level {i} tensor shape: {tensor.shape}, values: {len(level)}")
        
        # Decode audio using SNAC
        with torch.no_grad():
            try:
                audio_tensor = self.snac_model.decode(snac_tensors)
            except Exception as e:
                self.logger.error(f"SNAC decode error: {e}")
                self.logger.error(f"Tensor shapes: {[t.shape for t in snac_tensors]}")
                raise
        
        # Convert to numpy array
        audio_data = audio_tensor.cpu().numpy().squeeze()
        sample_rate = self.config.get("sample_rate", 24000)
        
        # Validate and potentially limit audio
        audio_data = self._validate_and_limit_audio(audio_data, text, sample_rate)
        
        return audio_data, sample_rate
    
    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotion tags"""
        return self.supported_emotions.copy()
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "parameters": "~3B",
            "architecture": "Llama-style transformer with SNAC codec",
            "audio_quality": "24 kHz, mono",
            "streaming_rate": "~0.98 kbps",
            "license": "Apache 2.0 (Open Source)",
            "emotions_supported": len(self.supported_emotions),
            "device": str(self.model.device) if self.model and hasattr(self.model, 'device') else 'Unknown'
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Maya1 TTS Service")
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument("--description", default="Female, in her 30s with an American accent, warm and friendly tone", 
                       help="Voice description")
    parser.add_argument("--output", default="maya1_output.wav", help="Output audio file (default: maya1_output.wav)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens (default: 2048)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize Maya1 TTS service
        logger.info("Initializing Maya1 TTS service...")
        maya1_service = Maya1TTSService(logger=logger)
        
        # Get model info
        model_info = maya1_service.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        # Generate speech
        logger.info(f"Generating speech for text: '{args.text}'")
        logger.info(f"Voice description: '{args.description}'")
        
        start_time = time.time()
        audio_data, sample_rate = maya1_service.generate_speech(
            text=args.text,
            description=args.description,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )
        generation_time = time.time() - start_time
        
        # Save audio to file
        logger.info(f"Saving audio to {args.output}")
        sf.write(args.output, audio_data, sample_rate)
        
        logger.info(f"Successfully generated speech in {generation_time:.2f}s and saved to {args.output}")
        logger.info(f"Audio duration: {len(audio_data) / sample_rate:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

