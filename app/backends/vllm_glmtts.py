"""
vLLM wrapper for GLM-TTS LLM inference.

Provides high-performance inference using vLLM's optimized engine
while maintaining compatibility with GLM-TTS's custom embedding and
token handling requirements.

Usage:
    Set GLM_TTS_ENGINE=vllm environment variable to enable.
"""

import os
from typing import Dict, List, Optional, Tuple
import torch

# Conditional import - vLLM is optional
try:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None


class VLLMGLMTTSWrapper:
    """
    vLLM-based wrapper for GLM-TTS LLM inference.
    
    Replaces the standard HuggingFace LlamaForCausalLM with vLLM's
    optimized inference engine for ~2x speedup.
    
    Key differences from HuggingFace:
    - Uses PagedAttention for efficient KV cache management
    - Optimized CUDA kernels for faster generation
    - Requires different input format (token IDs, not embeddings directly)
    
    Note: vLLM's prompt_embeds support requires vLLM >= 0.6.0
    """
    
    def __init__(
        self,
        model_path: str,
        special_token_ids: Dict[str, int],
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.45,  # Lower to leave room for Flow model + vocoder
        max_model_len: int = 4096,
        quantization: Optional[str] = None,  # 'fp8', 'awq', 'gptq', or None
        logger=None,
    ):
        """
        Initialize vLLM engine with GLM-TTS model.
        
        Args:
            model_path: Path to the GLM-TTS LLM checkpoint
            special_token_ids: Dict with 'ats', 'ate', 'boa', 'eoa', 'pad' token IDs
            dtype: Model dtype ('float16', 'bfloat16', 'float32')
            gpu_memory_utilization: Fraction of GPU memory for KV cache
            max_model_len: Maximum sequence length
            quantization: Quantization method ('fp8' for on-the-fly FP8 on Ada/Hopper/Blackwell,
                          'awq'/'gptq' for pre-quantized models, or None for no quantization)
            logger: Optional logger instance
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm>=0.6.0"
            )
        
        self.logger = logger
        self.special_token_ids = special_token_ids
        self.ats = special_token_ids['ats']
        self.ate = special_token_ids['ate']
        self.boa = special_token_ids['boa']
        self.eoa = special_token_ids['eoa']
        self.pad = special_token_ids['pad']
        
        # Map dtype string to torch dtype
        dtype_map = {
            "float16": "float16",
            "fp16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float32": "float32",
            "fp32": "float32",
        }
        vllm_dtype = dtype_map.get(dtype.lower(), "float16")
        
        if self.logger:
            self.logger.info(f"Initializing vLLM engine with model: {model_path}")
            self.logger.info(f"vLLM config: dtype={vllm_dtype}, gpu_util={gpu_memory_utilization}, quantization={quantization}")
        
        # Note: Multi-GPU with VLLM_DEVICE is complex because vLLM spawns subprocesses.
        # For now, use single GPU with lower gpu_memory_utilization.
        # TODO: Implement proper multi-GPU by running vLLM in a separate process.
        
        # Build LLM kwargs
        llm_kwargs = {
            "model": model_path,
            "dtype": vllm_dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": True,
            "skip_tokenizer_init": True,  # We provide token IDs directly
            "enable_prefix_caching": False,  # Disable for variable-length prompts
        }
        
        # Add quantization if specified (fp8 for Ada/Hopper/Blackwell, awq/gptq for pre-quantized)
        if quantization and quantization.lower() != "none":
            llm_kwargs["quantization"] = quantization.lower()
            if self.logger:
                self.logger.info(f"Enabling vLLM quantization: {quantization}")
        
        # Initialize vLLM engine
        self.llm = LLM(**llm_kwargs)
        
        if self.logger:
            self.logger.info("vLLM engine initialized successfully")
    
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        beam_size: int = 1,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        sample_method: str = "ras",
        spk: str = None,
        temperature: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.15,  # vLLM default: 1.0=no penalty, >1.0=penalize
    ) -> torch.Tensor:
        """
        Generate speech tokens using vLLM.
        
        Matches the GLMTTS.inference() signature for drop-in replacement.
        
        Args:
            text: Input text token tensor [1, seq_len]
            text_len: Length of input text
            prompt_text: Prompt text tensor [1, seq_len]
            prompt_text_len: Length of prompt text
            prompt_speech_token: Prompt speech token tensor [1, seq_len]
            prompt_speech_token_len: Length of prompt speech tokens
            beam_size: Beam size (vLLM uses different sampling approach)
            sampling: Top-k value
            max_token_text_ratio: Max generation length multiplier
            min_token_text_ratio: Min generation length multiplier
            sample_method: 'ras' or 'topk' (mapped to vLLM sampling)
            spk: Speaker key (not used in PRETRAIN mode)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty (vLLM: 1.0=none, >1.0=penalize)
            
        Returns:
            torch.Tensor: Generated audio tokens (shifted by ATS offset) [1, num_tokens]
        """
        device = text.device
        
        # IMPORTANT: vLLM uses multiplicative repetition_penalty where:
        #   1.0 = no penalty, >1.0 = penalize repetition, <1.0 = ENCOURAGE repetition
        # The transformers backend uses 0.1 as default (additive penalty).
        # If we receive values <=1.0 (like 0.1 from transformers), remap to sensible vLLM value.
        if repetition_penalty <= 1.0:
            if self.logger:
                self.logger.debug(
                    f"Remapping repetition_penalty {repetition_penalty} -> 1.15 for vLLM "
                    "(values <=1.0 encourage repetition in vLLM)"
                )
            repetition_penalty = 1.15  # Moderate penalty to prevent loops
        
        # 1. Preprocess Prompt Tokens (add ATS offset)
        if prompt_speech_token_len != 0 and prompt_text_len != 0:
            prompt_speech_token = prompt_speech_token + self.ats
        
        # 2. Build input token sequence (matching GLMTTS format)
        # Format: [prompt_text, text, BOA, prompt_speech_token]
        boa_tensor = torch.tensor([self.boa], device=device).unsqueeze(0)
        
        input_ids = torch.cat([
            prompt_text,
            text,
            boa_tensor,
            prompt_speech_token
        ], dim=1).to(torch.long)
        
        # Convert to list for vLLM
        prompt_token_ids = input_ids.squeeze(0).tolist()
        
        # 3. Calculate generation bounds
        text_length = text_len.item()
        min_len = int(text_length * min_token_text_ratio)
        max_len = int(text_length * max_token_text_ratio)
        
        # 4. Configure vLLM sampling parameters
        # Map GLM-TTS sampling to vLLM parameters
        sampling_params = SamplingParams(
            max_tokens=max_len,
            min_tokens=min_len,
            temperature=temperature,
            top_k=sampling if sampling > 0 else -1,  # -1 disables top-k
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[self.eoa],  # Stop at End-of-Audio
            skip_special_tokens=False,
        )
        
        # 5. Generate with vLLM
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids)],
            sampling_params,
        )
        
        # 6. Extract generated tokens
        generated_tokens = outputs[0].outputs[0].token_ids
        
        # 7. Validate tokens are in audio range and clean degenerate patterns
        out_tokens = []
        for token in generated_tokens:
            if self.ats <= token <= self.ate:
                out_tokens.append(token)
            elif token == self.eoa:
                break  # Stop at EOA
            else:
                if self.logger:
                    self.logger.warning(
                        f"Token {token} outside audio range ({self.ats}, {self.ate})"
                    )
        
        # 8. Apply additional validation to detect and fix degenerate outputs
        out_tokens = self._validate_and_clean_tokens(out_tokens, text_length)
        
        # 9. Return tokens relative to ATS (matching GLMTTS output format)
        result = torch.tensor([out_tokens], dtype=torch.int64, device=device) - self.ats
        return result
    
    def _truncate_repetitions(self, tokens: List[int], max_consecutive: int = 20) -> List[int]:
        """
        Detect and truncate repetitive token sequences.
        
        Long runs of identical tokens indicate degenerate generation (stuck model).
        This truncates output when such patterns are detected.
        
        Args:
            tokens: List of audio tokens
            max_consecutive: Maximum allowed consecutive identical tokens
            
        Returns:
            Cleaned token list, truncated at first long repetition
        """
        if len(tokens) < max_consecutive:
            return tokens
        
        consecutive_count = 1
        last_token = tokens[0] if tokens else None
        
        for i, token in enumerate(tokens[1:], start=1):
            if token == last_token:
                consecutive_count += 1
                if consecutive_count >= max_consecutive:
                    if self.logger:
                        self.logger.warning(
                            f"Detected {consecutive_count} consecutive identical tokens "
                            f"(token={token}) at position {i}. Truncating output."
                        )
                    # Return tokens up to where repetition started
                    return tokens[:i - max_consecutive + 1]
            else:
                consecutive_count = 1
                last_token = token
        
        return tokens
    
    def _validate_and_clean_tokens(self, tokens: List[int], text_length: int) -> List[int]:
        """
        Validate and clean generated tokens, removing degenerate patterns.
        
        Args:
            tokens: List of raw audio tokens (already filtered to valid range)
            text_length: Length of input text (for ratio validation)
            
        Returns:
            Cleaned token list
        """
        if not tokens:
            return tokens
        
        original_len = len(tokens)
        
        # 1. Truncate repetitive sequences (stuck generation)
        tokens = self._truncate_repetitions(tokens, max_consecutive=20)
        
        # 2. Check token ratio (expected: 2x-20x text length for normal speech)
        # Higher ratios often indicate garbage generation
        # Configurable via GLM_TTS_MAX_TOKEN_RATIO env var (default: 20.0)
        max_safe_ratio = float(os.environ.get("GLM_TTS_MAX_TOKEN_RATIO", "20.0"))
        ratio = len(tokens) / max(text_length, 1)
        
        if ratio > max_safe_ratio:
            max_tokens = int(text_length * max_safe_ratio)
            if self.logger:
                self.logger.warning(
                    f"High token ratio {ratio:.1f}x (expected ≤{max_safe_ratio}x). "
                    f"Truncating from {len(tokens)} to {max_tokens} tokens."
                )
            tokens = tokens[:max_tokens]
        
        # Log if we modified the output
        if len(tokens) != original_len and self.logger:
            self.logger.info(
                f"Token validation: {original_len} -> {len(tokens)} tokens "
                f"({original_len - len(tokens)} removed)"
            )
        
        return tokens


def is_vllm_available() -> bool:
    """Check if vLLM is available."""
    return VLLM_AVAILABLE


def should_use_vllm() -> bool:
    """Check if vLLM should be used based on environment variable."""
    engine = os.environ.get("GLM_TTS_ENGINE", "transformers").lower()
    return engine == "vllm" and VLLM_AVAILABLE
