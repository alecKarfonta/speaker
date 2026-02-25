"""
TensorRT Accelerated HiFT Vocoder for GLM-TTS.

This module provides TensorRT-accelerated inference for the HiFT vocoder,
offering 2-3x speedup over PyTorch inference on supported NVIDIA GPUs.

Usage:
    # Build engine (one-time, GPU-specific):
    python scripts/export_hift_tensorrt.py --hift-path /path/to/hift.pt --output /path/to/hift.engine

    # Use in inference:
    export GLM_TTS_VOCODER_ENGINE=tensorrt
    export GLM_TTS_TENSORRT_ENGINE_PATH=/path/to/hift.engine
"""

import os
import logging
import pathlib
from typing import Optional, Tuple, Union

import torch
import numpy as np

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


logger = logging.getLogger(__name__)


def is_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    return TENSORRT_AVAILABLE


def is_onnx_available() -> bool:
    """Check if ONNX is available for export."""
    return ONNX_AVAILABLE


class HiFTForExport(torch.nn.Module):
    """
    Wrapper around HiFTGenerator for clean ONNX export.
    
    Flattens the inference method into a simple forward pass
    without optional arguments that complicate ONNX export.
    """
    
    def __init__(self, hift_model):
        super().__init__()
        self.model = hift_model
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ONNX export.
        
        Args:
            mel: Mel spectrogram (B, 80, T)
            
        Returns:
            Audio waveform (B, 1, T_out)
        """
        # Call inference without cache (cache_source defaults to empty)
        audio, _ = self.model.inference(mel)
        return audio


def export_hift_to_onnx(
    hift_checkpoint_path: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    opset_version: int = 17,
    device: str = "cuda",
) -> str:
    """
    Export HiFT vocoder to ONNX format.
    
    Args:
        hift_checkpoint_path: Path to hift.pt checkpoint
        output_path: Path to save ONNX model
        opset_version: ONNX opset version (17 recommended for TensorRT 10+)
        device: Device to use for export
        
    Returns:
        Path to exported ONNX model
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX is required for export. Install with: pip install onnx onnxruntime")
    
    # Import HiFT components
    import sys
    sys.path.insert(0, "/app/GLM-TTS")
    from utils.hift_util import HiFTInference
    
    logger.info(f"Loading HiFT model from {hift_checkpoint_path}...")
    hift = HiFTInference(str(hift_checkpoint_path), device=device)
    
    # Remove weight normalization for cleaner export
    logger.info("Removing weight normalization for export...")
    try:
        hift.model.remove_weight_norm()
    except Exception as e:
        logger.warning(f"Could not remove weight normalization: {e}")
    
    # Wrap model for export
    export_model = HiFTForExport(hift.model)
    export_model.eval()
    
    # Create dummy input - (batch=1, channels=80, time=100)
    # Using a representative sequence length
    dummy_mel = torch.randn(1, 80, 100, device=device)
    
    logger.info(f"Exporting to ONNX: {output_path}...")
    
    torch.onnx.export(
        export_model,
        (dummy_mel,),
        str(output_path),
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={
            "mel": {0: "batch", 2: "seq_len"},
            "audio": {0: "batch", 2: "audio_len"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    # Validate exported model
    logger.info("Validating ONNX model...")
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    
    logger.info(f"ONNX export successful: {output_path}")
    return str(output_path)


def build_tensorrt_engine(
    onnx_path: Union[str, pathlib.Path],
    engine_path: Union[str, pathlib.Path],
    precision: str = "fp16",
    min_seq_len: int = 10,
    opt_seq_len: int = 200,
    max_seq_len: int = 1000,
    workspace_gb: float = 4.0,
) -> str:
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ("fp32", "fp16")
        min_seq_len: Minimum expected mel sequence length
        opt_seq_len: Optimal (most common) mel sequence length
        max_seq_len: Maximum expected mel sequence length
        workspace_gb: GPU memory workspace in GB
        
    Returns:
        Path to built TensorRT engine
    """
    if not TENSORRT_AVAILABLE:
        raise ImportError("TensorRT is required. Install with: pip install tensorrt>=10.0.0")
    
    logger.info(f"Building TensorRT engine from {onnx_path}...")
    logger.info(f"Precision: {precision}, Seq lengths: min={min_seq_len}, opt={opt_seq_len}, max={max_seq_len}")
    
    # Create builder and network
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    
    # Create network with explicit batch
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # Parse ONNX
    parser = trt.OnnxParser(network, trt_logger)
    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    
    # Set precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")
        else:
            logger.warning("FP16 not supported on this platform, using FP32")
    
    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # Set dynamic shape ranges for mel input (batch, 80, seq_len)
    profile.set_shape(
        "mel",
        min=(1, 80, min_seq_len),
        opt=(1, 80, opt_seq_len),
        max=(1, 80, max_seq_len),
    )
    config.add_optimization_profile(profile)
    
    # Build engine
    logger.info("Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine
    with open(str(engine_path), "wb") as f:
        f.write(serialized_engine)
    
    logger.info(f"TensorRT engine saved: {engine_path}")
    return str(engine_path)


class HiFTTensorRT:
    """
    TensorRT inference wrapper for HiFT vocoder.
    
    Drop-in replacement for HiFTInference with identical interface.
    """
    
    def __init__(
        self,
        engine_path: Union[str, pathlib.Path],
        device: str = "cuda",
    ):
        """
        Initialize TensorRT vocoder.
        
        Args:
            engine_path: Path to serialized TensorRT engine
            device: CUDA device (e.g., "cuda", "cuda:0")
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is required. Install with: pip install tensorrt>=10.0.0")
        
        self.device = device
        self.device_id = 0
        if ":" in device:
            self.device_id = int(device.split(":")[1])
        
        self.sample_rate = 24000
        
        logger.info(f"Loading TensorRT engine from {engine_path}...")
        
        # Load engine
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(str(engine_path), "rb") as f:
            self.engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers (will be resized dynamically)
        self._input_buffer = None
        self._output_buffer = None
        self._stream = torch.cuda.Stream(device=self.device)
        
        logger.info("TensorRT vocoder initialized successfully")
    
    def _allocate_buffers(self, mel_shape: Tuple[int, int, int]):
        """Allocate/reallocate buffers for given input shape."""
        batch, channels, seq_len = mel_shape
        
        # Set input shape in context
        self.context.set_input_shape("mel", mel_shape)
        
        # Get output shape
        output_shape = self.context.get_tensor_shape("audio")
        
        # Allocate input buffer if needed
        if self._input_buffer is None or self._input_buffer.shape != mel_shape:
            self._input_buffer = torch.empty(
                mel_shape, dtype=torch.float32, device=self.device
            )
        
        # Allocate output buffer if needed
        if self._output_buffer is None or self._output_buffer.shape != tuple(output_shape):
            self._output_buffer = torch.empty(
                tuple(output_shape), dtype=torch.float32, device=self.device
            )
    
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Run TensorRT inference.
        
        Args:
            mel: Mel spectrogram (B, 80, T)
            
        Returns:
            Audio waveform (B, 1, T_out)
        """
        # Ensure input is on correct device and dtype
        mel = mel.to(device=self.device, dtype=torch.float32).contiguous()
        
        # Allocate buffers for this input shape
        self._allocate_buffers(tuple(mel.shape))
        
        # Copy input to buffer
        self._input_buffer.copy_(mel)
        
        # Set tensor addresses
        self.context.set_tensor_address("mel", self._input_buffer.data_ptr())
        self.context.set_tensor_address("audio", self._output_buffer.data_ptr())
        
        # Execute
        with torch.cuda.stream(self._stream):
            success = self.context.execute_async_v3(self._stream.cuda_stream)
        
        self._stream.synchronize()
        
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return self._output_buffer.clone()
    
    def __del__(self):
        """Cleanup TensorRT resources."""
        if hasattr(self, "context") and self.context is not None:
            del self.context
        if hasattr(self, "engine") and self.engine is not None:
            del self.engine
