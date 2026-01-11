#!/usr/bin/env python3
"""
Export HiFT vocoder to TensorRT engine.

This script exports the HiFT vocoder model to ONNX format and then builds
a TensorRT engine for accelerated inference.

Usage:
    python scripts/export_hift_tensorrt.py \
        --hift-path /app/GLM-TTS/ckpt/hift/hift.pt \
        --output /app/data/hift.engine \
        --precision fp16

The resulting engine file is GPU architecture-specific and must be built
on the same GPU where it will be used for inference.
"""

import argparse
import logging
import os
import sys
import tempfile

# Add app to path
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/GLM-TTS")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export HiFT vocoder to TensorRT engine"
    )
    parser.add_argument(
        "--hift-path",
        type=str,
        default="/app/GLM-TTS/ckpt/hift/hift.pt",
        help="Path to HiFT checkpoint (hift.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/app/data/hift.engine",
        help="Output path for TensorRT engine"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp16",
        help="TensorRT precision mode"
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to save intermediate ONNX model (optional, uses temp file if not specified)"
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=10,
        help="Minimum mel sequence length for dynamic shapes"
    )
    parser.add_argument(
        "--opt-seq-len",
        type=int,
        default=200,
        help="Optimal mel sequence length for dynamic shapes"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1000,
        help="Maximum mel sequence length for dynamic shapes"
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=4.0,
        help="GPU memory workspace for TensorRT builder (GB)"
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export and use existing ONNX file (requires --onnx-path)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device for export"
    )
    
    args = parser.parse_args()
    
    # Import after args parsing to catch import errors early
    from app.backends.hift_tensorrt import (
        export_hift_to_onnx,
        build_tensorrt_engine,
        is_tensorrt_available,
        is_onnx_available,
    )
    
    # Check dependencies
    if not is_onnx_available():
        logger.error("ONNX is required. Install with: pip install onnx onnxruntime")
        sys.exit(1)
    
    if not is_tensorrt_available():
        logger.error("TensorRT is required. Install with: pip install tensorrt>=10.0.0")
        sys.exit(1)
    
    # Validate inputs
    if not os.path.exists(args.hift_path):
        logger.error(f"HiFT checkpoint not found: {args.hift_path}")
        sys.exit(1)
    
    if args.skip_onnx and not args.onnx_path:
        logger.error("--skip-onnx requires --onnx-path to be specified")
        sys.exit(1)
    
    if args.skip_onnx and not os.path.exists(args.onnx_path):
        logger.error(f"ONNX file not found: {args.onnx_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Export to ONNX
        if args.skip_onnx:
            onnx_path = args.onnx_path
            logger.info(f"Skipping ONNX export, using existing: {onnx_path}")
        else:
            # Use provided path or temp file
            if args.onnx_path:
                onnx_path = args.onnx_path
            else:
                # Create temp file in same dir as output for easy cleanup
                onnx_path = args.output.replace(".engine", ".onnx")
            
            logger.info("=" * 60)
            logger.info("Step 1: Exporting HiFT to ONNX")
            logger.info("=" * 60)
            export_hift_to_onnx(
                hift_checkpoint_path=args.hift_path,
                output_path=onnx_path,
                device=args.device,
            )
        
        # Step 2: Build TensorRT engine
        logger.info("=" * 60)
        logger.info("Step 2: Building TensorRT Engine")
        logger.info("=" * 60)
        build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=args.output,
            precision=args.precision,
            min_seq_len=args.min_seq_len,
            opt_seq_len=args.opt_seq_len,
            max_seq_len=args.max_seq_len,
            workspace_gb=args.workspace_gb,
        )
        
        # Cleanup temp ONNX file if we created one
        if not args.onnx_path and os.path.exists(onnx_path):
            logger.info(f"Keeping intermediate ONNX file: {onnx_path}")
        
        logger.info("=" * 60)
        logger.info("SUCCESS!")
        logger.info(f"TensorRT engine saved to: {args.output}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("To use the TensorRT vocoder, set these environment variables:")
        logger.info(f"  export GLM_TTS_VOCODER_ENGINE=tensorrt")
        logger.info(f"  export GLM_TTS_TENSORRT_ENGINE_PATH={args.output}")
        
    except Exception as e:
        logger.exception(f"Failed to build TensorRT engine: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
