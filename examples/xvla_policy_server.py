#!/usr/bin/env python3
"""
XVLA Policy Server for LeRobot
=============================

This script runs a LeRobot policy server specifically configured for XVLA models.
It should be run in a separate conda environment from the validation client.

Usage:
    # Basic usage
    python examples/xvla_policy_server.py \
        --pretrained_name_or_path="path/to/your/finetuned/xvla/model" \
        --host="127.0.0.1" \
        --port=8080

    # With custom settings
    python examples/xvla_policy_server.py \
        --pretrained_name_or_path="your_username/xvla_finetuned_model" \
        --host="0.0.0.0" \
        --port=8080 \
        --fps=30 \
        --inference_latency=0.033 \
        --device="cuda"

Server Environment Setup:
    conda create -n lerobot_server python=3.10
    conda activate lerobot_server
    pip install lerobot[all]
    # Install any additional dependencies for your specific XVLA model
"""

import logging
from dataclasses import dataclass
from typing import Optional

import draccus
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve


@dataclass
class XVLAServerConfig:
    """Configuration for XVLA policy server."""
    
    # Model configuration
    pretrained_name_or_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8080
    
    # Performance configuration  
    fps: int = 30
    inference_latency: float = 0.033  # ~30 FPS
    obs_queue_timeout: float = 1.0
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path is required")
            
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def main(config: XVLAServerConfig):
    """Start the XVLA policy server."""
    
    print("="*60)
    print("XVLA POLICY SERVER")
    print("="*60)
    print(f"Model: {config.pretrained_name_or_path}")
    print(f"Device: {config.device}")
    print(f"Server: {config.host}:{config.port}")
    print(f"FPS: {config.fps}")
    print(f"Inference latency: {config.inference_latency}s")
    print("="*60)
    
    # Validate CUDA availability if requested
    if config.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        config.device = "cpu"
        
    if config.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create LeRobot server config
    server_config = PolicyServerConfig(
        host=config.host,
        port=config.port,
        fps=config.fps,
        inference_latency=config.inference_latency,
        obs_queue_timeout=config.obs_queue_timeout,
    )
    
    print(f"\nStarting server on {config.host}:{config.port}...")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # This will load the XVLA model and start the gRPC server
        serve(server_config)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        raise


if __name__ == "__main__":
    # Use draccus for CLI parsing (same as LeRobot)
    config = draccus.parse(XVLAServerConfig)
    main(config)