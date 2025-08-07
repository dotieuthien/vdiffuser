"""Launch the VDiffuser inference server for diffusion models."""

import os
import sys
import logging
import argparse
from typing import Optional

from vdiffuser.entrypoints.api_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

from vdiffuser.configs import SdxlConfig, QwenImageConfig, FluxConfig

# Configure logging for VDiffuser
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("vdiffuser.server")

SUPPORTED_MODEL_TYPES = {
    "sdxl": SdxlConfig,
    "flux": FluxConfig,
    "qwen_image": QwenImageConfig
}


def validate_diffusion_args(server_args) -> None:
    """Validate VDiffuser-specific server arguments."""

    # Validate model type if specified
    if hasattr(server_args, 'model_path') and server_args.model_path:
        model_type = getattr(server_args, 'model_type', None)
        if model_type and model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(SUPPORTED_MODEL_TYPES.keys())}"
            )

    # Set diffusion-specific defaults
    if not hasattr(server_args, 'max_batch_size') or server_args.max_batch_size is None:
        server_args.max_batch_size = 8  # Conservative default for diffusion models
        logger.info(
            f"Set max_batch_size to {server_args.max_batch_size} for diffusion models")

    if not hasattr(server_args, 'max_num_seqs') or server_args.max_num_seqs is None:
        server_args.max_num_seqs = 32  # Reasonable default for image generation
        logger.info(
            f"Set max_num_seqs to {server_args.max_num_seqs} for diffusion models")

    # Enable image caching by default
    if not hasattr(server_args, 'enable_cache') or server_args.enable_cache is None:
        server_args.enable_cache = True
        logger.info("Enabled image caching for diffusion pipeline stages")


def setup_vdiffuser_config(server_args) -> Optional[object]:
    """Setup VDiffuser-specific configuration based on model type."""

    model_type = getattr(server_args, 'model_type', None)
    if not model_type:
        # Try to infer from model path
        model_path = getattr(server_args, 'model_path', '')
        if 'sdxl' in model_path.lower():
            model_type = 'sdxl'
        elif 'flux' in model_path.lower():
            model_type = 'flux'
        elif 'qwen' in model_path.lower():
            model_type = 'qwen_image'

    if model_type in SUPPORTED_MODEL_TYPES:
        config_class = SUPPORTED_MODEL_TYPES[model_type]
        config = config_class()
        logger.info(f"Initialized {model_type} configuration")
        return config

    logger.warning(
        f"Unknown model type: {model_type}. Using default configuration.")
    return None


def print_vdiffuser_info(server_args) -> None:
    """Print VDiffuser server information."""
    logger.info("=" * 60)
    logger.info("🎨 VDiffuser - Diffusion Model Inference Server")
    logger.info("=" * 60)

    model_type = getattr(server_args, 'model_type', 'auto-detected')
    model_path = getattr(server_args, 'model_path', 'not specified')

    logger.info(f"Model Type: {model_type}")
    logger.info(f"Model Path: {model_path}")
    logger.info(f"Supported Models: {', '.join(SUPPORTED_MODEL_TYPES.keys())}")
    logger.info(
        f"Max Batch Size: {getattr(server_args, 'max_batch_size', 'default')}")
    logger.info(
        f"Max Sequences: {getattr(server_args, 'max_num_seqs', 'default')}")
    logger.info(
        f"Cache Enabled: {getattr(server_args, 'enable_cache', False)}")

    host = getattr(server_args, 'host', '0.0.0.0')
    port = getattr(server_args, 'port', 30000)
    logger.info(f"Server URL: http://{host}:{port}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        # Parse server arguments
        # server_args = prepare_server_args(sys.argv[1:])

        # VDiffuser-specific validation and setup
        # validate_diffusion_args(server_args)

        # Setup model configuration
        # vdiffuser_config = setup_vdiffuser_config(server_args)

        # Print server information
        # print_vdiffuser_info(server_args)

        # Launch the server with VDiffuser configuration
        logger.info("Starting VDiffuser inference server...")
        launch_server()

    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start VDiffuser server: {e}")
        sys.exit(1)
    finally:
        logger.info("Cleaning up VDiffuser server processes...")
        kill_process_tree(os.getpid(), include_parent=False)
