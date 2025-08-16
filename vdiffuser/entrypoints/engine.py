"""
The entry point of inference server

This file implements python APIs for the inference engine.
"""

import uvloop
import torch
import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import os
import signal
import threading
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import zmq
import zmq.asyncio
from PIL.Image import Image

from vdiffuser.server_args import ServerArgs, PortArgs
from vdiffuser.managers.pipeline_manager import PipelineManager, run_pipeline_process
from vdiffuser.managers.scheduler import run_scheduler_process
from vdiffuser.managers.shared_gpu_memory import create_shared_dict


# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)
logger = logging.getLogger(__name__)


def _launch_subprocesses(
    server_args: ServerArgs, port_args: Optional[PortArgs] = None
) -> Tuple[PipelineManager, Dict]:
    # Configure global environment
    
    #

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # Create shared dictionary for multi-process tensor sharing
    input_shared_dict = create_shared_dict()
    output_shared_dict = create_shared_dict()
    logger.info("Created shared dictionary for multi-process tensor sharing")

    pipe_reader, pipe_writer = mp.Pipe(duplex=False)
    # Launch scheduler process
    scheduler_proc = mp.Process(
        target=run_scheduler_process,
        args=(
            server_args,
            port_args,
            pipe_writer,
            input_shared_dict,
            output_shared_dict,
        ),
    )
    
    scheduler_proc.start()
    
    # Wait for scheduler to be ready and read the status
    try:
        scheduler_info = pipe_reader.recv()
        logger.info(f"Scheduler status: {scheduler_info}")
    except Exception as e:
        logger.error(f"Failed to receive scheduler status: {e}")
        scheduler_info = None
    finally:
        # Close the pipe writer to prevent broken pipe errors
        pipe_writer.close()

    # Launch pipeline manager with shared dictionary
    pipeline_manager = PipelineManager(server_args, port_args, input_shared_dict, output_shared_dict)

    return pipeline_manager, scheduler_info
