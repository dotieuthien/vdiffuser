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

from vdiffuser.server_args import ServerArgs
from vdiffuser.managers.pipeline_manager import PipelineManager, run_pipeline_process
from vdiffuser.managers.scheduler import run_scheduler_process

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)


def _launch_subprocesses(
    server_args: ServerArgs,
) -> Tuple[PipelineManager, Dict]:
    # Configure global environment

    # Allocate ports for inter-process communications

    pipe_reader, pipe_writer = mp.Pipe(duplex=False)
    # Launch scheduler process
    scheduler_proc = mp.Process(
        target=run_scheduler_process,
        args=(
            server_args,
            pipe_writer,
        ),
    )
    
    scheduler_proc.start()
    
     # Get status from subprocess
    try:
        status_info = pipe_reader.recv()  # This will receive the {"status": "ready"} message
        print(f"Scheduler status: {status_info}")
    except Exception as e:
        print(f"Failed to get scheduler status: {e}")
    

    print("#"*100)

    # Launch pipeline process
    pipeline_manager = PipelineManager(server_args)

    pipeline_manager, scheduler_info = None, None
    return pipeline_manager, scheduler_info
