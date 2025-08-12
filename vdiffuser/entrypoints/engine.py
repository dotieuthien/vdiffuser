"""
The entry point of inference server

This file implements python APIs for the inference engine.
"""

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

import torch
import uvloop


def _launch_subprocesses(
    server_args: ServerArgs,
) -> Tuple[PipelineManager, Dict]:
    # Configure global environment
    
    # Allocate ports for inter-process communications
    
    # Launch scheduler process
    scheduler_proc = mp.Process(
        target=run_scheduler_process,
        args=(
            server_args,
        ),
    )
    
    # Launch pipeline process
    pipeline_proc = mp.Process(
        target=run_pipeline_process,
        args=(
            server_args,
        ),
    )
    pipeline_proc.start()
    
    pipeline_manager, scheduler_info = None, None
    return pipeline_manager, scheduler_info


