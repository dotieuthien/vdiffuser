import faulthandler
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from http import HTTPStatus
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import psutil
import setproctitle
import torch
import zmq
from torch.distributed import barrier

from vdiffuser.server_args import ServerArgs
class Scheduler:
    def __init__(
        self,
        server_args: ServerArgs,
    ):
        self.server_args = server_args

        # Init inter-process communication
        context = zmq.Context(2)
        
    def init_memory_pool_and_cache(self):
        """Initialize memory pool and cache."""
        pass

    def event_loop(self):
        """Event loop."""
        pass

    def event_looop_overlap(self):
        """Run the scheduler."""
        pass
    
    def receive_requests(self):
        """Receive requests from the server."""
        pass
    
    def process_input_requests(self, request: InputRequest):
        """Process input requests."""
        pass
    
    def handle_generate_request(self, request: GenerateRequest):
        """Handle generate request."""
        pass
    
    def _add_request_to_queue(self, request: Request):
        """Add request to queue."""
        pass
    
    def _extend_requests_queue(self, requests: List[Request]):
        """Extend requests queue."""
        pass
    
    def get_next_batch_to_run(self):
        """Get next batch to run."""
        pass
    
    def update_running_batch(self, batch_id: int, status: str):
        """Update running batch."""
        pass
    
    def run_batch(self, batch_id: int):
        """Run batch."""
        pass
    
    def process_batch_result(self, batch_id: int):
        """Process batch results."""
        pass
    
    
def run_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    pipe_writer,
):
    # Generate the prefix
    prefix = ""

    # Config the process
    setproctitle.setproctitle(f"vdiffuser::scheduler{prefix.replace(' ', '_')}")
    # faulthandler.enable()
    # kill_itself_when_parent_died()
    # parent_process = psutil.Process().parent()

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            gpu_id,
        )
        pipe_writer.send(
            {
                "status": "ready",
            }
        )

    except Exception:
        # traceback = get_exception_traceback()
        # logger.error(f"Scheduler hit an exception: {traceback}")
        # parent_process.send_signal(signal.SIGQUIT)
        pass
    
