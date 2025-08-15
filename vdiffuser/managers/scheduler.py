import faulthandler
import logging
import os
import signal
import sys
import threading
import time
import uuid
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

from vdiffuser.server_args import ServerArgs, PortArgs
from vdiffuser.utils import (
    kill_itself_when_parent_died, 
    get_exception_traceback,
    get_zmq_socket,
)
from vdiffuser.managers.shared_gpu_memory import (
    create_shared_tensor,
    read_shared_tensor,
)
from vdiffuser.managers.tp_worker import TpWorker


logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        input_shared_dict=None,
        output_shared_dict=None,
    ):
        self.server_args = server_args

        # Use provided shared dictionary for multi-process tensor sharing
        self.input_shared_dict = input_shared_dict
        self.output_shared_dict = output_shared_dict

        # Init inter-process communication
        context = zmq.Context(2)
        
        self.recv_from_pipeline_manager = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, False
        )
        
        self.send_to_pipeline_manager = get_zmq_socket(
            context, zmq.PUSH, port_args.pipeline_manager_ipc_name, False
        )
        
        self.tp_worker = TpWorker(
            server_args=server_args,
        )
        
    # def init_memory_pool_and_cache(self):
    #     """Initialize memory pool and cache."""
    #     pass

    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            try:
                recv_req = self.recv_from_pipeline_manager.recv_pyobj(zmq.NOBLOCK)
                print(f"Scheduler received request: {recv_req}")
                
                model_name, keys_in_shared_memory = recv_req
                print(f"Scheduler received request: {model_name}, {keys_in_shared_memory}")
                # If the request contains tensor keys, read the shared tensors
                if isinstance(keys_in_shared_memory, list) and self.input_shared_dict is not None:
                    # This is a list of tensor keys from ModelClient
                    tensors = {}
                    for tensor_key in keys_in_shared_memory:
                        tensor = read_shared_tensor(self.input_shared_dict, tensor_key)
                        if tensor is not None:
                            tensors[tensor_key] = tensor
                            print(f"Scheduler read shared tensor: {tensor_key}, shape: {tensor.shape}, device: {tensor.device}")
                        else:
                            print(f"Scheduler could not find tensor: {tensor_key}")
                    
                    # Process the tensors here
                    # You can add your tensor processing logic here
                    print(f"Scheduler loaded {len(tensors)} tensors for processing")
                    output_tensors = self.tp_worker(model_name, tensor)
                    
                    # save the output tensors to the output shared dictionary
                    request_id = str(uuid.uuid4())
                    keys_out_shared_memory = []
                    for key, tensor in output_tensors.items():
                        print("#"*100)
                        print(len(tensor))
                        create_shared_tensor(self.output_shared_dict, f"{request_id}_{key}", tensor)
                        keys_out_shared_memory.append(f"{request_id}_{key}")
                        
                    self.send_to_pipeline_manager.send_pyobj((request_id, keys_out_shared_memory))
                    
                    
                elif isinstance(recv_req, tuple) and len(recv_req) == 2:
                    # This is a (request_id, created_time) tuple from PipelineManager
                    request_id, created_time = recv_req
                    print(f"Scheduler received pipeline request: {request_id} created at {created_time}")
                    
                    # Process the pipeline request here
                    # You can add your pipeline processing logic here
                
            except zmq.Again:
                # No message available, sleep briefly to avoid busy waiting
                time.sleep(0.001)  # 1ms sleep
                continue

    # def event_looop_overlap(self):
    #     """Run the scheduler."""
    #     pass
    
    # def receive_requests(self):
    #     """Receive requests from the server."""
    #     pass
    
    # def process_input_requests(self, request: InputRequest):
    #     """Process input requests."""
    #     pass
    
    # def handle_generate_request(self, request: GenerateRequest):
    #     """Handle generate request."""
    #     pass
    
    # def _add_request_to_queue(self, request: Request):
    #     """Add request to queue."""
    #     pass
    
    # def _extend_requests_queue(self, requests: List[Request]):
    #     """Extend requests queue."""
    #     pass
    
    # def get_next_batch_to_run(self):
    #     """Get next batch to run."""
    #     pass
    
    # def update_running_batch(self, batch_id: int, status: str):
    #     """Update running batch."""
    #     pass
    
    # def run_batch(self, batch_id: int):
    #     """Run batch."""
    #     pass
    
    # def process_batch_result(self, batch_id: int):
    #     """Process batch results."""
    #     pass
    
    
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    input_shared_dict=None,
    output_shared_dict=None,
):
    # Generate the prefix
    prefix = ""

    # Config the process
    setproctitle.setproctitle(f"vdiffuser::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            input_shared_dict,
            output_shared_dict,
        )
        pipe_writer.send(
            {
                "status": "ready",
            }
        )
        
        scheduler.event_loop_normal()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
    
