import asyncio
import dataclasses
import logging
import os
import time
import signal
import uuid
from collections import OrderedDict
from typing import Dict, List, Union, Optional

import fastapi

import zmq
import zmq.asyncio

import torch

from vdiffuser.server_args import ServerArgs, PortArgs
from vdiffuser.hf_diffusers_utils import (
    get_pipeline,
)

from vdiffuser.utils import (
    get_zmq_socket,
)
from vdiffuser.managers.io_struct import (
    GenerateReqInput,
)
from vdiffuser.managers.shared_gpu_memory import (
    create_shared_tensor,
    read_shared_tensor,
    create_shared_dict,
)

logger = logging.getLogger(__name__)


class ModelClient:
    def __init__(
        self,
        send_to_scheduler,
        recv_from_scheduler,
        input_shared_dict,
        output_shared_dict,
        model_name: Union["tokenizer_2", "text_encoder", "unet", "vae", None] = None,
    ):
        self.send_to_scheduler = send_to_scheduler
        self.recv_from_scheduler = recv_from_scheduler
        self.input_shared_dict = input_shared_dict
        self.output_shared_dict = output_shared_dict
        self.model_name = model_name
        
    def _send_one_request(
        self,
        keys_in_shared_memory: List[str],
    ):
        print(f"ModelClient sending request {keys_in_shared_memory}")
        self.send_to_scheduler.send_pyobj((self.model_name, keys_in_shared_memory))

    def __call__(self, *args, **kwargs):
        print("#"*100)
        # create a request id
        request_id = str(uuid.uuid4())
        created_time = time.time()
        
        keys_in_shared_memory = []
        
        # save in GPU memory pool
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                create_shared_tensor(self.input_shared_dict, f"{request_id}_arg_{i}", arg)
                keys_in_shared_memory.append(f"{request_id}_arg_{i}")
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                create_shared_tensor(self.input_shared_dict, f"{request_id}_kwarg_{key}", value)
                keys_in_shared_memory.append(f"{request_id}_kwarg_{key}")
                
        self._send_one_request(keys_in_shared_memory)
        
        while True:
            try:
                recv_req = self.recv_from_scheduler.recv_pyobj(zmq.NOBLOCK)
                print(f"ModelClient received from scheduler: {recv_req}")
                request_id, keys_out_shared_memory = recv_req
                print(f"ModelClient received from scheduler: {request_id}, {keys_out_shared_memory}")
                break
            except zmq.Again:
                # No message available, sleep briefly to avoid busy waiting
                time.sleep(0.001)  # 1ms sleep
                continue
            
        # wait for 10 seconds
        time.sleep(10)


class PipelineManager:
    """
    Centralized manager for image generate and edit templates.

    This class encapsulates all image generate and edit-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for image generate and edit management.
    """
    def __init__(
        self, 
        server_args: ServerArgs, 
        port_args: PortArgs, 
        input_shared_dict=None,
        output_shared_dict=None,
    ):
        self.server_args = server_args
        self.port_args = port_args
        
        # Use provided shared dictionary or create a new one for multi-process tensor sharing
        if input_shared_dict is not None:
            self.input_shared_dict = input_shared_dict
        else:
            self.input_shared_dict = create_shared_dict()
            
        if output_shared_dict is not None:
            self.output_shared_dict = output_shared_dict
        else:
            self.output_shared_dict = create_shared_dict()
        
        # Init inter-process communication
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.pipeline_manager_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        
        # Request states
        self.no_create_loop = False
        
        # Read model args
        self.model_path = server_args.model_path
        self.pipeline = server_args.pipeline
        self.pipeline = get_pipeline(
            server_args.pipeline,
            server_args.model_path,
            just_pipeline=True,
        )
        
        self.pipeline.__class__ = type(
            "PatchedPipeline",
            (self.pipeline.__class__,),
            {
                "do_classifier_free_guidance": property(lambda self: True)
            }
        )
        self.pipeline.text_encoder = ModelClient(
            self.send_to_scheduler,
            self.recv_from_scheduler,
            self.input_shared_dict,
            self.output_shared_dict,
            "text_encoder",
        )
        self.pipeline.unet = ModelClient(
            self.send_to_scheduler,
            self.recv_from_scheduler,
            self.input_shared_dict,
            self.output_shared_dict,
            "unet",
        )
        self.pipeline.vae = ModelClient(
            self.send_to_scheduler,
            self.recv_from_scheduler,
            self.input_shared_dict,
            self.output_shared_dict,
            "vae",
        )
        
    def _send_one_request(
        self,
        request_id: str,
        created_time: float,
    ):
        print(f"PipelineManager sending request id {request_id} that created at {created_time}")
        self.send_to_scheduler.send_pyobj((request_id, created_time))
        
    async def generate_request(
        self,
        obj: GenerateReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        print("Generating image...")
        # create a request id
        request_id = str(uuid.uuid4())
        created_time = time.time()
        print("Generating image...")
        self._send_one_request(request_id, created_time)
        
        # Enable classifier-free guidance with guidance_scale > 1.0
        image = self.pipeline(
            obj.text, 
            num_inference_steps=20,
            guidance_scale=7.5  # Standard CFG value for stable diffusion
        ).images[0]
        return image
        
    # def auto_create_handle_loop(self):
    #     if self.no_create_loop:
    #         return

    #     self.no_create_loop = True
    #     loop = asyncio.get_event_loop()
    #     self.asyncio_tasks.add(
    #         loop.create_task(print_exception_wrapper(self.handle_loop))
    #     )

    #     self.event_loop = loop

    #     # We cannot add signal handler when the tokenizer manager is not in
    #     # the main thread due to the CPython limitation.
    #     if threading.current_thread() is threading.main_thread():
    #         signal_handler = SignalHandler(self)
    #         loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
    #         # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
    #         loop.add_signal_handler(
    #             signal.SIGQUIT, signal_handler.running_phase_sigquit_handler
    #         )
    #     else:
    #         logger.warning(
    #             "Signal handler is not added because the tokenizer manager is "
    #             "not in the main thread. This disables graceful shutdown of the "
    #             "tokenizer manager when SIGTERM is received."
    #         )
    #     self.asyncio_tasks.add(
    #         loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
    #     )


def run_pipeline_process(
    server_args: ServerArgs,
    port_args: PortArgs = None,
    shared_dict=None,
):
    # kill_itself_when_parent_died()
    # setproctitle.setproctitle("sglang::detokenizer")
    # configure_logger(server_args)
    # parent_process = psutil.Process().parent()

    try:
        manager = PipelineManager(server_args, port_args, shared_dict)
        manager.event_loop()
    except Exception:
        # traceback = get_exception_traceback()
        # logger.error(f"PipelineManager hit an exception: {traceback}")
        # parent_process.send_signal(signal.SIGQUIT)
        pass