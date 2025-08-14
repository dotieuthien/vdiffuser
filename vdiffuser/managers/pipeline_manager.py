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
)

logger = logging.getLogger(__name__)


class ModelClient:
    def __init__(
        self,
        send_to_scheduler,
        recv_from_scheduler,
    ):
        self.send_to_scheduler = send_to_scheduler
        self.recv_from_scheduler = recv_from_scheduler
        
    def _send_one_request(
        self,
        keys_in_shared_memory: List[str],
    ):
        print(f"ModelClient sending request {keys_in_shared_memory}")
        self.send_to_scheduler.send_pyobj(keys_in_shared_memory)

    def __call__(self, *args, **kwargs):
        print("#"*100)
        # create a request id
        request_id = str(uuid.uuid4())
        created_time = time.time()
        
        keys_in_shared_memory = []
        
        # save in GPU memory pool
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                create_shared_tensor(f"{request_id}_arg_{i}", arg)
                keys_in_shared_memory.append(f"{request_id}_arg_{i}")
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                create_shared_tensor(f"{request_id}_kwarg_{key}", value)
                keys_in_shared_memory.append(f"{request_id}_kwarg_{key}")
                
        self._send_one_request(keys_in_shared_memory)
        # wait for 10 seconds
        time.sleep(10)


class PipelineManager:
    """
    Centralized manager for image generate and edit templates.

    This class encapsulates all image generate and edit-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for image generate and edit management.
    """
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.port_args = port_args
        
        # Init inter-process communication
        context = zmq.asyncio.Context(2)
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
        )
        self.pipeline.unet = ModelClient(
            self.send_to_scheduler,
            self.recv_from_scheduler,
        )
        self.pipeline.vae = ModelClient(
            self.send_to_scheduler,
            self.recv_from_scheduler,
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
):
    # kill_itself_when_parent_died()
    # setproctitle.setproctitle("sglang::detokenizer")
    # configure_logger(server_args)
    # parent_process = psutil.Process().parent()

    try:
        manager = PipelineManager(server_args)
        manager.event_loop()
    except Exception:
        # traceback = get_exception_traceback()
        # logger.error(f"PipelineManager hit an exception: {traceback}")
        # parent_process.send_signal(signal.SIGQUIT)
        pass