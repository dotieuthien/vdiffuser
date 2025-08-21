import asyncio
from concurrent.futures import ThreadPoolExecutor
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

from transformers.modeling_outputs import BaseModelOutputWithPooling

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
from vdiffuser.managers.torch_shared_tensors import (
    create_shared_tensor,
    create_shared_tensor_tuple,
    read_shared_tensor,
    read_shared_tensor_tuple,
    create_shared_dict,
)

logger = logging.getLogger(__name__)


class ModelClient:
    def __init__(
        self,
        send_to_scheduler,
        recv_from_scheduler,
        input_shared_queue,
        output_shared_queue,
        model_name: Union["tokenizer_2", "text_encoder", "unet", "vae", None] = None,
    ):
        self.send_to_scheduler = send_to_scheduler
        self.recv_from_scheduler = recv_from_scheduler
        
        self.input_shared_queue = input_shared_queue
        self.output_shared_queue = output_shared_queue
        
        self.model_name = model_name
        
    def _send_one_request(
        self,
        keys_in_shared_memory: List[str],
    ):
        print(f"ModelClient {self.model_name} sending request {keys_in_shared_memory}")
        self.send_to_scheduler.send_pyobj((self.model_name, keys_in_shared_memory))

    def __call__(self, *args, **kwargs):
        print("#"*100)
        # create a request id
        request_id = str(uuid.uuid4())
        
        shared_input_tensor_keys = []
        
        # get all input tensors from args and kwargs
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                # push input tensors via data queue (CUDA IPC) 
                self.input_shared_queue.put((f"{request_id}_arg_{i}", arg))
                shared_input_tensor_keys.append(f"{request_id}_arg_{i}")
        for key, arg in kwargs.items():
            if isinstance(arg, torch.Tensor):
                self.input_shared_queue.put((f"{request_id}_kwarg_{key}", arg))
                shared_input_tensor_keys.append(f"{request_id}_kwarg_{key}")
                
        # send keys of input tensors to scheduler
        self._send_one_request(shared_input_tensor_keys)
        
        while True:
            try:
                recv_req = self.recv_from_scheduler.recv_pyobj(zmq.NOBLOCK)
                request_id, shared_output_tensor_keys = recv_req
                break
            except zmq.Again:
                # No message available, sleep briefly to avoid busy waiting
                time.sleep(0.001)  # 1ms sleep
                continue
            
        tensors = {}
        for key in shared_output_tensor_keys:
            def blocking_get():
                return self.output_shared_queue.get()
            
            # wait for the tensor to be available
            while key not in tensors.keys():
                rid, tensor = blocking_get()
                tensors[rid] = tensor
                print("#"*100)
                if isinstance(tensor, torch.Tensor):
                    print(f"PipelineManager received tensor shape and device: {tensor.shape}, {tensor.device}")
                elif isinstance(tensor, tuple):
                    for t in tensor:
                        print(f"PipelineManager received tensor shape and device: {t.shape}, {t.device}")
                
        for key, tensor in tensors.items():
            # assign tensor to text_encoder_output
            if "last_hidden_state" in key:
                last_hidden_state = tensor
            elif "pooler_output" in key:
                pooler_output = tensor
            elif "hidden_states" in key:
                hidden_states = tensor
                hidden_states = list(hidden_states)
                for i, hidden_state in enumerate(hidden_states):
                    hidden_states[i] = hidden_state.to(device="cuda")
                
                hidden_states = tuple(hidden_states)
                
        text_encoder_output = BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state.to(device="cuda"),
            pooler_output=pooler_output.to(device="cuda"),
            hidden_states=hidden_states,
        )
        
        self._send_one_request("done")
                
        return text_encoder_output


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
        input_shared_queue=None,
        output_shared_queue=None,
    ):
        self.server_args = server_args
        self.port_args = port_args
        
        self.input_shared_queue = input_shared_queue
        self.output_shared_queue = output_shared_queue
        
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
        
        def patch_scheduler(scheduler):
            if hasattr(scheduler, '_step_index') or hasattr(scheduler, 'step_index'):
                from diffusers import DDIMScheduler
                return DDIMScheduler.from_config(self.pipeline.scheduler.config)
            return scheduler
        
        self.pipeline.scheduler = patch_scheduler(self.pipeline.scheduler)
        
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
            self.input_shared_queue,
            self.output_shared_queue,
            "text_encoder",
        )
        self.thread_pool = ThreadPoolExecutor()
        
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
        # create a request id
        request_id = str(uuid.uuid4())
        created_time = time.time()
        
        self._send_one_request(request_id, created_time)
        
        def generate_image():
            # Enable classifier-free guidance with guidance_scale > 1.0
            return self.pipeline(
                obj.text, 
                num_inference_steps=10,
                guidance_scale=7.5  # Standard CFG value for stable diffusion
            ).images[0]

        try:
            # Run the blocking function in the thread pool
            loop = asyncio.get_running_loop()
            image = await loop.run_in_executor(
                self.thread_pool,
                generate_image
            )
            return image
        except Exception as e:
            print(f"Error generating image: {e}")
            raise
        
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