import asyncio
import dataclasses
import logging
import os
import time
import signal
from collections import OrderedDict
from typing import Dict, List, Union, Optional

import fastapi

import zmq
import zmq.asyncio

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

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Centralized manager for image generate and edit templates.

    This class encapsulates all image generate and edit-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for image generate and edit management.
    """
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, True
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
        )
        
    async def generate_request(
        self,
        obj: GenerateReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        created_time = time.time()
        print("Generating image...")
        image = self.pipeline(obj.text).images[0]
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