import dataclasses
import logging
import os
import signal
from collections import OrderedDict
from typing import Dict, List, Union


from vdiffuser.server_args import ServerArgs
from vdiffuser.hf_diffusers_utils import (
    get_pipeline,
)


logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Centralized manager for image generate and edit templates.

    This class encapsulates all image generate and edit-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for image generate and edit management.
    """
    def __init__(self, server_args: ServerArgs):
        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        
        # Read model args
        self.model_path = server_args.model_path
        self.pipeline = server_args.pipeline
        self.pipeline = get_pipeline(
            server_args.pipeline,
            server_args.model_path,
        )
        
    def generate_request(
        self,
        obj,
        created_time: Optional[float] = None,
    ):
        pass

    def _send_one_request(
        self, 
        obj,
        created_time: Optional[float] = None,
    ):
        self.send_to_scheduler.send_json(request)

    def _receive_one_request(self) -> dict:
        return self.recv_from_scheduler.recv_json()


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