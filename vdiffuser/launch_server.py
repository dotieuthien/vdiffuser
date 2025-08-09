"""Launch the VDiffuser inference server for diffusion models."""

import os
import sys

from vdiffuser.entrypoints.http_server import launch_server
from vdiffuser.server_args import prepare_server_args
from vdiffuser.utils import kill_process_tree


if __name__ == "__main__":
    # server_args = prepare_server_args(sys.argv[1:])
    server_args = {}
    
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
