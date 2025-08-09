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

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import torch
import uvloop


def _launch_subprocesses():
    template_manager, scheduler_info = None, None
    return template_manager, scheduler_info


