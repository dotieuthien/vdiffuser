"""
The definition of objects transferred between different
processes (TextEncoder, DiffusionModel, Sampler, VAEEncoder, VAEDecoder).
"""

import copy
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union


