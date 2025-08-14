import gc
import inspect
import logging

import torch
import torch.utils.benchmark as benchmark

from diffusers.utils.testing_utils import torch_device


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

NUM_WARMUP_ROUNDS = 5


def benchmark_fn(f, *args, **kwargs):
    """Benchmark a function call and return the average execution time in seconds."""
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=1,
    )
    return float(f"{(t0.blocked_autorange().mean):.3f}")


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


# Adapted from https://github.com/lucasb-eyer/cnn_vit_benchmarks/blob/15b665ff758e8062131353076153905cae00a71f/main.py
def calculate_flops(model, input_dict):
    try:
        from torchprofile import profile_macs
    except ModuleNotFoundError:
        raise ModuleNotFoundError("torchprofile is required for FLOPS calculation. Install with: pip install torchprofile")

    # This is a hacky way to convert the kwargs to args as `profile_macs` cries about kwargs.
    sig = inspect.signature(model.forward)
    param_names = [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and p.name != "self"
    ]
    bound = sig.bind_partial(**input_dict)
    bound.apply_defaults()
    args = tuple(bound.arguments[name] for name in param_names)

    model.eval()
    with torch.no_grad():
        macs = profile_macs(model, args)
    flops = 2 * macs  # 1 MAC operation = 2 FLOPs (1 multiplication + 1 addition)
    return flops


def calculate_params(model):
    return sum(p.numel() for p in model.parameters())


# Users can define their own in case this doesn't suffice. For most cases,
# it should be sufficient.
def model_init_fn(model_cls, group_offload_kwargs=None, layerwise_upcasting=False, **init_kwargs):
    model = model_cls.from_pretrained(**init_kwargs).eval()
    if group_offload_kwargs and isinstance(group_offload_kwargs, dict):
        model.enable_group_offload(**group_offload_kwargs)
    else:
        model.to(torch_device)
    if layerwise_upcasting:
        model.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=init_kwargs.get("torch_dtype", torch.bfloat16)
        )
    return model