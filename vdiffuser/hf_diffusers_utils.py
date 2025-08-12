import warnings
from pathlib import Path
from typing import Optional, Union, Any

# Keep existing imports that are referenced in the tokenizer function
from vdiffuser.utils import is_remote_url


def get_pipeline(
    pipeline_name: str,
    model_path: str,
    *args,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    torch_dtype: Optional[str] = "auto",
    **kwargs,
) -> Any:
    """Gets a diffusion pipeline for the given model name via Hugging Face Diffusers.

    Args:
        pipeline_name: The name of the pipeline class to import from diffusers 
                      (e.g., 'StableDiffusionPipeline', 'StableDiffusionXLPipeline')
        model_path: Path or name of the model to load
        *args: Additional positional arguments to pass to the pipeline
        trust_remote_code: Whether to trust remote code
        revision: Model revision to use
        torch_dtype: Torch dtype to use (string or "auto")
        **kwargs: Additional keyword arguments to pass to the pipeline
    """

    model_name_or_path = model_path

    # Import torch for dtype handling
    import torch

    # Handle torch_dtype conversion
    if torch_dtype == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype, torch.float32)

    # Dynamically import the specific pipeline class from diffusers
    try:
        import diffusers
        pipeline_class = getattr(diffusers, pipeline_name)
    except ImportError:
        raise ImportError(
            "diffusers library is required but not installed. "
            "Please install it with: pip install diffusers"
        )
    except AttributeError:
        raise AttributeError(
            f"Pipeline class '{pipeline_name}' not found in diffusers. "
            f"Available pipelines can be found in the diffusers documentation."
        )

    try:
        print("Load pipeline ", pipeline_class)
        # Load the specific pipeline class
        # pipeline = pipeline_class.from_pretrained(
        #     model_name_or_path,
        #     *args,
        #     trust_remote_code=trust_remote_code,
        #     revision=revision,
        #     torch_dtype=torch_dtype,
        #     **kwargs,
        # )
        pipeline = None
    except Exception as e:
        err_msg = (
            f"Failed to load the diffusion pipeline '{pipeline_name}' from '{model_name_or_path}'. "
            f"Error: {str(e)}. "
            "Make sure the model path is correct and the model is compatible with the specified pipeline."
        )
        raise RuntimeError(err_msg) from e

    # # Move to GPU if available
    # if torch.cuda.is_available():
    #     pipeline = pipeline.to("cuda")

    return pipeline
