import zmq
import zmq.asyncio

from vdiffuser.server_args import ServerArgs, PortArgs
from vdiffuser.hf_diffusers_utils import (
    get_pipeline,
)


class TpWorker:
    def __init__(
        self,
        server_args: ServerArgs,
    ):
        self.server_args = server_args
        
        self.pipeline = get_pipeline(
            server_args.pipeline,
            server_args.model_path,
        )
        
        self.text_encoder = self.pipeline.text_encoder
        self.diffusion_model = self.pipeline.unet
        self.vae = self.pipeline.vae

    def _call_text_encoder(self, *args, **kwargs):
        input_ids = args[0]
        output = self.text_encoder(
            input_ids=input_ids.to(self.text_encoder.device),
            output_hidden_states=True,
        )
        print(type(output))
        return output
    
    def _call_diffusion_model(self, *args, **kwargs):
        output = self.diffusion_model(*args, **kwargs)
        return output
    
    def _call_vae(self, *args, **kwargs):
        output = self.vae(*args, **kwargs)
        return output
    
    def __call__(self, model_name, *args, **kwargs):
        if model_name == "text_encoder":
            return self._call_text_encoder(*args, **kwargs)
        elif model_name == "unet":
            return self._call_diffusion_model(*args, **kwargs)
        elif model_name == "vae":
            return self._call_vae(*args, **kwargs)
        else:
            raise ValueError(f"Model name {model_name} not supported")