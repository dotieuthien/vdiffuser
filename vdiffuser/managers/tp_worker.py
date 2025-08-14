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

    def run(self):
        pass