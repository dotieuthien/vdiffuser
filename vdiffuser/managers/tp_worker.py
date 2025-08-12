from vdiffuser.server_args import ServerArgs, PortArgs


class TpWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        # nccl_port: int,
    ):
        self.server_args = server_args
        # self.nccl_port = nccl_port

    def run(self):
        pass