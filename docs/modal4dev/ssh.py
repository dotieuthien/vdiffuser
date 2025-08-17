import subprocess
import time
import modal

# Constants
SSH_PORT = 22
TIMEOUT_SECONDS = 3600 * 8  # 1 hour
SSH_PUBLIC_KEY_PATH = "id_rsa.pub"
GPU = "l4"
CPU = 8
MOUNT_ROOT_DIR = "/workspace_sgl"
MODEL_CACHE_PATH = "/cache"

# Initialize Modal app
app = modal.App("modal-ssh")

# Init volume for projects

cuda_version = "12.1.3"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

model_volume = modal.Volume.from_name("sglang", create_if_missing=True)
model_cache_volume = modal.Volume.from_name("model-cache", create_if_missing=True)

# Create base image with SSH server and TensorRT components
image = (
    modal.Image.from_registry(
		f"nvidia/cuda:12.8.1-devel-ubuntu22.04",
		setup_dockerfile_commands=["RUN ln -s /usr/bin/python3 /usr/bin/python"],
  add_python="3.12"
	)
    .apt_install(
        "openssh-server",
        "openmpi-bin",
        "libopenmpi-dev",
        "git",
        "git-lfs",
        "wget",
        "ffmpeg",
        "curl"
    )
    .run_commands("mkdir -p /run/sshd")
    .env({"HF_HUB_CACHE": MODEL_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .entrypoint([])
    .add_local_file(SSH_PUBLIC_KEY_PATH, "/root/.ssh/authorized_keys")  # Add this line
)

@app.function(
    image=image,
    timeout=TIMEOUT_SECONDS,
    gpu=GPU,
    cpu=CPU,
    volumes={MOUNT_ROOT_DIR: model_volume, MODEL_CACHE_PATH: model_cache_volume},
)
def start_ssh_server():
    """Start SSH server and keep it running for the specified timeout period."""
    # Start SSH daemon in debug mode
    sshd_process = subprocess.Popen(
        ["/usr/sbin/sshd", "-D", "-e"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Forward SSH port
    with modal.forward(port=SSH_PORT, unencrypted=True) as tunnel:
        hostname, port = tunnel.tcp_socket
        connection_cmd = f"ssh -p {port} root@{hostname}"
        print(f"SSH connection command: {connection_cmd}")

        try:
            time.sleep(TIMEOUT_SECONDS)
        except KeyboardInterrupt:
            print("\nShutting down SSH server...")
        finally:
            sshd_process.terminate()
