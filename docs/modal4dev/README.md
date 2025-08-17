# Modal SSH Server

This directory contains a Modal app that sets up an SSH server with GPU access for remote development and experimentation.

## Quick Start

### 1. Run the SSH Server

Start the SSH server with the following command:

```bash
modal run -d ssh.py::start_ssh_server
```

The `-d` flag runs the server in detached mode, allowing it to run in the background.

### 2. Connect via SSH

Once the server starts, it will display the SSH connection command in the logs:

```
SSH connection command: ssh -p <PORT> root@<HOSTNAME>
```

Copy and run this command to connect to your remote development environment.

## Public Key Setup

### Current Configuration

The setup is already configured to use the `id_rsa.pub` file in this directory for authentication. The public key is automatically added to the container's authorized keys during image build.

### Setting Up Your Own Public Key

To use your own SSH key:

1. **Generate a new SSH key pair** (if you don't have one):
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```

2. **Replace the public key file**:
   - Copy your public key content to `modal/id_rsa.pub`
   - Or replace the entire file with your own public key file

3. **Verify the public key format**:
   ```bash
   cat modal/id_rsa.pub
   ```
   Should look like: `ssh-rsa AAAAB3NzaC1yc2E... your_email@hostname`

### Using Multiple Keys

If you need to support multiple users or keys:

1. Edit the `ssh.py` file
2. Modify the `.add_local_file()` line to add multiple key files:
   ```python
   .add_local_file("id_rsa.pub", "/root/.ssh/authorized_keys")
   ```

## Server Configuration

### Resources
- **GPU**: L4 (configurable via `GPU` variable)
- **CPU**: 8 cores (configurable via `CPU` variable)
- **Timeout**: 8 hours (configurable via `TIMEOUT_SECONDS`)

### Volumes Mounted
- `/workspace`: Main workspace volume
- `/cache`: Model cache for HuggingFace models

## Customization

To modify the configuration, edit the constants in `ssh.py`:

```python
SSH_PORT = 22              # SSH port
TIMEOUT_SECONDS = 3600 * 8 # Session timeout
GPU = "l4"                 # GPU type
CPU = 8                    # CPU cores
```
