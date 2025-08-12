# Install VDiffuser for Develop

## Install from source

```bash
# Use the last release branch
git clone https://github.com/dotieuthien/vdiffuser.git
cd vdiffuser

# Install the python packages
pip install --upgrade pip
pip install -e .
```

## Start the server

```bash
# Launch the vdiffuser server with FLUX.1-dev model
python -m vdiffuser.launch_server --model black-forest-labs/FLUX.1-dev --pipeline FluxPipeline
```
