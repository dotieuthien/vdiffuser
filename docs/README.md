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
# Launch the vdiffuser server
python -m vdiffuser.launch_server --model GraydientPlatformAPI/boltning-hyperd-sdxl --pipeline StableDiffusionXLPipeline
```

## Dev
```bash
# Current code still has some bugs
# It will create a lot of processes but cannot collect
# Get all processes
ps -ef | grep python
# And
pkill -9 python
```

