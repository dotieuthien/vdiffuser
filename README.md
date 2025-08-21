# 🎨 VDiffuser

**Inference Server for Diffusion Models**

VDiffuser is a high-performance inference server for image generation models, designed to maximize GPU utilization and minimize CPU overhead. Compatible with OpenAI SDK for seamless integration.

## ✨ Key Features

- 🚀 **High Performance**: Optimized GPU utilization with minimal CPU overhead
- 🔄 **Dynamic Batching**: Maximize throughput with dynamic batching
- 📡 **OpenAI Compatible**: Drop-in replacement for OpenAI image generation API
- 🎯 **Model Support**: ...

## 🛠️ Why we built this
Existing frameworks help with model loading and inference, but running an API that serves several requests at once—while fully utilizing GPU resources—quickly gets messy. Async batching, CPU overhead, and large pipelines make it hard to keep things fast and production-ready. We built this to keep performance high, avoid over-engineering for tiny GPUs (that's better for personal use), and provide a clean, practical solution for scaling real workloads. 🚀

## 🚀 Quickstart

### Install from source

```bash
# Use the last release branch
git clone https://github.com/dotieuthien/vdiffuser.git
cd vdiffuser

# Install the python packages
pip install --upgrade pip
pip install -e .
```

### Start the server

```bash
# Launch the vdiffuser server
vdiffuser serve --model GraydientPlatformAPI/boltning-hyperd-sdxl --pipeline StableDiffusionXLPipeline
```

## 🧰 CLI

### Commands

- `vdiffuser serve`: Launch the HTTP server
  - Required:
    - `--model`: HF repo ID or local path
    - `--pipeline`: Diffusers pipeline class name

- `vdiffuser version`: Show package version

Notes:
- Running with flags only defaults to `serve`, e.g. `vdiffuser --model ... --pipeline ...`.

### Examples

```bash
# Start server
vdiffuser serve --model GraydientPlatformAPI/boltning-hyperd-sdxl --pipeline StableDiffusionXLPipeline

# Help
vdiffuser serve -h

# Version
vdiffuser version
```

##
**Made with ❤️ by the VDiffuser Team**

