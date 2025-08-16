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
python -m vdiffuser.launch_server --model GraydientPlatformAPI/boltning-hyperd-sdxl --pipeline StableDiffusionXLPipeline
```

##
**Made with ❤️ by the VDiffuser Team**

