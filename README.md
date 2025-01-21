# Nvdiffrast Render Tool
Rendering model with Nvdiffrast.

## Description
[Nvdiffrast](https://nvlabs.github.io/nvdiffrast/#cube.py): Modular Primitives for High-Performance Differentiable Rendering.


## Installation
Developed on Python 3.8.19, pytorch 2.0.0, cuda 11.2. 

Clone the repository and submodules:
```bash
git clone --recursive https://gitlab.bj.sensetime.com/huangzhiwei.vendor/nvdiffrast_tool.git
cd nvdiffrast_tool
pip install -r requirements.txt
```
Install Pytorch from [official website](https://pytorch.org/get-started/previous-versions/). 

Install nvdiffrast:
```bash
cd nvdiffrast
sudo apt-get update && DEBIAN_FRONTEND=noninteractive sudo apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl
export CUDA_HOME=/usr/local/cuda
pip install -e .
cd ..
```

## Usage
```bash
python nvdiffrast_render.py --model /path/to/model_file --q qw qx qy qz --t x y z # (in opencv coordinate, world to camera)
```
