# Novel View Synthesis

Setting up the project:

## Create a Virtual Environment:
```
python3.8 -m venv venv_combined
```
## Activate the Virtual Environment:
```
source venv_combined/bin/activate
```
## Set Environment Variables:
```
export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-11.6/bin:$PATH"
```
## Install Core Dependencies:
```
python -m pip install -U pip wheel setuptools packaging ninja
```
## Install PyTorch and Related Libraries:
```
python -m pip install torchmetrics==0.11.4 torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
```
## Install Tiny CUDA NN and NVIDIA Apex:
```
python -m pip install -v git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex
```
## Install Additional Libraries:
```
python -m pip install scikit-image scipy tqdm imageio pyyaml imageio-ffmpeg lpips ConfigArgParse pandas pyrtools pyvista rerun-sdk einops==0.4.1 kornia==0.6.5 pytorch-lightning==1.7.7 matplotlib==3.5.2 opencv-python==4.6.0.66 'ptflops<=0.6.7' jupyter pymcubes trimesh dearpygui
```
## Install Custom Models and CUDA Kernel:
```
python -m pip install -v ngp_pl/models/csrc/
python -m pip install src/cuda/dist/kilonerf_cuda-0.0.0-cp38-cp38-linux_x86_64.whl
Install Torch-Scatter:
Install torch-scatter from a specific URL to match PyTorch version 1.13.0 with CUDA support:
```

# For replacing instant_ngp as trainer model
1. Place data in folder data/nsvf/Synthetic_NeRF/Lego
2. Run below bash script
```
./train_instant_ngp.sh
```
This performs below steps
a. Scaling of data to required [-0.5, 0.5] as required by the model
b. Train from sub_module ngp_pl
c. Copy checkpoint to logs/ folder where all checkpoints, occupancies are stored
d. Extracting occupancy grid from instant_ngp
e. Distilling instant_ngp into KiloNeRF model
f. Fine-tuning KiloNeRF model