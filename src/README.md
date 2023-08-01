
# KiloNeRF

Open a terminal in the root directory of this repo and execute 
```
python3.8 -m venv venv_combined
source venv_combined/bin/activate
export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-11.6/bin:$PATH"
python -m pip install -U pip wheel setuptools packaging ninja
python -m pip install torchmetrics==0.11.4 torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
python -m pip install -v git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex
python -m pip install scikit-image scipy tqdm imageio pyyaml imageio-ffmpeg lpips ConfigArgParse pandas pyrtools pyvista rerun-sdk einops==0.4.1 kornia==0.6.5 pytorch-lightning==1.7.7 matplotlib==3.5.2 opencv-python==4.6.0.66 'ptflops<=0.6.7' jupyter pymcubes trimesh dearpygui
python -m pip install cuda/dist/kilonerf_cuda-0.0.0-cp38-cp38-linux_x86_64.whl
```

### Download pretrained models
https://www.dropbox.com/sh/tgolvg5h54sdguq/AABZRKkly9PEM9JVubXLPptya?dl=0

### Download NSVF datasets
Credit to NSVF authors for providing their datasets: https://github.com/facebookresearch/NSVF

```
cd $KILONERF_HOME/data/nsvf
wget https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip && unzip -n Synthetic_NSVF.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NeRF.zip && unzip -n Synthetic_NeRF.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip && unzip -n BlendedMVS.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip && unzip -n TanksAndTemple.zip
```
Since we slightly adjusted the bounding boxes for some scenes, it is important that you
use the provided `unzip` argument to avoid overwriting our bounding boxes.

## Usage

To benchmark a trained model run:  
`bash benchmark.sh`

You can launch the **interactive viewer** by running:  
`bash render_to_screen.sh`

To train a model yourself run  
`bash train.sh`

The default dataset is `Synthetic_NeRF_Lego`, you can adjust the dataset by
setting the dataset variable in the respective script.
