#!/bin/bash
# Initialize a conda environment for NPU-based training.

# conda create -n omnigen2 python=3.11 -y
conda create -n bagel python=3.10.6 -y
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate bagel
# pip install torch==2.8.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
cd 2.6.0
pip3 install torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl
pip3 install torch_npu-2.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
cd ..
pip install numpy==1.26.4
pip install pyyaml
pip install -r requirements.txt
pip install -r requirements_npu.txt
pip install numpy==1.26.4
# pip install flash-attn==2.7.4.post1 --no-build-isolation


######################## 910B install torchvision-npu


# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# python setup.py bdist_wheel
# cd dist
# pip install torchvision_npu-0.16.*.whl
