#!/bin/bash
set -e  # 出错立即停止

nvcc --version
sudo apt update
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_26.3.2-2-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -f ~/miniconda3/miniconda.sh

export PATH=~/miniconda3/bin:$PATH

pip install ninja
pip install cuda-python==13.0.0
pip install torch --index-url https://download.pytorch.org/whl/cu130     

git clone  --recursive https://github.com/NVIDIA/cutlass.git
cd cutlass/python
export CUTLASS_NVCC_ARCHS=100
pip install numpy

pip install -e .

cd ~
git clone https://github.com/liziwen-AI/NVFP4_GEMM.git
cd NVFP4_GEMM






