#!/bin/bash
nvcc --version
sudo apt update
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_26.3.2-2-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
source ~/.bashrc
conda --version

pip install ninja
pip install cuda-python==13.0.0
pip install torch --index-url https://download.pytorch.org/whl/cu130     

git clone  --recursive https://github.com/NVIDIA/cutlass.git
cd cutlass
cd python
export CUTLASS_NVCC_ARCHS=100
pip install numpy
python setup_cutlass.py develop
# 2. 安装 cutlass 核心组件
python setup_cutlass.py develop
# 3. 安装 pycute 组件（NVFP4 算子开发经常需要依赖 CuTe 布局映射）
python setup_pycute.py develop
# 4. 安装 cutlass_library 组件
python setup_library.py develop

cd ~
git clone https://github.com/liziwen-AI/NVFP4_GEMM.git
cd NVFP4_GEMM






