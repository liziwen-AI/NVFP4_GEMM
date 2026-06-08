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
pip install -e .
pip install numpy










