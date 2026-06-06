nvcc --version

pip install cuda-python==13.0.0
pip install torch --index-url https://download.pytorch.org/whl/cu130     

git clone  --recursive https://github.com/NVIDIA/cutlass.git
cd cutlass
cd python
export CUTLASS_NVCC_ARCHS=100
pip install -e .
pip install numpy










