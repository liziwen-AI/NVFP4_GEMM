sudo apt update
sudo apt install nsight-compute
sudo apt install cmake
sudo apt install -y build-essential

# sudo apt purge -y 'nvidia-driver-*' 'nvidia-dkms-*'
# sudo apt autoremove -y
sudo apt install -y nvidia-driver-580-open
sudo reboot

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-0

export PATH=/usr/local/cuda/bin:$PATH





