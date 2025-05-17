---

I. Environment Preparation (Hardware and System)  
1. Hardware Requirements  
   • NVIDIA GPU (CUDA compute capability 3.5+ recommended, RTX 30/40 series preferred)  
   • RAM ≥16GB, storage ≥50GB (for CUDA and deep learning libraries)  

2. System Configuration  
   • Windows/Linux/macOS (Ubuntu 22.04 LTS or Windows 11 recommended)  
   • Install the latest NVIDIA drivers: Verify driver version via `nvidia-smi` (must be ≥535.54)  

---

II. Python Environment Management  
1. Virtual Environment Tools  
| Tool        | Use Case                   | Installation Command                          |  
|-------------|----------------------------|-----------------------------------------------|  
| Miniconda   | Lightweight environment    | `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh` |  
| virtualenv  | Pure Python isolation      | `pip install virtualenv`                      |  

2. Create a Virtual Environment  
```bash  
# Create a Python 3.9 environment  
conda create -n dl_env python=3.9  
conda activate dl_env  
```  

3. Configure Domestic Mirrors  
```bash  
# Conda Tsinghua Mirror  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main  
conda config --set show_channel_urls yes  

# Pip Aliyun Mirror  
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple  
```  

---

III. GPU Acceleration Library Installation  
1. CUDA Toolkit Installation  
| Framework Version | Recommended CUDA Version | Verification Command |  
|-------------------|--------------------------|-----------------------|  
| TensorFlow 2.10   | 11.2                     | `nvcc --version`      |  
| PyTorch 2.0       | 11.8                     | `nvidia-smi`          |  

Install via official repository:  
```bash  
conda install cudatoolkit=11.8 -c nvidia  
```  

2. cuDNN Acceleration Library  
```bash  
# cuDNN matching CUDA 11.8  
conda install cudnn=8.6.0 -c nvidia  
```  

---

IV. Deep Learning Framework Installation  
1. TensorFlow GPU Version  
```bash  
# Install specific version (matching CUDA 11.2)  
pip install tensorflow==2.10.0  
```  

2. PyTorch GPU Version  
```bash  
# Install via PyTorch official channel  
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  
```  

---

V. Environment Verification  
1. GPU Availability Check  
```python  
import torch  
print(f"PyTorch GPU availability: {torch.cuda.is_available()}")  # Should return True  
```  

2. Version Consistency Check  
```python  
import tensorflow as tf  
print(f"TF-CUDA version: {tf.test.is_built_with_cuda()}")  # Verify CUDA compilation status  
```  

---

VI. Troubleshooting  
1. CUDA Version Conflict  
   • Symptom: `Could not load dynamic library 'cudart64_110.dll'`  
   • Solution: Check version compatibility via `conda list cudatoolkit`  

2. Virtual Environment Recognition Issue  
   • Symptom: Jupyter Notebook cannot select kernel  
   • Fix: Execute `python -m ipykernel install --user --name dl_env`  

---

Version Compatibility Reference Table  
Framework | Python Version | CUDA Version | cuDNN Version  
---|---|---|---  
TensorFlow 2.10 | 3.7-3.10 | 11.2 | 8.1  
PyTorch 2.0 | 3.8-3.11 | 11.8 | 8.6  

*Note: Full version matrix available at [NVIDIA Developer Documentation](https://developer.nvidia.com/cuda-toolkit-archive)*
