---

### I. Hardware and System Preparation
**1. Hardware Requirements**  
- NVIDIA GPU (Compute Capability 3.5+ supported, preferably RTX 30/40 series)  
- RAM ≥ 16GB, Storage space ≥ 50GB (for installing CUDA and deep learning libraries)

**2. System Configuration**  
- Windows/Linux/macOS (recommended: Ubuntu 22.04 LTS or Windows 11)  
- Install the latest NVIDIA driver: verify driver version with `nvidia-smi` (needs ≥ 535.54)

---

### II. Python Environment Management
**1. Virtual Environment Tools Selection**  
| Tool         | Suitable Scenario                   | Installation Command                                                     |  
|--------------|-------------------------------------|---------------------------------------------------------------------------|  
| Miniconda    | Lightweight environment management  | `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh` |  
| virtualenv   | Pure Python environment isolation   | `pip install virtualenv`                                                   |  

**2. Create Virtual Environment**  
```bash
# Create a Python 3.9 environment
conda create -n dl_env python=3.9  
conda activate dl_env
```

**3. Configure China-based Mirror Sources**  
```bash
# Conda Tsinghua Source configuration
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --set show_channel_urls yes

# Pip Aliyun (Alibaba Cloud) mirror configuration
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
```

---

### III. GPU Acceleration Libraries Installation
**1. CUDA Toolkit Installation**  
| Framework Version | Recommended CUDA Version | Verification Command |  
|---------------------|---------------------------|------------------------|  
| TensorFlow 2.10     | 11.2                      | `nvcc --version`       |  
| PyTorch 2.0         | 11.8                      | `nvidia-smi`           |  

Install via official repository:  
```bash
conda install cudatoolkit=11.8 -c nvidia
```

**2. cuDNN acceleration Library**  
```bash
# Match cuDNN version 8.6.0 with CUDA 11.8
conda install cudnn=8.6.0 -c nvidia
```

---

### IV. Deep Learning Framework Deployment
**1. TensorFlow GPU Version**  
```bash
# Install specific version (matching CUDA 11.2)
pip install tensorflow==2.10.0
```

**2. PyTorch GPU Version**  
```bash
# Install via official PyTorch channel
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

### V. Environment Verification
**1. GPU availability check**  
```python
import torch
print(f"PyTorch GPU available: {torch.cuda.is_available()}")  # Should return True
```

**2. Version consistency check**  
```python
import tensorflow as tf
print(f"TF-CUDA compatibility: {tf.test.is_built_with_cuda()}")  # Verify CUDA support in TF
```

---

### VI. Common Issue Resolution
**1. CUDA Version Conflicts**  
- Symptom: `Could not load dynamic library 'cudart64_110.dll'`  
- Solution: Use `conda list cudatoolkit` to verify version compatibility

**2. Virtual Environment Recognition Issues**  
- Symptom: Jupyter Notebook cannot select the kernel  
- Fix: Run `python -m ipykernel install --user --name dl_env`

---

### Compatibility Reference Table
| Framework          | Python Version | CUDA Version | cuDNN Version |  
|-------------------|----------------|--------------|--------------|  
| TensorFlow 2.10   | 3.7-3.10       | 11.2         | 8.1          |  
| PyTorch 2.0       | 3.8-3.11       | 11.8         | 8.6          |  

*Note: For a complete version matrix, refer to the [NVIDIA Developer Documentation](https://developer.nvidia.com/cuda-toolkit-archive)*
