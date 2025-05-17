---

一、环境准备（硬件与系统）
1. 硬件要求  
   • NVIDIA显卡（支持CUDA计算能力3.5+，推荐RTX 30/40系列）

   • 内存≥16GB，存储空间≥50GB（用于安装CUDA及深度学习库）


2. 系统配置  
   • Windows/Linux/macOS（推荐Ubuntu 22.04 LTS或Windows 11）

   • 安装最新NVIDIA驱动：通过`nvidia-smi`命令验证驱动版本（需≥535.54）


---

二、Python环境管理
1. 创建虚拟环境
```bash
# 创建Python3.9环境
conda create -n dl_env python=3.9  
conda activate dl_env
```

2. 配置国内镜像源
```bash
# Conda清华源配置
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --set show_channel_urls yes

# Pip阿里源配置
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
```

---

三、GPU加速库安装
1. CUDA Toolkit安装

框架版本 ：TensorFlow 2.10；推荐CUDA版本：11.2 |
框架版本 ：PyTorch 2.0；推荐CUDA版本：11.8 ；

通过官方仓库安装：
```bash
conda install cudatoolkit=11.8 -c nvidia
```

2. cuDNN加速库
```bash
# 匹配CUDA 11.8的cuDNN
conda install cudnn=8.6.0 -c nvidia
```

---

四、深度学习框架安装
1. TensorFlow GPU版
```bash
# 指定版本安装（匹配CUDA 11.2）
pip install tensorflow==2.10.0
```

2. PyTorch GPU版
```bash
# 通过PyTorch官方通道安装
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

五、环境验证
1. GPU可用性检测
```python
import torch
print(f"PyTorch GPU可用状态：{torch.cuda.is_available()}")  # 应返回True
```

2. 版本一致性检查
```python
import tensorflow as tf
print(f"TF-CUDA版本：{tf.test.is_built_with_cuda()}")  # 验证CUDA编译状态
```

---

六、常见问题解决
1. CUDA版本冲突  
   • 现象：`Could not load dynamic library 'cudart64_110.dll'`  

   • 方案：通过`conda list cudatoolkit`检查版本匹配性


2. 虚拟环境识别异常  
   • 现象：Jupyter Notebook无法选择内核  

   • 修复：执行`python -m ipykernel install --user --name dl_env`


---

版本兼容性参考表
框架 | Python版本 | CUDA版本 | cuDNN版本  
---|---|---|---
TensorFlow 2.10 | 3.7-3.10 | 11.2 | 8.1  
PyTorch 2.0 | 3.8-3.11 | 11.8 | 8.6  

*注：完整版本矩阵可参考[NVIDIA开发者文档](https://developer.nvidia.com/cuda-toolkit-archive)*
