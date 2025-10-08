### Windows系统，从源代码安装（建议，便于开发）

```bash
# 环境准备，先创造python=3.12的环境和下载VS C++开发资源
conda create -n qlib python=3.12

# 安装前置的库（我安装时的等级为2.3.3和3.1.4）
pip install numpy
pip install --upgrade cython

# 从源代码安装，便于修改和开发
git clone https://github.com/slc03/qlib.git 
cd qlib 
python -m pip install -e .
```