# Installation

### Dependencies Installation

This repository is built in PyTorch 1.13.1 and Python 3.8.11.
Follow these intructions

1. Clone our repository
```

cd OMoE-Net
```

2. Create conda environment
The Conda environment used can be recreated using the env.yml file
```
conda env create -f env.yml
```


### Dataset Download and Preperation

We use DIV2K (800 training images) with self-generated degradation for training.Test datasets are self-degraded from standard test sets: BSD68, Urban100, Kodak24, CBSD68.
All tasks: gsn, sp, jpeg, gb, mb
gsn: Gaussian Noise
sp: Salt & Pepper Noise
jpeg: JPEG Compression
gb: Gaussian Blur
mb: Motion Blur
The training and test datasets share the same directory structure (HR + LR_{task}).
The full dataset structure is organized as:

```
|-- data
|    |-- Train                    # 训练集 (DIV2K 800张)
|    |    |-- HR                  # 清晰原图
|    |    |    |-- 0001.png
|    |    |    |-- 0002.png
|    |    |    ...
|    |    |-- LR_gsn               # 高斯噪声退化
|    |    |    |-- 0001.png
|    |    |    |-- 0002.png
|    |    |    ...
|    |    |-- LR_sp                # 椒盐噪声退化
|    |    |    |-- 0001.png
|    |    |    |-- 0002.png
|    |    |    ...
|    |    |-- LR_jpeg              # JPEG压缩退化
|    |    |    |-- 0001.png
|    |    |    |-- 0002.png
|    |    |    ...
|    |    |-- LR_gb                # 高斯模糊退化
|    |    |    |-- 0001.png
|    |    |    |-- 0002.png
|    |    |    ...
|    |    |-- LR_mb                # 运动模糊退化
|    |    |    |-- 0001.png
|    |    |    |-- 0002.png
|    |    |    ...
|    |
|    |-- test                      # 测试集（4个数据集，结构与Train完全一致）
|    |    |-- test_bsd68           # BSD68 测试集
|    |    |    |-- HR
|    |    |    |-- LR_gsn
|    |    |    |-- LR_sp
|    |    |    |-- LR_jpeg
|    |    |    |-- LR_gb
|    |    |    |-- LR_mb
|    |    |
|    |    |-- test_urban100        # Urban100 测试集
|    |    |    |-- HR
|    |    |    |-- LR_gsn
|    |    |    |-- LR_sp
|    |    |    |-- LR_jpeg
|    |    |    |-- LR_gb
|    |    |    |-- LR_mb
|    |    |
|    |    |-- test_kodak24         # Kodak24 测试集
|    |    |    |-- HR
|    |    |    |-- LR_gsn
|    |    |    |-- LR_sp
|    |    |    |-- LR_jpeg
|    |    |    |-- LR_gb
|    |    |    |-- LR_mb
|    |    |
|    |    |-- test_cbsd68          # CBSD68 测试集
|    |    |    |-- HR
|    |    |    |-- LR_gsn
|    |    |    |-- LR_sp
|    |    |    |-- LR_jpeg
|    |    |    |-- LR_gb
|    |    |    |-- LR_mb
...
```
