# LGRobustSAM

This repository contains the implementation for our paper.

## Overview

We release the core implementation and basic usage instructions to improve reproducibility.

## Setup

1) Create a conda environment and activate it.

```bash
conda create --name lgrobustsam python=3.10 -y
conda activate lgrobustsam
```

2) Use the command below to check your CUDA version.

```bash
nvidia-smi
```

3) Replace the CUDA version with yours in the command below.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[$YOUR_CUDA_VERSION]
# For example:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# cu117 = CUDA 11.7
```

4) Install the remaining dependencies.

```bash
pip install -r requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

## Demo

```bash
python eval-lg.py --gpu 0
```

## Training

```bash
python training/train.py
```

