# Gaussian-Manipulation

## Installation

This installation guide is based on the DMesh repository (https://github.com/SonSang/dmesh).
This guide is written for Ubuntu/Linux environment. Please ensure you have Python 3.9 and recommend using Anaconda to manage the environment.

### Prerequisites

- Python 3.9
- Anaconda (recommended)
- NVIDIA GPU with CUDA 11.8 support

### Step 1: Create Conda Environment

```bash
conda create -n gaussian-manipulation python=3.9
conda activate gaussian-manipulation
```

### Step 2: Install PyTorch 2.2.1 with CUDA 11.8

Install PyTorch 2.2.1 with CUDA 11.8 support:

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

Or using conda:

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 3: Install pytorch3d

First, install the required dependencies:

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

Then install pytorch3d. For PyTorch 2.2.1 with CUDA 11.8, you can install it using:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Step 4: Install Additional Dependencies

Install other required Python packages:

```bash
pip install -r requirements.txt
```
