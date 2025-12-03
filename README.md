# Gaussian-Manipulation

## Installation

This installation guide is based on the DMesh repository (https://github.com/SonSang/dmesh). Please ensure you have Python 3.9 and recommend using Anaconda to manage the environment.

### Prerequisites

- Python 3.9
- Anaconda (recommended)
- NVIDIA GPU with CUDA 11.8 support

### Step 1: Clone the Repository

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/your-username/Gaussian-Manipulation.git
cd Gaussian-Manipulation
```

If you've already cloned the repository without submodules, initialize them:

```bash
git submodule update --init --recursive
```

### Step 2: Create Conda Environment

```bash
conda create -n gaussian-manipulation python=3.9
conda activate gaussian-manipulation
```

### Step 3: Install PyTorch 2.2.1 with CUDA 11.8

Install PyTorch 2.2.1 with CUDA 11.8 support:

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

Or using conda:

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 4: Install pytorch3d

First, install the required dependencies:

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

Then install pytorch3d. For PyTorch 2.2.1 with CUDA 11.8, you can install it using:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Step 5: Install Additional Dependencies

After installing pytorch3d, install the remaining required Python packages including dependencies for video preprocessing and RAFT:

```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python` - For image and video processing
- `numpy` - For numerical operations
- `tqdm` - For progress bars
- `Pillow` - For image handling (required by RAFT)
- `scipy` - For scientific computing (required by RAFT)

### Step 6: Install RAFT Submodule and Download Weights

This project uses RAFT (Recurrent All-Pairs Field Transforms) for optical flow estimation. RAFT is included as a git submodule.

Run the installation script to download RAFT weights:

```bash
chmod +x install_raft.sh
./install_raft.sh
```

**Note:** The RAFT weights will be available at:
- `RAFT/models/raft-things.pth` (and other models)

### Step 7: Install DG-Mesh

> Make sure you have `cuda 11.8` and `gcc 11` installed

```
# Install nvdiffrast
pip install git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
pip install git+https://github.com/NVlabs/nvdiffrast/

# Install pytorch3d
export FORCE_CUDA=1
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# Or install with prebuilt wheel
# pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8%2B5043d15/pytorch3d-0.7.8%2B5043d15pt2.2.1cu118-cp39-cp39-linux_x86_64.whl

cd DG-Mesh

# Install submodules
pip install dgmesh/submodules/diff-gaussian-rasterization
pip install dgmesh/submodules/simple-knn
# If direct install fails:
# cd dgmesh/submodules/diff-gaussian-rasterization
# pip install -e . --no-build-isolation
# cd ../../../
# cd dgmesh/submodules/simple-knn
# pip install -e . --no-build-isolation

# Install other dependencies
pip install -r requirements.txt
```