FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install basic dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget git curl build-essential gcc-11 g++-11 libgl1-mesa-dev libglib2.0-0 \
    unzip sudo vim && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya
ENV PATH=$CONDA_DIR/bin:$PATH

# Add conda-forge and remove defaults to avoid TOS issues (defaults requires TOS acceptance)
RUN conda config --remove channels defaults && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Create conda environment with Python 3.11 and accept TOS
# We use --override-channels to ensure only conda-forge is used, which shouldn't require TOS
RUN conda create -y -n 3dgrut python=3.11 --override-channels -c conda-forge
SHELL ["conda", "run", "-n", "3dgrut", "/bin/bash", "-c"]

# Set environment variables for CUDA build
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install PyTorch and related packages
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Clone 3DGRUT repository (we'll copy it from local context instead to include local changes)
WORKDIR /workspace/3dgrut
COPY . /workspace/3dgrut

# Install Python dependencies from requirements.txt
# Note: We manually install requirements to handle specific versions and git repos
RUN pip install plyfile torchmetrics tensorboard fire omegaconf hydra-core scikit-learn wandb polyscope>=2.3.0 addict rich slangtorch==1.3.4 kornia 'opencv-python<4.12.0' einops imageio msgpack dataclasses_json 'setuptools<72.1.0' tqdm libigl pygltflib usd-core gdown kaggle

# Install fused-ssim with no build isolation (fix for torch import error during build)
# We need to ensure CUDA is available for build, or force it if possible.
# Since we are building in docker, we might not have GPU access during build time unless enabled.
# However, PyTorch should be able to build extensions if CUDA toolkit is present (which it is from base image)
RUN pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157

# Install Kaolin
RUN pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html

# Install package in editable mode
RUN pip install -e .

# Default command
CMD ["/bin/bash"]