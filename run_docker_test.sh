#!/bin/bash
set -e

# Build the docker image
# We tag it as 3dgrut:latest
docker build -t 3dgrut:latest .

# Run the training inside the container
# We mount the dataset directory.
# Since the container path is /workspace/3dgrut/data/real_data, we need to make sure the host path matches
# The dataset on host is at ~/3dgrut/data/real_data (which is /home/azureuser/3dgrut/data/real_data)
# We also need to mount /data/extracted because 'images' in ~/3dgrut/data/real_data is a symlink to /data/extracted/images
# So we should just mount /data/extracted to the same location inside the container OR reconstruct the link.
# Easier: Mount /data/extracted directly as data/real_data/images parent if possible, but the symlink points to absolute path.
# Let's verify the symlink. It points to /data/extracted/images.
# So we need /data/extracted/images to exist inside the container.
# We can mount /data/extracted to /data/extracted inside container.
# We also increase shared memory size to avoid Bus error
# Also setting PYTORCH_CUDA_ALLOC_CONF inside the container via ENV var

docker run --gpus all --shm-size=8g \
    -v /home/azureuser/3dgrut/data/real_data:/workspace/3dgrut/data/real_data \
    -v /data/extracted:/data/extracted \
    -v /home/azureuser/3dgrut/runs_docker:/workspace/3dgrut/runs \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    3dgrut:latest \
    conda run -n 3dgrut python train.py --config-name apps/nerf_synthetic_3dgrt.yaml \
    path=data/real_data out_dir=runs experiment_name=real_data_3dgrt_docker +dataset.downsample_factor=16