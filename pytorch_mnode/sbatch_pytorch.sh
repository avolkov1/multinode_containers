#!/bin/bash

usage() {
cat <<EOF

sbatch -N 2 -p dgx-1v_32g \
    --output="slurm-%j-pytorch_hvd_mnist.out" \
    --wrap=./pytorch_mnode/sbatch_pytorch.sh

EOF
}

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
    --privileged \
    --script=./pytorch_mnode/pytorch_hvd_mnist_example.sh
