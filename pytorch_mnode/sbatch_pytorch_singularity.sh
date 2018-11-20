#!/bin/bash

usage() {
cat <<EOF

module load singularity/3.0.0

sbatch -N 2 -p dgx-1v_32g \
    --output="slurm-%j-pytorch_hvd_mnist.out" \
    --wrap=./pytorch_mnode/sbatch_pytorch_singularity.sh

EOF
}

srun srun_singularity.sh \
    --container=/cm/shared/singularity/pytorch_hvd_apex.simg \
    --script=./pytorch_mnode/pytorch_hvd_mnist_example.sh
