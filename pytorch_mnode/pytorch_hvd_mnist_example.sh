#!/bin/bash

usage() {
cat <<EOF

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
    --privileged \
    --script=./pytorch_mnode/pytorch_hvd_mnist_example.sh

EOF
}

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# to avoid warning: WARNING: local probe returned unhandled shell:unknown assuming bash
export SHELL=/bin/bash

export NCCL_SOCKET_IFNAME=^docker0,virbr0

# using ppr doesn't work with ompi3.0.0 b/c of bug:
#   https://github.com/open-mpi/ompi/pull/4293
# --report-bindings --bind-to none --map-by ppr:4:socket \
# --report-bindings --bind-to socket --map-by ppr:2:socket \
# --report-bindings --bind-to none --map-by slot \
mpirun -H $hostlist -mca btl_tcp_if_exclude docker0,lo,virbr0 \
    -x NCCL_SOCKET_IFNAME \
    --report-bindings --bind-to none --map-by slot \
    -np ${np} \
    python ./pytorch_mnode/pytorch_hvd_mnist.py \
    $@

# options: --epochs=1 --batch_size=256
