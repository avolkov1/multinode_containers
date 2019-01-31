#!/bin/bash

usage() {
cat <<EOF

nodes=($(scontrol show hostname $SLURM_NODELIST)) && nodes=${nodes[@]} && nodes=${nodes// /,}
export NODES=$nodes
PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-p 22" pdsh -w $NODES NODES=$NODES \
    pdsh_docker.sh --noderank=%n \
        --container=nvcr.io/nvidian/sae/avolkov:tf1.12.0py3_cuda10.0_cudnn7_ubuntu16_nccl2.3.7_hvd_ompi3_ibverbs \
        --privileged \
        --script=./tensorflow_mnode/hvd_mnist_example.sh \
        --workingdir=${PWD}

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:tf1.12.0py3_cuda10.0_cudnn7_ubuntu16_nccl2.3.7_hvd_ompi3_ibverbs \
    --privileged \
    --script=./tensorflow_mnode/hvd_mnist_example.sh

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:tf1.8.0py3_cuda9.0_cudnn7_nccl2.2.13_hvd_ompi3_ibverbs \
    --privileged \
    --script=./tensorflow_mnode/hvd_mnist_example.sh

srun srun_singularity.sh \
    --container=/cm/shared/singularity/tf1.8.0py3.simg \
    --script=./tensorflow_mnode/hvd_mnist_example.sh

EOF
}

# to avoid warning: WARNING: local probe returned unhandled shell:unknown assuming bash
export SHELL=/bin/bash

evars=''
if [ ! -z "${ENVLIST// }" ]; then
    for evar in ${ENVLIST//,/ } ; do
        evars="-x ${evar} ${evars}"
    done
fi

export NCCL_SOCKET_IFNAME=^docker0,lo,virbr0
# export NCCL_IB_DISABLE=1

# using ppr doesn't work with ompi3.0.0 b/c of bug:
#   https://github.com/open-mpi/ompi/pull/4293
# --report-bindings --bind-to none --map-by ppr:4:socket \
# Container should have ompi3.1.x for the ppr binding to work.
mpirun -x LD_LIBRARY_PATH -x SHELL ${evars} -H $hostlist -np $np \
    -mca btl_tcp_if_exclude docker0,lo,virbr0 \
    -x NCCL_SOCKET_IFNAME \
    --report-bindings --bind-to none --map-by slot \
    python ./tensorflow_mnode/tensorflow_mnist.py "$@"
