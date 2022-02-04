#!/bin/bash

usage() {
cat <<EOF

# clear checkpoints directory
rm -r ./checkpoints/*

# SINGLE NODE
run_dock_asuser.sh \
    --dockname=avolkov_tf \
    --container=nvcr.io/nvidian/sae/avolkov:tf19.09py3_tf1.14.0_ssh \
    --noninteractive \
    --entrypoint=bash \
    --dockcmd=./tensorflow_mnode/hvd_fashion_mnist_example.sh
    --log-dir tboard/distributed_amp \
    --epochs=10


# MULTI NODE

## INSIDE-OUT
nodes=($(scontrol show hostname $SLURM_NODELIST)) && nodes=${nodes[@]} && nodes=${nodes// /,}
export NODES=$nodes
PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-p 22" pdsh -w $NODES NODES=$NODES PATH=$PATH \
    pdsh_docker.sh --noderank=%n \
        --container=nvcr.io/nvidian/sae/avolkov:tf19.09py3_tf1.14.0_ssh \
        --ibdevices \
        --workingdir=${PWD} \
        --script=./tensorflow_mnode/hvd_fashion_mnist_example.sh \
        --log-dir=tboard/distributed \
        --epochs=10


# salloc needs to be done prior to srun commands.
salloc -N 2 -p some-partition

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:tf19.09py3_tf1.14.0_ssh \
    --ibdevices \
    --script=./tensorflow_mnode/hvd_fashion_mnist_example.sh \
    --log-dir=tboard/distributed \
    --epochs=10

TF_ENABLE_AUTO_MIXED_PRECISION=1 srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:tf19.09py3_tf1.14.0_ssh \
    --envlist=TF_ENABLE_AUTO_MIXED_PRECISION \
    --ibdevices \
    --script=./tensorflow_mnode/hvd_fashion_mnist_example.sh \
    --log-dir=tboard/distributed_amp \
    --epochs=10


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
export NCCL_IB_DISABLE=0

hostlistopts=''
if [ ! -z "${hostlist}" ]; then
    hostlistopts="-H $hostlist"
fi

if [ -z ${np:+x} ]; then
    np=$(bash -c 'nvidia-smi -L | wc -l')
fi

# using ppr doesn't work with ompi3.0.0 b/c of bug:
#   https://github.com/open-mpi/ompi/pull/4293
# --report-bindings --bind-to none --map-by ppr:4:socket \
# Container should have ompi3.1.x for the ppr binding to work.
mpirun -x LD_LIBRARY_PATH -x SHELL ${evars} $hostlistopts -np $np \
    -mca btl_tcp_if_exclude docker0,lo,virbr0 \
    -x NCCL_SOCKET_IFNAME \
    -x NCCL_IB_DISABLE \
    --report-bindings --bind-to none --map-by slot \
    python ./tensorflow_mnode/fashion_mnist_dist.py "$@"
