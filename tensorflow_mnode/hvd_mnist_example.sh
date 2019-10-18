#!/bin/bash

usage() {
cat <<EOF

# clear checkpoints directory
rm -r ./checkpoints/*

# SINGLE NODE
run_dock_asuser.sh \
    --dockname=avolkov_tf \
    --container=nvcr.io/nvidian/sae/avolkov:tf1.12.0py3_cuda10.0_cudnn7_ubuntu16_nccl2.3.7_hvd_ompi3_ibverbs \
    --noninteractive \
    --entrypoint=bash \
    --dockcmd=./tensorflow_mnode/hvd_mnist_example.sh

singularity exec --nv --cleanenv \
    /cm/shared/singularity/tf1.8.0py3.simg \
    ./tensorflow_mnode/hvd_mnist_example.sh

# MULTI NODE

## INSIDE-OUT
nodes=($(scontrol show hostname $SLURM_NODELIST)) && nodes=${nodes[@]} && nodes=${nodes// /,}
export NODES=$nodes
# --privileged \
# --dockopts="'--cap-add=IPC_LOCK --device=/dev/infiniband'" \
PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-p 22" pdsh -w $NODES NODES=$NODES PATH=$PATH \
    pdsh_docker.sh --noderank=%n \
        --container=nvcr.io/nvidian/sae/avolkov:tf1.12.0py3_cuda10.0_cudnn7_ubuntu16_nccl2.3.7_hvd_ompi3_ibverbs \
        --ibdevices \
        --script=./tensorflow_mnode/hvd_mnist_example.sh \
        --workingdir=${PWD}

# salloc needs to be done prio to srun commands.
salloc -N 2 -p some-partition

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:tf19.09py3_tf1.14.0_ssh \
    --ibdevices \
    --script=./tensorflow_mnode/hvd_mnist_example.sh

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:tf1.12.0py3_cuda10.0_cudnn7_ubuntu16_nccl2.3.7_hvd_ompi3_ibverbs \
    --ibdevices \
    --script=./tensorflow_mnode/hvd_mnist_example.sh

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:tf1.8.0py3_cuda9.0_cudnn7_nccl2.2.13_hvd_ompi3_ibverbs \
    --privileged \
    --script=./tensorflow_mnode/hvd_mnist_example.sh

module load singularity/3.1.0
srun srun_singularity.sh \
    --container=/cm/shared/singularity/tf1.8.0py3.simg \
    --script=./tensorflow_mnode/hvd_mnist_example.sh

## OUTSIDE-IN
salloc -N 2 -p some-partition
# module unload xalt
module load PrgEnv/GCC+OpenMPI/2018-05-24
module load openmpi/3.1.0
module load singularity/3.1.0
# interactive run
srun --ntasks-per-node=8 --pty bash
# -np 8 for one DGX-1. Use 16 for two DGX-1s
NCCL_SOCKET_IFNAME=^docker0,lo,virbr0 \
  mpirun -mca btl_tcp_if_exclude docker0,lo,virbr0 -x NCCL_SOCKET_IFNAME \
    -np 16 singularity exec --nv \
    /cm/shared/singularity/tf1.8.0py3.simg bash -c '
python ./tensorflow_mnode/tensorflow_mnist.py
'

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
    python ./tensorflow_mnode/tensorflow_mnist.py "$@"
