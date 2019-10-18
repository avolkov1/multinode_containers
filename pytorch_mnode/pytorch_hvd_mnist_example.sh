#!/bin/bash

usage() {
cat <<EOF

# SINGLE NODE
run_dock_asuser.sh \
    --dockname=avolkov_pytorch \
    --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
    --noninteractive \
    --entrypoint=bash \
    --dockcmd=./pytorch_mnode/pytorch_hvd_mnist_example.sh

singularity exec --nv --cleanenv \
    /cm/shared/singularity/pytorch_hvd_apex.simg \
    bash -c '
./pytorch_mnode/pytorch_hvd_mnist_example.sh
'

# MULTI NODE

## INSIDE-OUT
nodes=($(scontrol show hostname $SLURM_NODELIST)) && nodes=${nodes[@]} && nodes=${nodes// /,}
export NODES=$nodes
PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-p 22" pdsh -w $NODES NODES=$NODES \
    pdsh_docker.sh --noderank=%n \
        --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
        --privileged \
        --script=./pytorch_mnode/pytorch_hvd_mnist_example.sh \
        --workingdir=${PWD}

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
    --privileged \
    --script=./pytorch_mnode/pytorch_hvd_mnist_example.sh

srun srun_singularity.sh \
    --container=/cm/shared/singularity/pytorch_hvd_apex.simg \
    --script=./pytorch_mnode/pytorch_hvd_mnist_example.sh

## OUTSIDE-IN
salloc -N 2 -p some-partition
# module unload xalt
module load PrgEnv/GCC+OpenMPI/2018-05-24
module load openmpi/3.1.0
module load singularity/3.1.0
srun --ntasks-per-node=8 --pty bash
#-np 8 for one DGX-1. Use 16 for two DGX-1s
mpirun -mca btl_tcp_if_exclude docker0,lo,virbr0 \
    -np 16 singularity exec --nv \
    /cm/shared/singularity/pytorch_hvd_apex.simg bash -c '
python ./pytorch_mnode/pytorch_hvd_mnist.py
'

EOF
}

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# to avoid warning: WARNING: local probe returned unhandled shell:unknown assuming bash
export SHELL=/bin/bash

export NCCL_SOCKET_IFNAME=^docker0,virbr0,lo
export NCCL_IB_DISABLE=0

evars=''
if [ ! -z "${ENVLIST// }" ]; then
    for evar in ${ENVLIST//,/ } ; do
        evars="-x ${evar} ${evars}"
    done
fi

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
# --report-bindings --bind-to socket --map-by ppr:2:socket \
# --report-bindings --bind-to none --map-by slot \
mpirun -x PATH -x LD_LIBRARY_PATH -x SHELL ${evars} $hostlistopts \
    -mca btl_tcp_if_exclude docker0,lo,virbr0 \
    -x NCCL_SOCKET_IFNAME \
    -x NCCL_IB_DISABLE \
    --report-bindings --bind-to none --map-by slot \
    -np ${np} \
    python ./pytorch_mnode/pytorch_hvd_mnist.py \
    $@

# options: --epochs=1 --batch_size=256
