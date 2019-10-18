#!/bin/bash

usage() {
cat <<EOF

srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
    --privileged \
    --script=./pytorch_mnode/pytorch_apex_mnist_example.sh

srun srun_singularity.sh \
    --container=/cm/shared/singularity/pytorch_hvd_apex.simg \
    --script=./pytorch_mnode/pytorch_apex_mnist_example.sh

EOF
}

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function join_by { local IFS="$1"; shift; echo "$*"; }

# to avoid warning:
#   WARNING: local probe returned unhandled shell:unknown assuming bash
export SHELL=/bin/bash

evars=''
if [ ! -z "${ENVLIST// }" ]; then
    for evar in ${ENVLIST//,/ } ; do
        evars="-x ${evar} ${evars}"
    done
fi

# echo HOSTLIST: $hostlist
# echo NP: $np

allhostsarr=(${hostlist//,/ })
# echo ALLHOSTSARR: ${allhostsarr[@]}

MASTERHOST=${allhostsarr[0]}
MASTERHOST=(${MASTERHOST//:/ })
# can control NUM_GPUS via --slots_per_node parameter
export NUM_GPUS=${MASTERHOST[1]}
MASTERHOST=${MASTERHOST[0]}

# For PyTorch distributed each process forks processes for number of gpus. Each
# node should have 1 slot hence need to customize the hostlist for PyTorch.
hostarray=()
for _host in ${allhostsarr[@]}; do
    host_=(${_host//:/ })
    hostarray+="${host_[0]}:1 "
done
# echo HOSTARRAY: ${hostarray[@]}

export TORCH_HOSTLIST="$(join_by , ${hostarray[@]})"
# echo TORCH_HOSTLIST: ${TORCH_HOSTLIST}

# Number of processes for mpirun now is just the number of nodes.
export NNODES=${#allhostsarr[@]}

# PyTorch distributed communicates through socket probably for coordination.
# Hence need to set the master node's ip address.
export MASTER_ADDR=$(getent hosts $MASTERHOST | cut -d' ' -f1)
echo MASTER_ADDR: $MASTER_ADDR

export LAUNCHSCRIPT=${_basedir}/_pytorch_mnist_apex.sh

cat <<EOF > ${LAUNCHSCRIPT}
#!/bin/bash

# OpenMPI Environment Var used to determined node rank.
export NODE_RANK=\${OMPI_COMM_WORLD_RANK}
# echo NODE_RANK: \$NODE_RANK

# would be nice to get RDMA and infiniband working
export NCCL_SOCKET_IFNAME=^docker0,virbr0,lo,ib0
export NCCL_IB_DISABLE=1

if [ -e /opt/conda/bin/activate ]; then
    source /opt/conda/bin/activate
fi

python -m torch.distributed.launch \\
    --nproc_per_node=${NUM_GPUS} \\
    --nnodes=${NNODES} \\
    --node_rank=\${NODE_RANK} \\
    --master_addr="${MASTER_ADDR}" \\
    --master_port=4321 \\
  ${_basedir}/pytorch_apex_mnist.py \
    $@  # any remaining pass-through args

EOF

chmod u+x ${LAUNCHSCRIPT}

mpilaunchcmd=$(cat <<EOF
mpirun -x LD_LIBRARY_PATH -x SHELL ${evars} -H $TORCH_HOSTLIST \
    -mca btl_tcp_if_exclude docker0,lo,virbr0 \
    --report-bindings --bind-to none --map-by slot \
    -np $NNODES \
    ${LAUNCHSCRIPT}
EOF
)
echo RUNNING: $mpilaunchcmd

eval $mpilaunchcmd
