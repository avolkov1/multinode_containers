#!/bin/bash

usage() {
cat <<EOF

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

# using ppr doesn't work with ompi3.0.0 b/c of bug:
#   https://github.com/open-mpi/ompi/pull/4293
# --report-bindings --bind-to none --map-by ppr:4:socket \
# Container should have ompi3.1.x for the ppr binding to work.
mpirun -x LD_LIBRARY_PATH -x SHELL ${evars} -H $hostlist -np $np \
    -mca btl_tcp_if_exclude docker0,lo,virbr0 \
    --report-bindings --bind-to none --map-by slot \
    python ./tensorflow_mnode/tensorflow_mnist.py "$@"
