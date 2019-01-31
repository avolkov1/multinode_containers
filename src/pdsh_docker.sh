#!/bin/bash

# Example to run:
#     salloc -N 2 -p some_queue  # salloc -N 2 -p hsw_v100
#     nodes=($(scontrol show hostname $SLURM_NODELIST)) && nodes=${nodes[@]} && nodes=${nodes// /,}
#     export NODES=$nodes
#     PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-p 22" pdsh -w $NODES NODES=$NODES \
#         pdsh_docker.sh --noderank=%n \
#             --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
#             --privileged \
#             --script=$(pwd)/pytorch_mnode/pytorch_hvd_mnist_example.sh \
#             --workingdir=${PWD}

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dockname="${USER}_dock"

container=""

envlist=""

privileged=false

dockopts=''

datamnts=""

slots_per_node=''

nodelist=""
noderank=""

script=""

script_help=false

workingdir="${PWD}"

sshconfigdir="${HOME}/mpisshconfig"

sshdport=12345

usage() {
cat <<EOF
Usage: $(basename $0) [-h|--help]
    [--dockname=name] [--container=docker-container] [--datamnts=dir1,dir2,...]
    [--privileged] [--dockopts="--someopt1=opt1 --someopt2=opt2"]
    [--slots_per_node=somenum]
    [--script=scriptpath] [--<remain_args>]

    PDSH launcher script to orchestrate multinode docker jobs. The docker
    container needs to be built with multinode capabilities. Typically this
    requires an ssh server and client to be installed in the container. Example:
        https://github.com/avolkov1/shared_dockerfiles/blob/master/tensorflow/v1.4.0-nvcr/Dockerfile.tf18.03_ssh

    Use equal sign "=" for arguments, do not separate by space.

    Typical usage examples:
        export NODES=node1,node2  # list of nodes by hostname (DNS) or ip-addresses
        PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-p 22" pdsh -w \$NODES NODES=\$NODES \\
            pdsh_docker.sh --noderank=%n \\
                --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \\
                --privileged \\
                --script=./pytorch_mnode/pytorch_hvd_mnist_example.sh \\
                --workingdir=\${PWD}
        # another call undersubsribe GPUs and use GPU 1 (2nd GPU)
        PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-p 22" \\
            pdsh -w \$NODES NODES=\$NODES CUDA_VISIBLE_DEVICES=1 \\
            pdsh_docker.sh --noderank=%n \\
                --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \\
                --privileged \\
                --slots_per_node=1 \\
                --envlist=CUDA_VISIBLE_DEVICES \\
                --script=./pytorch_mnode/pytorch_hvd_mnist_example.sh \\
                --workingdir=\${PWD}

    Prerequisites for this script:
        1. SLURM - This script is specifically written to be run in SLURM.
        2. ~/.ssh/config - Setup with options for compute nodes. Refer to README.
            This is important. Also requires a customized sshd_config.
            Refer to README.
        3. run_dock_asuser.sh - Download this script and add it to the path.
            Recommended place: ~/bin/run_dock_asuser.sh
            https://github.com/avolkov1/helper_scripts_for_containers/blob/master/run_dock_asuser.sh

    Convenience environment variables are injected into the container. Use
    these in the script.
        hostlist - Comma separated list of nodes and slots (slots_per_node). Ex.:
            hostlist="node1:4,node2:4" # this is injected into the container
            mpirun -H $hostlist ... # in the script use it with mpi command.

        np - Number of processes to launch. Calculated as:
            slots_per_node * nnodes
            mpirun -np $np ... # in the script use it with mpi command.

        ENVLIST - If the --envlist option is used the ENVLIST variable is just
            a list of the environment variables. Refer to --envlist option desc.

    --dockname - Name to use when launching container.
        Default: ${dockname}

    --container - Docker container tag/url. Required parameter.

    --datamnts - Data directory(s) to mount into the container. Comma separated.
        Ex: "--datamnts=/datasets,/scratch" would map "/datasets:/datasets"
        and "/scratch:/scratch" in the container.

    --privileged - This option is typically necessary for RDMA. Refer to
        run_dock_asuser --help for more information about this option. With
        some containers seems to cause network issues so disabled by default.
        Nvidia docker ignores NV_GPU and NVIDIA_VISIBLE_DEVICES when run with
        privileged option. Use CUDA_VISIBLE_DEVICES to mask CUDA processes.

    --slots_per_node - When formulating the hostlist array specify slots per
        node. Typically with multigpu jobs 1 slot per GPU so then slots per
        node is number of GPUs per node. This is the default that is
        automatically set. If undersubscribing or oversubscribing GPUs or doing
        model parallelism, or for any other reason specify slots_per_node
        as needed.

    --nodelist - Specify an explicit node list to use. Used in special cases
        when have a multinode allocation, but would like to run on a subset of
        nodes. Example:
        salloc -N 3 -p dgx-1v
        # assume SLURM_NODELIST=dgx[01-03]
        # We want to run on dgx02 and dgx03 but not dgx01.
        srun -N 2 --exclude=dgx01 srun_docker.sh --nodelist=dgx[02-03] ...

    --script - Specify a script to run. Specify script with full or relative
        paths (relative to current working directory). Ex.:
            --script=examples/ex1.py
        This is a required parameter.

    --sshconfigdir - Specify the sshconfig directory. Refer to README for
        detailed explanation. Default: ~/mpisshconfig

    --sshdport - Specify the sshconfig port. This needs to match settings in
        two places:
            1. sshconfigdir/sshd_config
            2. ~/.ssh/config
        Default: ${sshdport}

    --envlist - Environment variable(s) to add into the container. Comma separated.
        Useful for CUDA_VISIBLE_DEVICES for example. These are only guaranteed
        to be available in the master node. Example:
            export MYVAR1=myvar1
            export MYVAR2=myvar2
            srun srun_docker.sh \\
                --container=<some_container> \\
                --script=<some_script> \\
                --envlist=MYVAR1,MYVAR2

        Within the script a convenience environment variable ENVLIST is set.
        Parse the ENVLIST for mpirun command as
            evars=''
            if [ ! -z "${ENVLIST// }" ]; then
                for evar in ${ENVLIST//,/ } ; do
                    evars="-x ${evar} ${evars}"
                done
            fi

            mpirun $evars ...

    --dockopts - Additional docker options not covered above. These are passed
        to the docker service session. Use quotes to keep the additional
        options together. Example:
            --dockopts="--ipc=host -e MYVAR=SOMEVALUE -v /datasets:/data"
        The "--ipc=host" can be used for MPS with nvidia-docker2. Any
        additional docker option that is not exposed above can be set through
        this option. In the example the "/datasets" is mapped to "/data" in the
        container instead of using "--datamnts".

    --script_help - Pass --help to script.

    --<remain_args> - Additional args to pass through to scripts. These must
        not conflict wth args for this launcher script i.e. don't use sshdport
        for script.

    -h|--help - Displays this help.

EOF
}


remain_args=()

while getopts ":h-" arg; do
    case "${arg}" in
    h ) usage
        exit 2
        ;;
    - ) [ $OPTIND -ge 1 ] && optind=$(expr $OPTIND - 1 ) || optind=$OPTIND
        eval _OPTION="\$$optind"
        OPTARG=$(echo $_OPTION | cut -d'=' -f2)
        OPTION=$(echo $_OPTION | cut -d'=' -f1)
        case $OPTION in
        --dockname ) larguments=yes; dockname="$OPTARG"  ;;
        --container ) larguments=yes; container="$OPTARG"  ;;
        --privileged ) larguments=no; privileged=true  ;;
        --datamnts ) larguments=yes; datamnts="$OPTARG"  ;;
        --envlist ) larguments=yes; envlist="$OPTARG"  ;;
        --slots_per_node ) larguments=yes; slots_per_node="$OPTARG"  ;;
        --nodelist ) larguments=yes; nodelist="$OPTARG"  ;;
        --noderank ) larguments=yes; noderank="$OPTARG"  ;;
        --script ) larguments=yes; script="$OPTARG"  ;;
        --script_help ) larguments=no; script_help=true  ;;
        --sshconfigdir ) larguments=yes; sshconfigdir="$OPTARG"  ;;
        --workingdir ) larguments=yes; workingdir="$OPTARG"  ;;
        --sshdport ) larguments=yes; sshdport="$OPTARG"  ;;
        --dockopts ) larguments=yes;
            dockopts="$( cut -d '=' -f 2- <<< "$_OPTION" )";  ;;
        --help ) usage; exit 2 ;;
        --* ) remain_args+=($_OPTION) ;;
        esac
        OPTIND=1
        shift
        ;;
    esac
done

cd "$workingdir"

function join_by { local IFS="$1"; shift; echo "$*"; }

if [ -z "$script" ]; then
    echo "ERROR: JOB SCRIPT NOT SPECIFIED. Specify via --script=<path>. Refer to --help"
    exit 2
fi

if [ -z "$container" ]; then
    echo "ERROR: CONTAINER NOT SPECIFIED. Specify via --container=<container_tag>. Refer to --help"
    exit 2
fi

# grab all other remaning args.
remain_args+=($@)

if [ "$script_help" = true ] ; then
    remain_args+=("--help")

fi

export remainargs="$(join_by : ${remain_args[@]})"
# export remainargs="$remain_args"
# echo remainargs: $remainargs

procid=$noderank


privilegedopt=''
if [ "$privileged" = true ] ; then
    privilegedopt="--privileged"
fi


if [ -z ${slots_per_node:+x} ]; then
    # ngpus_per_node=$(bash -c 'nvidia-smi -L | wc -l')
    slots_per_node=$(bash -c 'nvidia-smi -L | wc -l')
    # slots_per_node=($(bash -c 'nvidia-smi -L | wc -l'))
    # slots_per_node=${slots_per_node[0]}
    # echo slots_per_node: $slots_per_node
fi


#if [ -z "$nodelist" ]; then
#    allhostsarr_=($(scontrol show hostname $SLURM_NODELIST))
#else
#    allhostsarr_=($(scontrol show hostname $nodelist))
#fi
# allhostsarr_=(${NODES//,/ })
# ntasks=$SLURM_NTASKS
# echo ntasks: $ntasks
# allhostsarr=(${allhostsarr_[@]::$ntasks})
allhostsarr=(${NODES//,/ })
# echo allhostsarr: ${allhostsarr[@]}
hostarray=()
# for _host in $(scontrol show hostname $SLURM_NODELIST); do
for _host in ${allhostsarr[@]}; do
    hostarray+="${_host}:${slots_per_node} "
done
# hostlist=$(join_by , ${hostarray})
export hostlist="$(join_by , ${hostarray[@]})"
# echo hostlist: $hostlist

# workerhosts=$(scontrol show hostname $SLURM_NODELIST | tail -n +2)
workerhosts=${allhostsarr[@]:1}
# echo workerhosts: $workerhosts

# export sshconfigdir="${HOME}/mpisshconfig"
export sshconfigdir

export sshdport

export scriptpath="${script}"

nnodes=${#allhostsarr[@]}
# echo nnodes: $nnodes
np=$(( $slots_per_node * $nnodes ))
export np
# echo NP: $np


# on SLURM with gres allocation of GPUs set the NV_GPU to allocated GPUs.
if [ -z ${NV_GPU:+x} ]; then
export NV_GPU="$(nvidia-smi -q | grep UUID | awk '{ print $4 }' | tr '\n' ',')"
fi
# echo NV_GPU: $NV_GPU

envvars="sshconfigdir,sshdport,hostlist,scriptpath,remainargs,np"
# additional environment variables to export
if [ ! -z "${envlist// }" ]; then
    export ENVLIST=$envlist
    for evar in ${envlist//,/ } ; do
        envvars="${envvars},${evar}"
    done
    envvars="${envvars},ENVLIST"
fi

# echo DOCKOPTS: $dockopts
function launch_dock_sess() {
    # --privileged
    run_dock_asuser.sh --dockname=${dockname} ${privilegedopt} --net=host \
        --datamnts="${datamnts}" \
        --envlist=${envvars} \
        --container=${container} --daemon \
        --dockopts="${dockopts}"
}

# echo WORKINGDIR: $(pwd)

if [ "$procid" -eq "0" ]; then
    echo Master $(hostname)

    docker stop ${dockname} 2>&1 >/dev/null
    docker rm ${dockname} 2>&1 >/dev/null

    launch_dock_sess

    # Verify all the worker nodes are up.
    for _host in ${workerhosts}; do
        ssh -o "StrictHostKeyChecking no" -q $_host exit
        while [ $? -ne 0 ]; do
            ssh -o "StrictHostKeyChecking no" -q $_host exit
        done
    done

    docker exec ${dockname} bash -c '
    /usr/sbin/sshd -p $sshdport  -f ${sshconfigdir}/sshd_config

    # /usr/local/openmpi/bin/mpirun -mca btl_tcp_if_exclude docker0,lo \
    #     -np 2 -H $hostlist \
    #     hostname

    # echo remainargs: $remainargs
    # echo ${remainargs//:/ }
    ${scriptpath} ${remainargs//:/ }

    # stop sshd: 15 TERM, 9 KILL
    pgrep -af sshd | grep $sshdport | awk '\''{print $1}'\'' | xargs -i kill -15 {}

    '

    docker stop ${dockname}
    docker rm ${dockname}

    # Stop sshd in all the workers. This will force the workers to exit.
    for _host in ${workerhosts}; do
        # ssh $_host 'pgrep -af sshd | grep $sshdport | awk '\''{print $1}'\'' | \
        ssh $_host 'pgrep -af sshdport | awk '\''{print $1}'\'' | \
            xargs -i kill -15 {}' || true  # ignore ssh errors
    done

else
    echo Worker $(hostname)

    docker stop ${dockname} 2>&1 >/dev/null
    docker rm ${dockname} 2>&1 >/dev/null

    launch_dock_sess

    docker exec ${dockname} bash -c '
    # echo sshdport: $sshdport
    /usr/sbin/sshd -p $sshdport -f ${sshconfigdir}/sshd_config

    pid=$( pgrep -af sshd | grep $sshdport | awk '\''{print $1}'\'' )

    while ps -p $pid &>/dev/null; do
        sleep 5
    done
    '

    docker stop ${dockname}
    docker rm ${dockname}

fi
