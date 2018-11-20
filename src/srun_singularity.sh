#!/bin/bash

# to run:
#     salloc -N 2 -p some_queue  # salloc -N 2 -p hsw_v100
#     srun srun_singularity.sh

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

iname="${USER}_sing"

container=""

envlist=""

singopts=''

datamnts=""

slots_per_node=''

nodelist=""

script=""

script_help=false

sshconfigdir="${HOME}/mpisshconfig"

sshdport=12345


usage() {
cat <<EOF
Usage: $(basename $0) [-h|--help]
    [--iname=name] [--container=singularity-container] [--datamnts=dir1,dir2,...]
    [--envlist=env1,env2,...] [--singopts="--someopt1=opt1 --someopt2=opt2"]
    [--slots_per_node=somenum]
    [--script=scriptpath] [--<remain_args>]

    Use equal sign "=" for arguments, do not separate by space.

    SLURM srun launcher script to orchestrate multinode singularity jobs.
    This script orchestrates multinode jobs with ability to invoke MPI internally
    to the singularity container. This is achieved by running the singularity
    containers as services. Refer to:
        https://www.sylabs.io/guides/2.6/user-guide/running_services.html

    Typical MPI usage model with singularity is to call mpirun from outside the
    container:
        https://www.sylabs.io/guides/2.6/user-guide/faq.html#why-do-we-call-mpirun-from-outside-the-container-rather-than-inside
    But it is possible to setup mpirun from within the container as well. This
    script does the "within" approach. Main benefit is that one does not have
    to install/setup MPI outside of the container which can be burdensome and
    complicated.

    The singularity container needs to be built with multinode capabilities.
    Typically this requires an ssh server and client to be installed in the
    container. Example:
        https://github.com/avolkov1/shared_dockerfiles/blob/master/tensorflow/v1.4.0-nvcr/Dockerfile.tf18.03_ssh

    The dockerfile example needs to be converted to singularity by either
    docker2singularity utility or rewrite as singularity recipe/def file.

    Prerequisites for this script:
        1. SLURM - This script is specifically written to be run in SLURM.
        2. ~/.ssh/config - Setup with options for compute nodes. Refer to README.
            This is important. Also requires a customized sshd_config.
            Refer to README.
        3. Singularity 3.x+ is required.

    Typical usage examples:
        salloc -N 2 -p some_queue  # salloc -N 2 -p hsw_v100
        srun srun_singularity.sh \\
            --script=./hvd_example_script.sh \\
            --container=/cm/shared/singularity/tf1.8.0py3.simg

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

    Within the script the mpirun/pdsh/parallel command typically needs to set
    additional environment variables. Do not assumer workers have these set. Ex:
        mpirun -x LD_LIBRARY_PATH -x PATH -x SHELL \\
            -H $hostlist -np $np <remaining options and command>

    Additional gotchas. If the singularity job is killed or ends in an unexpected
    fashion one might have to manually cleanup files in "/var/run/singularity/instances"
    This is a bug of singularity with its instance feature.

    --iname - Name to use when launching container.
        Default: ${iname}

    --container - Singularity container file/url whatever is legal for start command:
            singularity instance start --help
        Required parameter.

    --datamnts - Data directory(s) to mount into the container. Comma separated.
        Ex: "--datamnts=/datasets,/scratch" would map "/datasets:/datasets"
        and "/scratch:/scratch" in the container.

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
        srun -N 2 --exclude=dgx01 srun_singularity.sh --nodelist=dgx[02-03] ...

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
            srun srun_singularity.sh \\
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

    --singopts - Additional singularity options not covered above. These are
        passed to the singularity service session. Use quotes to keep the
        additional options together. Example:
            --sigopts="-B /datasets:/data"
        In the example the "/datasets" is mapped to "/data" in the
        container instead of using "--datamnts".
        Options that are always set are --nv and --cleanenv. For additional
        options refer to:
            singularity instance start --help

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
        --iname ) larguments=yes; iname="$OPTARG"  ;;
        --container ) larguments=yes; container="$OPTARG"  ;;
        --datamnts ) larguments=yes; datamnts="$OPTARG"  ;;
        --envlist ) larguments=yes; envlist="$OPTARG"  ;;
        --slots_per_node ) larguments=yes; slots_per_node="$OPTARG"  ;;
        --nodelist ) larguments=yes; nodelist="$OPTARG"  ;;
        --script ) larguments=yes; script="$OPTARG"  ;;
        --script_help ) larguments=no; script_help=true  ;;
        --sshconfigdir ) larguments=yes; sshconfigdir="$OPTARG"  ;;
        --sshdport ) larguments=yes; sshdport="$OPTARG"  ;;
        --singopts ) larguments=yes;
            singopts="$( cut -d '=' -f 2- <<< "$_OPTION" )";  ;;
        --help ) usage; exit 2 ;;
        --* ) remain_args+=($_OPTION) ;;
        esac
        OPTIND=1
        shift
        ;;
    esac
done

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

procid=$SLURM_PROCID

mntdata=''
if [ ! -z "${datamnts// }" ]; then
    for mnt in ${datamnts//,/ } ; do
        mntdata="-B ${mnt}:${mnt} ${mntdata}"
    done
fi

if [ -z ${slots_per_node:+x} ]; then
    # ngpus_per_node=$(bash -c 'nvidia-smi -L | wc -l')
    slots_per_node=$(bash -c 'nvidia-smi -L | wc -l')
    # slots_per_node=($(bash -c 'nvidia-smi -L | wc -l'))
    # slots_per_node=${slots_per_node[0]}
    # echo slots_per_node: $slots_per_node
fi


if [ -z "$nodelist" ]; then
    allhostsarr_=($(scontrol show hostname $SLURM_NODELIST))
else
    allhostsarr_=($(scontrol show hostname $nodelist))
fi
ntasks=$SLURM_NTASKS
# echo ntasks: $ntasks
allhostsarr=(${allhostsarr_[@]::$ntasks})
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


# envvars=''
if [ ! -z "${envlist// }" ]; then
    for evar in ${envlist//,/ } ; do
        export SINGULARITYENV_${evar}=${!evar}
    done
    export SINGULARITYENV_ENVLIST="${envlist}"
fi

# additional environment variables to export
envstoexport="sshconfigdir,sshdport,hostlist,scriptpath,remainargs,np"
for evar in ${envstoexport//,/ } ; do
    export SINGULARITYENV_${evar}=${!evar}
done


function launch_sing_sess() {
    singularity instance start --nv --cleanenv \
        ${mntdata} ${singopts} ${container} ${iname}
}

if [ "$procid" -eq "0" ]; then
    echo Master $(hostname)

    singularity instance stop -a 2>/dev/null

    launch_sing_sess

    # Verify all the worker nodes are up.
    for _host in ${workerhosts}; do
        ssh -o "StrictHostKeyChecking no" -q $_host exit
        while [ $? -ne 0 ]; do
            ssh -o "StrictHostKeyChecking no" -q $_host exit
        done
    done

    singularity exec --cleanenv ${singopts} instance://${iname} bash -c '
    /usr/sbin/sshd -p $sshdport  -f ${sshconfigdir}/sshd_config

    # /usr/local/openmpi/bin/mpirun -mca btl_tcp_if_exclude docker0,lo \
    #     -np 2 -H $hostlist \
    #     hostname

    # sleep 200
    # printenv
    # echo remainargs: $remainargs
    # echo ${remainargs//:/ }
    ${scriptpath} ${remainargs//:/ }

    # stop sshd: 15 TERM, 9 KILL
    pgrep -af sshd | grep $sshdport | awk '\''{print $1}'\'' | xargs -i kill -15 {}

    '

    singularity instance stop -a

    # Stop sshd in all the workers. This will force the workers to exit.
    for _host in ${workerhosts}; do
        # ssh $_host 'pgrep -af sshd | grep $sshdport | awk '\''{print $1}'\'' | \
        ssh $_host 'pgrep -af sshdport | awk '\''{print $1}'\'' | \
            xargs -i kill -15 {}' || true  # ignore ssh errors
    done

else
    echo Worker $(hostname)

    singularity instance stop -a 2>/dev/null

    launch_sing_sess

    singularity exec --cleanenv ${singopts} instance://${iname} bash -c '
    # printenv
    # echo sshdport: $sshdport
    /usr/sbin/sshd -p $sshdport -f ${sshconfigdir}/sshd_config

    pid=$( pgrep -af sshd | grep $sshdport | awk '\''{print $1}'\'' )

    while ps -p $pid &>/dev/null; do
        sleep 5
    done
    '

    singularity instance stop -a

fi
