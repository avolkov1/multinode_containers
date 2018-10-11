
# Table of Contents
-   [SLURM Multinode Docker Container Orchestration](#slurm-multinode-docker-container-orchestration)
    -   [User sshd config files for containers](#user-sshd-config-files-for-containers)
    -   [User ssh configuration](#user-ssh-configuration)
    -   [Setup helper script `run_dock_asuser`](#setup-helper-script-run_dock_asuser)
-   [Building Containers with ssh for multinode capability](#building-containers-with-ssh-for-multinode-capability)
-   [Tensorflow multinode using Docker Containers](#tensorflow-multinode-using-docker-containers)
-   [PyTorch multinode using Docker Containers](#pytorch-multinode-using-docker-containers)
-   [Advanced usage of `srun`](#advanced-usage-of-srun)

### SLURM Multinode Docker Container Orchestration

This repo provides examples and scripts to orchestrate docker containers that
work across nodes. The docker options and session launcher boiler code is done
via orchestration helper script: [`srun_docker.sh`](src/srun_docker.sh)

A typical interactive Slurm run looks like this:

``` bash
salloc -N 2 -p <some_partition> # using two nodes
srun srun_docker.sh \
    --container=<your_container> \
    --privileged \
    --script=./<your_job_script>.sh
```

*Procedure*

-   Srun (sbatch also works) the `srun_docker.sh` script on allocated nodes:
    -   First node acts as the "master"/coordinator and "worker" node.
    -   Remaining nodes are worker nodes.
-   Workers nodes
    -   Start container run sshd and sleep/wait within the container.
    -   The container is running and processes can be launched within the context
        of this container service. When the mpirun is launched in master
        node it will startup processes in these worker nodes.
-   Master node
    -   Start container
    -   Run a loop trying to ssh to worker nodes to verify that those are
        running.
    -   Start sshd within the container and launch the job script
        within the container.
    -   Once the job script finishes, stop sshd, ssh to the workers and kill
        their sessions, and finally stop/rm launched containers.

There are numerous examples of the general idea of this approach such as:<br/>
<https://github.com/uber/horovod/blob/master/docs/docker.md#running-on-multiple-machines>

Most of these published examples run the containers as root. This makes it
difficult for working with linux systems where user permissions are relied on
for access to data and code. The `srun_docker.sh` script with the configuration
described below orchestrates the containers to run with the users id and
privileges.

##### User sshd config files for containers

Setup an `mpisshconfig` directory to enable generic sshd user configurations
for containers. A convenience script [`create_mpisshconfig.sh`](src/create_mpisshconfig.sh)
is provided to do this. Just run the script and it will generate an `mpisshconfig`
directory that can be copied somewhere within the user's home directory.

The script `srun_docker.sh` has an option `--sshconfigdir` that can be set
to match this location (or modify the `srun_docker.sh` with a desired default
value).

It is important to have correct permissions on the config files:

``` bash
$ tree -p ~/mpisshconfig/
$HOME/mpisshconfig/
├── [-rw-r--r--]  moduli
├── [-rw-r--r--]  ssh_config
├── [-rw-r--r--]  sshd_config
├── [-rw-------]  ssh_host_dsa_key
├── [-rw-r--r--]  ssh_host_dsa_key.pub
├── [-rw-------]  ssh_host_ecdsa_key
├── [-rw-r--r--]  ssh_host_ecdsa_key.pub
├── [-rw-------]  ssh_host_ed25519_key
├── [-rw-r--r--]  ssh_host_ed25519_key.pub
├── [-rw-------]  ssh_host_rsa_key
├── [-rw-r--r--]  ssh_host_rsa_key.pub
# a bunch of other files
```

These files were just copied from a container's `/etc/ssh` with ssh installed.
What matters most are the settings in `sshd_config`. The key files can be
regenerated.

The `HostKey` paths need to correspond to your home directory. The port needs to
be set to match `~/.ssh/config` (more on port settings below), `PermitRootLogin`
should be set to yes, and `StrictModes` set to no.

``` bash
# CHANGE THIS IN sshd_config from avolkov to your username
HostKey /home/avolkov/mpisshconfig/ssh_host_rsa_key
HostKey /home/avolkov/mpisshconfig/ssh_host_dsa_key
HostKey /home/avolkov/mpisshconfig/ssh_host_ecdsa_key
HostKey /home/avolkov/mpisshconfig/ssh_host_ed25519_key
```

Again, the [`create_mpisshconfig.sh`](src/create_mpisshconfig.sh) script sets
up the `sshd_config` file with the above modifications. Refer to it for
reference and customizations.

Under the hood the `srun_docker.sh` launches `sshd` within containers as:

``` bash
/usr/sbin/sshd -p $sshdport  -f ${sshconfigdir}/sshd_config
```

##### User ssh configuration

The multinode containers orchestration relies on ssh communication between
them. One way to setup a generic user oriented ssh authentication is via ssh
config. Suppose (use `sinfo` on SLURM to view the partition and node names) the
compute nodes on partition `dgx-1v` are called dgx\[01-04\], and another
partition `hsw_v100` has nodes hsw\[01-20\], one can set default user ssh configs
in the `~/.ssh/config` file:

``` bash
# FILE: ~/.ssh/config
# Generate ~/.ssh/id_rsa_mpi via:
#     ssh-keygen -f ${HOME}/.ssh/id_rsa_mpi -t rsa -b 4096 -C "your_email@example.com"
Host dgx* hsw*
    Port 12345
    PubKeyAuthentication yes
    StrictHostKeyChecking no
    # UserKnownHostsFile /dev/null
    UserKnownHostsFile ~/.ssh/known_hosts
    IdentityFile ~/.ssh/id_rsa_mpi
```

The user can change the port to anything they desire instead of `12345`.
Assuming that ssh only works for allocated nodes, the port value should just
be set to something that does not conflict with the running applications. One
should not have to worry about conflicting ports with other users unless the
nodes are setup in non-exclusive mode on SLURM. In the non-exclusive setups
users would need to somehow coordinate or insure their ports do not conflict.
The script `srun_docker.sh` has an option `--sshdport` that can be set to match
this port (or modify the `srun_docker.sh` with a desired default value). This
port value should also be set in the `sshd_config` file as described previously.

##### Setup helper script `run_dock_asuser`

Running containers as user requires setting a few additional options when
launching a docker container. For example:

``` bash
USEROPTS="-u $(id -u):$(id -g) -e HOME=$HOME -e USER=$USER -v $HOME:$HOME"
getent group > group
getent passwd > passwd
USERGROUPOPTS="-v $PWD/passwd:/etc/passwd:ro -v $PWD/group:/etc/group:ro"

docker run --rm -it $USEROPTS $USERGROUPOPTS <otheropts> <somecontainer-and-cmds> 
```

Once inside the launched container as above, the user will appear with their id
instead of typical root user. The above settings are somewhat verbose therefore
a wrapper script for launching a container as user is used. The
`srun_docker.sh` uses `run_dock_asuser.sh` to launch the docker service
sessions. Please download the script `run_dock_asuser.sh` from this
location:<br/>
<https://github.com/avolkov1/helper_scripts_for_containers/blob/master/run_dock_asuser.sh>

Place the script somewhere on the PATH. Recommended location: `~/bin/run_dock_asuser.sh`
The `$HOME/bin` directory is typically added to the user's PATH in
`~/.bash_profile`. If not then modify either `~/.bash_profile` or `~/.bashrc` to
append `$HOME/bin` to PATH. `PATH=$PATH:$HOME/bin`

The [`srun_docker.sh`](src/srun_docker.sh) script also should be installed
somewhere on the PATH. The `$HOME/bin` directory is a good location for the
`srun_docker.sh` script as well.

### Building Containers with ssh for multinode capability

The ssh approach is used in these demos to enable containers to
communicate across node boundaries within a cluster. This requires that
containers have ssh installed. The typical Dockerfile commands to do this are:

``` dockerfile
# some Dockerfile
# FROM ...
# setup your application/framework/library
FROM ubuntu:16.04

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-client openssh-server && \
    mkdir -p /var/run/sshd && \
    rm -rf /var/lib/apt/lists/*

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# ref: https://docs.docker.com/engine/examples/running_ssh_service/#build-an-eg_sshd-image
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
```

Further references for multinode containers and setup:

-   Dockerize SSH Service - <https://docs.docker.com/engine/examples/running_ssh_service/>

-   Mellanox Docker Support - <https://community.mellanox.com/docs/DOC-2971><br/>
    Refer to the dockerfile example with install command:

    ``` bash
    ${MOFED_DIR}/mlnxofedinstall --user-space-only --without-fw-update --all -q
    ```

### Tensorflow multinode using Docker Containers

There are a variety of examples on the internet for setting up multinode docker
containerized Tensorflow workloads. Uber posted instructions for Horovod:<br/>
<https://github.com/uber/horovod/blob/master/docs/docker.md#running-on-multiple-machines>

The examples below demonstrate how to do this on a SLURM cluster. A variety of
Dockerfiles with Tensorflow that also install ssh and can be used for these
demos are posted here:<br/>
<https://github.com/avolkov1/shared_dockerfiles/tree/master/tensorflow>

The dockerfile
[Dockerfile.tf1.8.0py3\_cuda9.0\_cudnn7\_nccl2.2.13\_hvd\_ompi3\_ibverbs](https://github.com/avolkov1/shared_dockerfiles/blob/master/tensorflow/v1.8.0/Dockerfile.tf1.8.0py3_cuda9.0_cudnn7_nccl2.2.13_hvd_ompi3_ibverbs)
is used for the example below.
Typically when working with docker containers in a cluster environment, one
needs to build and push the container to a docker registry from where the
compute nodes will have access to the registry. The example below pushes the
container to a private registry space on NGC (Nvidia GPU Cloud registry):

``` bash
export TAG=tf1.8.0py3_cuda9.0_cudnn7_nccl2.2.13_hvd_ompi3_ibverbs
docker build \
  -t nvcr.io/nvidian/sae/avolkov:$TAG \
  -f Dockerfile.${TAG} \
  $(pwd)

docker push nvcr.io/nvidian/sae/avolkov:${TAG}
```

A sample work script [`hvd_mnist_example.sh`](tensorflow_mnode/hvd_mnist_example.sh)
has been provided. Here is an example for running on a SLURM cluster:

``` bash
salloc -N 2 -p dgx-1v  # 2 nodes. Change -N to however many nodes you would like.
# use [--<remain_args>] for passing additional parameters to script
srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:tf1.8.0py3_cuda9.0_cudnn7_nccl2.2.13_hvd_ompi3_ibverbs \
    --privileged \
    --script=./tensorflow_mnode/hvd_mnist_example.sh
```

The script uses the Horovod `tensorflow_mnist.py` example that can be found here:<br/>
<https://github.com/uber/horovod/blob/master/examples/tensorflow_mnist.py>

The repo example [`tensorflow_mnist.py`](tensorflow_mnode/tensorflow_mnist.py)
has been slightly modified with a barrier to avoid downloading mnist dataset
redundantly.

Note the options `--container` and `--script`. These are required.
For additional instructions and help refer to:

``` bash
srun_docker.sh --help
```

Some other options to note are:

``` bash

    --slots_per_node - When formulating the hostlist array specify slots per
        node. Typically with multigpu jobs 1 slot per GPU so then slots per
        node is number of GPUs per node. This is the default that is
        automatically set. If undersubscribing or oversubscribing GPUs or doing
        model parallelism, or for any other reason specify slots_per_node
        as needed.

    --datamnts - Data directory(s) to mount into the container. Comma separated.
        Ex: "--datamnts=/datasets,/scratch" would map "/datasets:/datasets"
        and "/scratch:/scratch" in the container.

    --dockopts - Additional docker options not covered above. These are passed
        to the docker service session. Use quotes to keep the additional
        options together. Example:
            --dockopts="--ipc=host -e MYVAR=SOMEVALUE -v /datasets:/data"
        The "--ipc=host" can be used for MPS with nvidia-docker2. Any
        additional docker option that is not exposed above can be set through
        this option. In the example the "/datasets" is mapped to "/data" in the
        container instead of using "--datamnts".

    --privileged - This option is typically necessary for RDMA. Refer to
        run_dock_asuser --help for more information about this option. With
        some containers seems to cause network issues so disabled by default.
        Nvidia docker ignores NV_GPU and NVIDIA_VISIBLE_DEVICES when run with
        privileged option. Use CUDA_VISIBLE_DEVICES to mask CUDA processes.

    --<remain_args> - Additional args to pass through to scripts. These must
        not conflict wth args for this launcher script i.e. don't use sshdport
        for script.

    --script_help - Pass --help to script.
```

One can change which code to run, which container to use, what
directories/volumes to mount (home path is automatically mounted), how many
slots (slots are typically mapped to GPUs) to use, etc. In the script when orchestrating `mpirun` use the
injected environment variables `hostlist` and `np` for convenience.

``` bash
# If a different shell is used then don't set shell to bash.
# to avoid warning: WARNING: local probe returned unhandled shell:unknown assuming bash
export SHELL=/bin/bash

mpirun -H $hostlist -x SHELL -np $np \
# etc.
```

### PyTorch multinode using Docker Containers

Running Horovod code with PyTorch is very similar to running with Tensorflow.
The dockerfile
[Dockerfile.pytorch\_hvd\_apex](https://github.com/avolkov1/shared_dockerfiles/blob/master/pytorch/Dockerfile.pytorch_hvd_apex)
is used for the example below. Again, similar to Tensorflow case the container
should be built and pushed to a registry accessible by compute nodes.

``` bash
export TAG=pytorch_hvd_apex
docker build \
  -t nvcr.io/nvidian/sae/avolkov:$TAG \
  -f Dockerfile.${TAG} \
  $(pwd)

docker push nvcr.io/nvidian/sae/avolkov:${TAG}
```

A sample work script [`pytorch_hvd_mnist_example.sh`](pytorch_mnode/pytorch_hvd_mnist_example.sh)
has been provided. Example for running on a SLURM cluster:

``` bash
salloc -N 2 -p dgx-1v  # 2 nodes. Change -N to however many nodes you would like.
srun srun_docker.sh \
    --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
    --privileged \
    --script=./pytorch_mnode/pytorch_hvd_mnist_example.sh
```

The script uses the Horovod `pytorch_hvd_mnist.py` code based on
`pytorch_mnist.py` example that can be found here:<br/>
<https://github.com/uber/horovod/blob/master/examples/pytorch_mnist.py>

The repo example [`pytorch_hvd_mnist.py`](pytorch_mnode/pytorch_hvd_mnist.py)
has been slightly modified with a barrier so as to not redundantly download data.
Two additional examples are provided:

-   [`pytorch_apex_mnist_example.sh`](pytorch_mnode/pytorch_apex_mnist_example.sh),
    [`pytorch_apex_mnist.py`](pytorch_mnode/pytorch_apex_mnist.py) -

    The reference code is [`main.py`](https://github.com/NVIDIA/apex/blob/aa81713249bb17d715c247adeadc229a37adeefa/examples/distributed/main.py)
    taken from examples here:<br/>
    <https://github.com/NVIDIA/apex>

    The `pytorch_apex_mnist.py` has been modified with a barrier to avoid the
    race condition on downloading mnist datasets.

-   [`pytorch_dist_mnist_example.sh`](pytorch_mnode/pytorch_dist_mnist_example.sh),
    [`pytorch_dist_mnist.py`](pytorch_mnode/pytorch_dist_mnist.py) -

    Same as the apex example above, but modified to use PyTorch distributed
    framework (nccl backend) without Apex for comparison.

These examples demonstrate usage of `mpirun` to run non-MPI code. One could
have used [`pdsh`](https://github.com/chaos/pdsh) instead to the same effect.
The idea is to illustrate that `srun_docker.sh` is a dynamic and versatile
wrapper for enabling one to run multinode containers in various scenarios
on SLURM. Refer to the `pdsh` example for apex (which is not MPI based)
[`pytorch_apex_mnist_example_pdsh.sh`](pytorch_mnode/pytorch_apex_mnist_example_pdsh.sh).

### Advanced usage of `srun`

Above were examples of orchestrating `srun_docker.sh` script via srun. Here is
a list of sometimes useful srun launch commands.

``` bash
salloc -N 2 -p dgx-1v  # 2 nodes. Change -N to however many nodes you would like.

# run on just one node even though 2 nodes are allocated
srun -N 1 srun_docker.sh <additional parameters and options>

# assume nodes dgx01 and dgx02 are allocated. Run on dgx02
srun -N 1 --exclude=dgx01  srun_docker.sh --nodelist=dgx02 <additional parameters and options>

# Using a subset of GPUs without privileged option
NV_GPU=2,3 srun srun_docker.sh \
    --slots_per_node=2 \
    --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
    --script=./pytorch_mnode/pytorch_apex_mnist_example_pdsh.sh --epochs=5

# Using a subset of GPUs with privileged option
# Tensorflow MPI Horovod approach
srun srun_docker.sh \
    --slots_per_node=2  --dockopts="-e CUDA_VISIBLE_DEVICES=2,3" \
    --privileged \
    --container=nvcr.io/nvidian/sae/avolkov:tf1.8.0py3_cuda9.0_cudnn7_nccl2.2.13_hvd_ompi3_ibverbs \
    --script=./tensorflow_mnode/hvd_mnist_example.sh

# PyTorch Non-MPI approach
srun srun_docker.sh \
    --slots_per_node=2 --dockopts="-e CUDA_VISIBLE_DEVICES=2,3" \
    --privileged \
    --container=nvcr.io/nvidian/sae/avolkov:pytorch_hvd_apex \
    --script=./pytorch_mnode/pytorch_apex_mnist_example_pdsh.sh --epochs=5
```

These examples are straightforward to convert to sbatch scripts. Example:

``` bash
sbatch -N 2 -p dgx-1v \
    --output="slurm-%j-pytorch_hvd_mnist.out" \
    --wrap=./pytorch_mnode/sbatch_pytorch.sh
```

Refer to the script [`sbatch_pytorch.sh`](pytorch_mnode/sbatch_pytorch.sh) for
further details.

