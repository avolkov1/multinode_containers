#!/bin/bash

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


export BUILDIR=${_basedir}/../build
mkdir -p ${BUILDIR}

pushd ${BUILDIR}

export DOCKFILE=Dockerfile.sshconfig

cat <<'EOF' > ${DOCKFILE}
FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-client openssh-server && \
    mkdir -p /var/run/sshd

EOF

docker build -t ssh_container -f $DOCKFILE ${PWD}

export MPISSHCONFIG=mpisshconfig

rm -r ${MPISSHCONFIG}

docker container create --name extract ssh_container
docker container cp extract:/etc/ssh ${MPISSHCONFIG}
docker container rm -f extract

docker rmi ssh_container

# modify the sshd_config file
sed -i 's|Port 22|Port 12345|' ${MPISSHCONFIG}/sshd_config

sed -i "s|HostKey /etc/ssh/ssh_host_rsa_key|HostKey ${HOME}/mpisshconfig/ssh_host_rsa_key|" ${MPISSHCONFIG}/sshd_config
sed -i "s|HostKey /etc/ssh/ssh_host_dsa_key|HostKey ${HOME}/mpisshconfig/ssh_host_dsa_key|" ${MPISSHCONFIG}/sshd_config
sed -i "s|HostKey /etc/ssh/ssh_host_ecdsa_key|HostKey ${HOME}/mpisshconfig/ssh_host_ecdsa_key|" ${MPISSHCONFIG}/sshd_config
sed -i "s|HostKey /etc/ssh/ssh_host_ed25519_key|HostKey ${HOME}/mpisshconfig/ssh_host_ed25519_key|" ${MPISSHCONFIG}/sshd_config

sed -i 's|PermitRootLogin prohibit-password|PermitRootLogin yes|' ${MPISSHCONFIG}/sshd_config
sed -i 's|StrictModes yes|StrictModes no|' ${MPISSHCONFIG}/sshd_config


ls -l ${MPISSHCONFIG}
echo COPY ${MPISSHCONFIG} TO HOME

popd
