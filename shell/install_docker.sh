#!/bin/bash

set -e
echo "" > /tmp/docker-install.log
# 0. prepare user docker
#id docker >/dev/null 2>&1 || (
#    chattr -i /etc/passwd;
#    chattr -i /etc/shadow;
#    chattr -i /etc/group;
#    chattr -i /etc/gshadow;
#    groupadd docker
#    useradd -r -s /sbin/nologin docker -g docker;
#    chattr +i /etc/passwd;
#    chattr +i /etc/shadow;
#    chattr +i /etc/group;
#    chattr +i /etc/gshadow;
#    )
echo "0. prepare user docker" >> /tmp/docker-install.log

# 1. remove old version
yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
echo "1. remove old version" >> /tmp/docker-install.log

# 2. install utils and repo
yum install -y yum-utils \
  device-mapper-persistent-data \
  lvm2
yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo
yum-config-manager --disable docker-ce-edge
yum-config-manager --disable docker-ce-test
yum-config-manager --disable docker-ce-nightly
echo "2. install utils and repo" >> /tmp/docker-install.log

# 3. install docker-ce
VERSION_STRING=19.03.12
yum install -y docker-ce-${VERSION_STRING} docker-ce-cli-${VERSION_STRING} containerd.io
echo "3. install docker-ce" >> /tmp/docker-install.log

# 4. install docker-compose
if [ ! -f /usr/local/bin/docker-compose ];then
curl -s -L "https://github.com/docker/compose/releases/download/1.26.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
fi
chmod 755 /usr/local/bin/docker-compose
[[ -e /usr/bin/docker-compose ]] && (
  mv /usr/bin/docker-compose /home/bak_delete/docker-compose-`date +%Y%m%d-%H%M%S` && \
  ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose)
pip install --upgrade pip
pip install --ignore-installed requests
pip install six --user -U
which pip || (curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python /tmp/get-pip.py)
pip install docker-compose==1.26.2 --ignore-installed
echo "4. install docker-compose" >> /tmp/docker-install.log

[[ -d /etc/docker ]] || mkdir -p /etc/docker

cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ]
}
EOF

[[ -d /etc/systemd/system/docker.service.d ]] || mkdir -p /etc/systemd/system/docker.service.d

cat << EOF > /etc/systemd/system/docker.service.d/network.conf
[Service]
ExecStartPost=-/usr/bin/docker network create --driver=bridge --subnet=10.255.1.0/24 --ip-range=10.255.1.0/24 --gateway=10.255.1.254 internal
ExecStartPost=-/usr/bin/docker network create --driver=bridge --subnet=10.255.2.0/24 --ip-range=10.255.2.0/24 --gateway=10.255.2.254 production
EOF


# 5. start docker
systemctl daemon-reload
systemctl enable docker
systemctl start docker
echo "5. start docker" >> /tmp/docker-install.log

# 6. make sure iptables rules not lost when restart docker
sed -i 's/^IPTABLES_SAVE_ON_RESTART=.*$/IPTABLES_SAVE_ON_RESTART=\"yes\"/g' /etc/sysconfig/iptables-config
sed -i 's/^IPTABLES_SAVE_ON_STOP=.*$/IPTABLES_SAVE_ON_STOP=\"yes\"/g' /etc/sysconfig/iptables-config
#iptables-save|grep ":DOCKER-FILTER"
#res0=$?
#iptables-save|grep "PREROUTING -m addrtype --dst-type LOCAL -j RETURN"
#res1=$?
#iptables-save|grep "PREROUTING -m addrtype --dst-type LOCAL -j DOCKER-FILTER"
#res2=$?
#if [ $res0 -eq 1 ]; then
#  iptables -t nat -N DOCKER-FILTER
#  iptables -t nat -I PREROUTING -m addrtype --dst-type LOCAL -j RETURN
#  iptables -t nat -I PREROUTING -m addrtype --dst-type LOCAL -j DOCKER-FILTER
#elif [ $res1 -eq 1 ] && [ $res2 -eq 1 ]; then
#  iptables -t nat -I PREROUTING -m addrtype --dst-type LOCAL -j RETURN
#  iptables -t nat -I PREROUTING -m addrtype --dst-type LOCAL -j DOCKER-FILTER
#fi

echo "6. make sure iptables rules not lost when restart docker end " >> /tmp/docker-install.log

# 7. init file
mkdir /home/data/data
ln -s /home/data/data /data
[[ -d /data/docker ]] || mkdir /data/docker
[[ -d /data/docker/runtime ]] || mkdir /data/docker/runtime
[[ -d /data/docker/yml ]] || mkdir /data/docker/yml
[[ -d /data/docker/data ]] || mkdir /data/docker/data
[[ -d /data/docker/bin ]] || mkdir /data/docker/bin
echo "7. init file " >> /tmp/docker-install.log

docker login --username reg_user01 --password VOtern93 reg.easydevops.net

