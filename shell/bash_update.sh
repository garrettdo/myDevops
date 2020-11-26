#!/bin/bash
cd /usr/local/src/
wget http://ftp.gnu.org/gnu/bash/bash-5.0.tar.gz
tar zxvf bash-5.0.tar.gz
if [ ! -d "bash-5.0" ]; then
    echo "执行失败原因:没有bash5.0目录"
    exit 1
fi
cd bash-5.0
./configure&&make&&make install
mv /bin/bash /bin/bash.bak
ln -s /usr/local/bin/bash /bin/bash
echo "执行完毕"
rm -rf /tmp/bash_update.sh

