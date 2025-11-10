#!/bin/bash -xe

# From https://askubuntu.com/questions/1385440/ubuntu-sudo-apt-get-update-404-not-found-problem
DIR=/tmp/fix_apt_install
mkdir $DIR
cd $DIR

cat << EOF > $DIR/sources.list
deb http://archive.ubuntu.com/ubuntu/ focal main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu/ focal main restricted universe multiverse

deb http://archive.ubuntu.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu/ focal-updates main restricted universe multiverse

deb http://archive.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse

deb http://archive.ubuntu.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu/ focal-backports main restricted universe multiverse

deb http://archive.canonical.com/ubuntu focal partner
deb-src http://archive.canonical.com/ubuntu focal partner
EOF

#sed -i "s/focal/$(lsb_release -c -s)/" $DIR/sources.list
rm /etc/apt/sources.list
cp $DIR/sources.list /etc/apt/sources.list

#mv /etc/apt/sources.list.d/* $DIR || true

apt update
