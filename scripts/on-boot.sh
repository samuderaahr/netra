#!/bin/sh

sudo /home/flash/codes/scripts/nfs-restart.sh
sudo mount -t tmpfs -o rw,size=300M tmpfs /media/flash/ghost
