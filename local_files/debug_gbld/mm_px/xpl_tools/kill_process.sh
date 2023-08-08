#!/bin/bash

source ./env.sh

# Kill run_perception and xcamera_sim on the target
sshpass -p root ssh root@${TARGET_IP} "echo 'cleanup the perception process on board'; \
    ps | grep 'run_perception' | awk '{print $1}' | xargs slay; \
    ps | grep 'xcamera_sim' | awk '{print $1}' | xargs slay; \
"

# Kill local ssh connection to the target
ps -aef | grep "ssh.*${TARGET_IP}"| grep -v grep | awk '{print $2}' | xargs kill -9
