#!/bin/bash
# usage: ./record_dds.sh  /home/perception/wuchang/dds_store/DDS_FullStack/ /tmp/record 11 11 Heart*,CameraPerception*

trap "kill 0" EXIT
source ../env.sh

# try to get dds store path
if [ -z "$1" ]
  then
    echo "No input for dds store path, exit now......"
    exit
fi
DDS_STORE_DIR_REPLAY=$1
echo "replay path: ${DDS_STORE_DIR_REPLAY}"

DDS_STORE_DIR_RECORD=$2
echo "record path: ${DDS_STORE_DIR_RECORD}"

# try to get domain id in recording
re='^[0-9]+$'
if ! [[ $3 =~ $re ]]
  then
    echo "No input for domain id, exit now......"
    exit
fi
REPLAY_DOMAIN_ID=$3
echo "domain to replay: ${REPLAY_DOMAIN_ID}"

# try to get domain id in recording
re='^[0-9]+$'
if ! [[ $4 =~ $re ]]
  then
    echo "No input for domain id, exit now......"
    exit
fi
RECORD_DOMAIN_ID=$4
echo "domain to record: ${RECORD_DOMAIN_ID}"

# try to exclude topics in replaying
if ! [[ -z "$5" ]]
  then
    REPLAY_EXCLUDE_TOPIC="-e "$5
  else
    REPLAY_EXCLUDE_TOPIC=""
fi
echo "topics to exclude in replaying: ${REPLAY_EXCLUDE_TOPIC}"

# try to exclude topics in recording
if ! [[ -z "$6" ]]
  then
    RECORD_EXCLUDE_TOPIC="-e "$6
  else
    RECORD_EXCLUDE_TOPIC=""
fi
echo "topics to exclude in recording: ${RECORD_EXCLUDE_TOPIC}"

HOST_PATH=perception@${HOST_IP}:${HOST_FOLDER}
CP_DIR=$TOPDIR"/../bazel-bin/xpilot/modules/perception/run_perception"
RTIREPLAY_COMMAND="/opt/toolchains/rti_connext_dds-6.0.0/bin/rtireplayservice"
RTIRECORD_COMMAND="/opt/toolchains/rti_connext_dds-6.0.0/bin/rtirecordingservice"

# run xcamera
echo "now run xcamera......"
sleep 3
sshpass -p chengzi0109 ssh -tt perception@${HOST_IP} "echo 'enter HOST(perception@${HOST_IP})';
sshpass -p xpxpu ssh -tt root@172.20.1.22 'echo 'enter Xavier';
export NDDS_DOMAIN_ID=${REPLAY_DOMAIN_ID}; /xpilot/bin/xcamera -i'" &

# run perception
echo "now run perception......"
sleep 3
sshpass -p chengzi0109 ssh -tt perception@${HOST_IP} "echo 'enter HOST(perception@${HOST_IP})'; 
cd files_to_Xavier;
sshpass -p xpxpu scp -c aes128-ctr run_perception root@172.20.1.22:/xpilot/bin/perception/;
sshpass -p xpxpu scp -c aes128-ctr *xml root@172.20.1.22:/xpilot/rti;
sshpass -p xpxpu scp -c aes128-ctr -r config/. root@172.20.1.22:/xpilot/config/perception;
sshpass -p xpxpu ssh -tt root@172.20.1.22 'echo 'enter Xavier'; cd /xpilot/bin/perception/; 
rm /tmp/*core; rm /xpilot/dumps/run_perception*core; . /etc/.kshrc; export NDDS_DOMAIN_ID=${RECORD_DOMAIN_ID};
./run_perception -v 3 -c /xpilot/config/perception/SOP/Unified_ET1_Guangzhou.yaml -p'" &

# run dds record
echo "now run dds record......"
sleep 3
cm="./rti_record.sh ${DDS_STORE_DIR_RECORD} ${RECORD_DOMAIN_ID} -R ${RTIRECORD_COMMAND} ${RECORD_EXCLUDE_TOPIC}"
echo ${cm}
sshpass -p chengzi0109 ssh -tt perception@${HOST_IP} "echo 'enter HOST(perception@${HOST_IP})';
cd /home/perception/xplorer/tools/record_replay_tools;
export NDDS_DOMAIN_ID=128;
./rti_record.sh ${DDS_STORE_DIR_RECORD} ${RECORD_DOMAIN_ID} -R ${RTIRECORD_COMMAND} ${RECORD_EXCLUDE_TOPIC}" &

# run dds replay
sleep 45
echo "now run dds replay......"
cmm="./rti_replay.sh ${DDS_STORE_DIR_REPLAY} -v 6 ${REPLAY_DOMAIN_ID} -R ${RTIREPLAY_COMMAND} ${REPLAY_EXCLUDE_TOPIC}"
echo ${cmm}

sshpass -p chengzi0109 ssh -tt perception@${HOST_IP} "echo 'enter HOST(perception@${HOST_IP})';
cd /home/perception/xplorer/tools/record_replay_tools;
export NDDS_DOMAIN_ID=128;
./rti_replay.sh ${DDS_STORE_DIR_REPLAY} -v 6 ${REPLAY_DOMAIN_ID} -R ${RTIREPLAY_COMMAND} ${REPLAY_EXCLUDE_TOPIC}" 

wait
