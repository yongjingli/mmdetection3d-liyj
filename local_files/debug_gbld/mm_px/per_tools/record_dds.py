import sys
import os
import time
import threading
import paramiko
import subprocess
from absl import app
from absl import flags
from absl import logging

# Run the following command to install necessary packages.
# pip3 install paramiko absl-py

FLAGS = flags.FLAGS
flags.DEFINE_string("hostname", None, "XPU host name.")
flags.DEFINE_string("username", None, "Username on XPU.")
flags.DEFINE_string("password", None, "XPU password.")
flags.DEFINE_string("perception_binary_filepath", None, "Percpetion binary on XPU path.")
flags.DEFINE_string("xcamera_binary_filepath", None, "XCamera binary on XPU path.")
flags.DEFINE_string("perception_directory", None, "Absolute path for perception directory.")
flags.DEFINE_string("dds_directory", '/tmp/dds', "Directory to store dds.")
flags.DEFINE_integer("port", 22, "SSH port.")

def line_buffered(f):
    line_buf = ""
    while not f.channel.exit_status_ready():
        line_buf += f.read(1).decode('ascii')
        if line_buf.endswith('\n'):
            yield line_buf
            line_buf = ''

def continous_output(sout):
    for l in line_buffered(sout):
        print(l)

def remote_exec_command(command):
    pass

def local_exec_command(command, working_directory, disable_output=False):
    os.chdir(working_directory)
    logging.info("Working dir: %s", working_directory)
    logging.info("Local exec: %s", command)
    if disable_output:
        FNULL = open(os.devnull, 'w')
        subprocess.call([command], shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    else:
        subprocess.call([command], shell=True)

def remote_exec_command(command):
    logging.info("SSH exec %s", command)
    try:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.WarningPolicy)
        
        client.connect(FLAGS.hostname, port=FLAGS.port, username=FLAGS.username, password=FLAGS.password)

        stdin, stdout, stderr = client.exec_command(command, get_pty=True)
        continous_output(stdout)

    finally:
        client.close()


def local_run_rti_replay():
    working_dir = os.path.join(FLAGS.perception_directory, '../../../tools/record_replay_tools')
    working_dir = os.path.abspath(working_dir)
    command = "bash -c 'export NDDS_DOMAIN_ID=11; ./rti_replay.sh /xmotors_ai_shared/oss/xmotors-hq-data-upload/e28_gz_test/e28_gz_test/e28_gz_test/ota/20200312_V406_DEMO_6cam/0312_V406_DEMO_6cam_go/DDS/ -v 6 11'"
    local_exec_command(command, working_dir, True)

def local_run_rti_record():
    working_dir = os.path.join(FLAGS.perception_directory, '../../../tools/record_replay_tools')
    working_dir = os.path.abspath(working_dir)
    command = "bash -c 'export NDDS_DOMAIN_ID=11; ./rti_record.sh /tmp/dds 11'"
    local_exec_command(command, working_dir, True)

def remote_run_xcamera():
    command = "sh -c 'export NDDS_DOMAIN_ID=11; /data/xcamera -i'"
    remote_exec_command(command)

def remote_run_perception():
    command = ". /etc/.kshrc; export NDDS_DOMAIN_ID=11; " + FLAGS.perception_binary_filepath + "  -v 3 -c /xpilot/config/perception/SOP/Unified_ET1_Guangzhou.yaml -p"
    command = "sh -c '" + command + "'"
    remote_exec_command(command)

def main(argv):
    p = threading.Thread(target=remote_run_perception)
    x = threading.Thread(target=remote_run_xcamera)
    y = threading.Thread(target=local_run_rti_replay)
    d = threading.Thread(target=local_run_rti_record)
    logging.info("Main    : before running thread")
    print('Start replay')
    time.sleep(15)
    y.start()
    print('Start xcamera')
    x.start()
    time.sleep(5)

    print('Start perception')
    p.start()
    print('Wait for xcamera 20 seconds.')
    # time.sleep(20)
    # time.sleep(5)
    print('Start record')
    d.start()

    p.join()
    x.join()
    y.join()
    d.join()



if __name__ == '__main__':
  app.run(main)
