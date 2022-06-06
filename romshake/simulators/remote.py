import os
import subprocess
import time
import logging
import paramiko

SLEEPY_TIME = 1 * 60  # Time to wait between calls
HOST = 'skx.supermuc.lrz.de'
USER = 'di46bak'
SCRATCH_DIR = '/hppfs/scratch/0B/di46bak/'
REMOTE_DIR = '%s@%s:%s' % (USER, HOST, SCRATCH_DIR)


def issue_remote_command(cmd):
    done = False
    while not done:
        try:
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            ssh.connect(HOST, username=USER, look_for_keys=False)
            res = ssh.exec_command(cmd)[1].readlines()
            done = True
        except TimeoutError:
            # try again later when connection is hopefully back
            logging.info('Unable to make connection so could not execute'
                         ' the command %s. Trying again in %s seconds' % (
                                cmd, SLEEPY_TIME))
            time.sleep(SLEEPY_TIME)
    return res


def copy_file(source, dest):
    cmd = 'scp %s %s' % (source, dest)
    while True:
        res = subprocess.call(cmd.split())
        if res == 0:
            break
        time.sleep(SLEEPY_TIME)


def run_jobs(job_indices, folder):
    job_dir = os.path.join(REMOTE_DIR, folder, 'jobs')
    cmd = ('cd %s; for i in $(seq %s %s); do sbatch job$i; done' % (
        job_dir, min(job_indices), max(job_indices)))
    res = issue_remote_command(cmd)
    job_ids = [line.strip().split('job ')[1] for line in res]
    jobs_finished = False
    logging.info('Waiting for jobs to finish.')
    while not jobs_finished:
        time.sleep(SLEEPY_TIME)
        res = issue_remote_command('squeue -u di46bak')
        finished = [job_id not in str(res) for job_id in job_ids]
        if all(finished):
            jobs_finished = True
            logging.info('Jobs all finished.')
