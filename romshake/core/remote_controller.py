import os
import time
import logging
import paramiko
import subprocess

SLEEPY_TIME = 300  # Time to wait between calls (seconds)


class RemoteController():
    """Class for managing the remote system and launching jobs."""
    def __init__(self, host, user, scratch_dir, grid_search_job_file, folder,
                 grid_search_script):
        self.host = host
        self.user = user
        self.scratch_dir = scratch_dir
        self.grid_search_job_file = grid_search_job_file
        self.grid_search_script = grid_search_script
        self.folder = folder
        self.full_scratch_dir = '%s@%s:%s' % (user, host, scratch_dir)
        self.remote_wdir = '%s@%s:%s%s' % (user, host, scratch_dir, folder)

    def issue_remote_command(self, cmd):
        """Issues a command (cmd) on the remote system."""
        done = False
        while not done:
            try:
                ssh = paramiko.SSHClient()
                ssh.load_system_host_keys()
                ssh.connect(self.host, username=self.user, look_for_keys=False)
                logging.info('Executing command %s' % cmd)
                _, stdout, stderr = ssh.exec_command(cmd)
                done = True
            except:  # NOQA
                # try again later when connection is hopefully back
                logging.info('Unable to make connection so could not execute'
                             ' the command %s. Trying again in %s seconds' % (
                                 cmd, SLEEPY_TIME))
                time.sleep(SLEEPY_TIME)
        return stdout.readlines()

    def run_jobs(self, job_indices):
        """Runs jobs on the remote system."""
        job_dir = os.path.join(self.scratch_dir, self.folder, 'jobs')
        cmd = ('cd %s; for i in $(seq %s %s); do sbatch job$i; done' % (
            job_dir, min(job_indices), max(job_indices)))
        res = self.issue_remote_command(cmd)
        job_ids = [line.strip().split('job ')[1] for line in res]
        jobs_finished = False
        logging.info('Waiting for jobs to finish.')
        while not jobs_finished:
            time.sleep(SLEEPY_TIME)
            res = self.issue_remote_command('squeue -u di46bak')
            finished = [job_id not in str(res) for job_id in job_ids]
            if all(finished):
                jobs_finished = True
                logging.info('Jobs all finished.')
            else:
                logging.info(
                    'Jobs are still running. Checking again in %s seconds.' %
                    SLEEPY_TIME)


def copy_file(source, dest):
    cmd = 'scp %s %s' % (source, dest)
    while True:
        res = subprocess.call(cmd.split())
        if res == 0:
            break
        time.sleep(SLEEPY_TIME)
