import os
import subprocess
import pandas as pd

"""
Run this file on supermuc to submit jobs created by prepare_jobs.py
"""

idir = 'source_files'
os.chdir(idir)
df = pd.read_csv('sims_params.csv')
for idx, row in df.iterrows():
    os.chdir(str(idx))
    result = subprocess.check_output(['sbatch', 'sim_job'])
    job_id = result.split(' ')[-1]
    subprocess.run(['sbatch', '--dependency=afterok:%s' % job_id, 'gm_job'])
    os.chdir('..')
