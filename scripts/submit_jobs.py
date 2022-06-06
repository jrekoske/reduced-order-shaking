'''
Script for submitting jobs
'''

import os
import glob
import subprocess

idir = 'source_files'
os.chdir(os.path.join(idir, 'jobs'))
for job in glob.glob('job*'):
    subprocess.run(['sbatch', job])
