'''
Script for preparing snapshot jobs to be submitted on supermuc.

TODO: Add functionality for starting at a later point in the halton sequence.
'''

import os
import yaml
import time
import shutil
import itertools
import subprocess
import numpy as np
import pandas as pd
from halton import get_halton


def write_standard_rupture_format(lon, lat, depth, strike, dip, rake, M,
                                  tini, slip1_cm, slip2_cm, slip3_cm, fname):
    '''
    Writes a standard rupture format file.
    '''
    dt = 0.0002
    rho = 2700.0
    vs = 3464.0
    mu = vs**2*rho
    M0 = 10**(1.5 * M + 9.1)  # Convert from moment to moment magnitude
    area = M0/mu*1e4  # m^2 to cm^2

    T = 0.1
    vtime = np.arange(0, 4, dt)
    sliprate_cm = slip1_cm * 1/T**2 * vtime*np.exp(-vtime/T)

    nt1 = vtime.shape[0]
    nt2 = 0
    nt3 = 0

    fout = open(fname, 'w')
    fout.write('1.0\n')
    fout.write('POINTS 1\n')
    fout.write("%f %f %f %f %f %.10e %f %f\n" %
               (lon, lat, depth, strike, dip, area, tini, dt))
    fout.write("%f %f %d %f %d %f %d\n" %
               (rake, slip1_cm, nt1, slip2_cm, nt2, slip3_cm, nt3))
    np.savetxt(fout, sliprate_cm, fmt='%.18e')
    fout.close()


with open('config.yml') as f:
    config = yaml.safe_load(f)


# Prepare output directory
if os.path.exists(config['name']):
    shutil.rmtree(config['name'])
os.makedirs(config['name'])
source_dir = os.path.join(config['name'], 'source_files')
os.makedirs(source_dir)

# Copy some files that we will need
# for file in ['submit_jobs.py', 'mesh_socal_topo.msh2', 'rhomulambda-']

df = pd.read_csv(os.path.join(
    config['setup_dir'], config['params_setup_file']), index_col=0)

# Create data frame that contains the parameter information for each simulation
pvals = []
if config['sample_method'] == 'even':
    for colname in df:
        col = df[colname]
        if col['log']:
            p = np.logspace(np.log10(col['min']),
                            np.log10(col['max']), int(col['n']))
        else:
            p = np.linspace(col['min'], col['max'], int(col['n']))
        pvals.append(p)
    prod = itertools.product(*pvals)
    df_out = pd.DataFrame(list(prod))
elif config['sample_method'] == 'random':
    for colname in df:
        col = df[colname]
        pvals.append(np.random.uniform(col['min'], col['max'], config['n']))
    df_out = pd.DataFrame(pvals).T
elif config['sample_method'] == 'halton':
    for i, colname in enumerate(df):
        col = df[colname]
        pvals.append(get_halton(col['min'], col['max'], i + 2, config['n']))
    df_out = pd.DataFrame(pvals).T
else:
    raise ValueError('Invalid value for "sample_method"')

df_out.columns = df.columns
df_out.index.name = 'sim_no'
df_out.to_csv(os.path.join(config['name'],
              'sims_params.csv'), float_format='%.3f')

# For each simulation, generate a new source file (.srf) and create .nrf using
# rconv
for idx, row in df_out.iterrows():

    print('%s/%s' % (idx, df_out.shape[0]))

    sim_dir = os.path.join(source_dir, str(idx))
    sim_source_dir = os.path.join(sim_dir, 'output')

    for dir in [sim_dir, sim_source_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for param in row.index:
        config['source_params'][param] = row[param]

    srf_fname = os.path.join(sim_dir, 'source.srf')
    nrf_fname = srf_fname.replace('srf', 'nrf')

    write_standard_rupture_format(**config['source_params'], fname=srf_fname)

    # Assume SRF data are already projected in the correct coord system
    # This means lat/lons specified in parameter config need to use
    # the mesh coordinate system
    subprocess.call(
        ['/bin/zsh', '-i', '-c',
         ('rconv -i %s -m "+proj=lonlat +datum=WGS84 +units=m" -o %s') % (
             srf_fname, nrf_fname)])

    # Copy the parameter file to each simulation directory
    shutil.copyfile(os.path.join(config['setup_dir'], config['par_file']),
                    os.path.join(sim_dir, config['par_file']))

# Create job files
job_dir = os.path.join(source_dir, 'jobs')
os.makedirs(job_dir)

# Calculate number of jobs we should submit
nsims = df_out.shape[0]
njobs = max(5, df_out.shape[0] * config['t_sim'] / config['t_max'])
njobs = min(nsims, njobs)  # In case we have a very small no. of simulations
sims_groups = np.array_split(df_out.index.tolist(), njobs)

# Create and populate the job files
with open(os.path.join(config['setup_dir'], config['sim_job_file']), 'r') as myfile:
    data_temp = myfile.readlines()

for i in range(njobs):
    data = data_temp.copy()
    job_run_time = config['t_sim'] * len(sims_groups[i])
    data = [sub.replace(
        '00:30:00',
        time.strftime('%H:%M:%S', time.gmtime(job_run_time*3600)))
        for sub in data]

    for sim_idx in sims_groups[i]:
        data.append('\ncd ../%s' % sim_idx)
        data.append(
            ('\nmpiexec -n $SLURM_NTASKS SeisSol_Release_sskx_5_elastic '
             'parameters.par'))
        data.append((
            '\nmpiexec -n $SLURM_NTASKS python -u /dss/dsshome1/0B/di46bak/'
            'SeisSol/postprocessing/science/GroundMotionParametersMaps/'
            'ComputeGroundMotionParametersFromSurfaceOutput_Hybrid.py '
            'output/socal-surface.xdmf'))
    with open(os.path.join(job_dir, 'job%s' % i), 'w') as myfile:
        myfile.writelines(data)

if config['move_to_supermuc']:
    print('Copying files to supermuc...')
    subprocess.call(
        ['/bin/zsh', '-i', '-c', 'rsync -a {0} $SMS/{0} --delete '
         '--progress'.format(config['name'])])

# TODO:
# Other things to copy...
# 1 job submission script
# 2 mesh file (PUMGEN?)
# 3 material file (and two .nc files)
# 4 records points.dat

# other TODO:
# check for valid topography values
# check velocity (material) values

# check file names, parameter file and mesh file
# check loh1 vs socal naming for ground motion maps
