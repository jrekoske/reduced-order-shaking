import os
import time
import h5py
import shutil
import logging
import subprocess
import numpy as np
import pandas as pd

imt = 'PGV'
mask_file = 'mask.npy'
par_csv_file = 'parameters.csv'
par_file = 'seissol_parfile.par'
sim_job_file = 'sim_job'
mesh_file = 'loh1.puml.h5'
material_file = 'loh1.yaml'
h5_gm_cor_file = 'loh1-GME_corrected.h5'

# Maximum number of jobs to submit at a time
max_jobs = 10

take_log = True

t_sim = 0.5  # Time (hours) per simulation
t_max = 48   # Max wall time (hours) of a job on supermuc

xmin, xmax, ymin, ymax = -30e3, 30e3, -30e3, 30e3

remote_dir = 'di46bak@skx.supermuc.lrz.de:/hppfs/scratch/0B/di46bak/'


def load_data(folder, indices):
    logging.info('Loading data.')
    df = pd.read_csv(os.path.join(folder, par_csv_file), index_col=0)
    df = df.iloc[indices]
    params = df.values
    all_data = []
    mask_path = os.path.join(folder, mask_file)
    if os.path.exists(mask_path):
        elem_mask = np.load(mask_path)
    else:
        elem_mask = make_mask(folder, mask_path)
    for sim_idx in indices:
        h5f = h5py.File(os.path.join(
            folder, 'data', str(sim_idx), h5_gm_cor_file), 'r')
        data = np.array(h5f[imt]).flatten()[elem_mask]
        if take_log:
            data = np.log(data)
        all_data.append(data)
    return params, np.array(all_data).T


def make_mask(folder, mask_path):
    logging.info('Creating mask.')
    h5f = h5py.File(
        os.path.join(folder, 'data', '0', h5_gm_cor_file), 'r')

    connect = h5f['connect']
    geom = h5f['geometry']
    imts = list(h5f.keys())
    imts.remove('connect')
    imts.remove('geometry')
    elem_mask = np.zeros(connect.shape[0], dtype=bool)
    for i, element in enumerate(connect):
        cx, cy, _ = geom[element].mean(axis=0)
        if (cx >= xmin and cx <= xmax and cy >= ymin and cy <= ymax):
            elem_mask[i] = 1.0
    np.save(mask_path, elem_mask)
    return elem_mask


def evaluate(rb, params, indices, folder):
    prepare_common_files(folder)
    source_params = {
        'M': 6.0,
        'lon': 0.0,
        'lat': 0.0,
        'tini': 0.0,
        'slip1_cm': 100.0,
        'slip2_cm': 0.0,
        'slip3_cm': 0.0}
    for param, sim_idx in zip(params, indices):
        sim_params = {
            param_key: param[i]
            for i, param_key in enumerate(rb.bounds.keys())}
        source_params = {**source_params, **sim_params}
        write_source_files(folder, source_params, sim_idx)
    prepare_jobs(folder, indices)
    sync_files(folder, remote_dir, False)
    launch_jobs(folder)
    sync_files('%s/%s/' % (remote_dir, folder), folder, True)
    successful_indices = get_successful_indices(folder, indices)
    reorder_elements(folder, successful_indices)
    return load_data(folder, successful_indices)


def write_source_files(folder, source_params, sim_idx):
    logging.info('Writing source files for simulation index %s' % sim_idx)
    sim_dir = os.path.join(folder, 'data', str(sim_idx))
    odir = os.path.join(sim_dir, 'output')
    for dir in [sim_dir, odir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    srf_fname = os.path.join(sim_dir, 'source.srf')
    nrf_fname = srf_fname.replace('srf', 'nrf')
    write_standard_rupture_format(**source_params, fname=srf_fname)

    subprocess.call(
        ['/bin/zsh', '-i', '-c', (
            'rconv -i %s -m "+proj=lonlat +datum=WGS84 +units=m'
            ' +axis=ned" -o %s') % (srf_fname, nrf_fname)])

    shutil.copyfile(
        os.path.join(folder, par_file),
        os.path.join(sim_dir, par_file))


def prepare_jobs(folder, indices):
    logging.info('Preparing job files.')
    job_dir = os.path.join(folder, 'jobs')
    # Start with a clean job directory
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    os.makedirs(job_dir)

    # Calculate number of jobs we should submit
    nsims = len(indices)
    njobs = max(max_jobs, nsims * t_sim / t_max)
    njobs = min(nsims, njobs)
    sims_groups = np.array_split(indices, njobs)

    # Create and populate the job files
    with open(os.path.join(folder, sim_job_file), 'r') as myfile:
        data_temp = myfile.readlines()

    for i in range(njobs):
        data = data_temp.copy()
        job_run_time = t_sim * len(sims_groups[i])
        data = [sub.replace(
            '00:30:00',
            time.strftime('%H:%M:%S', time.gmtime(job_run_time*3600)))
            for sub in data]

        for sim_idx in sims_groups[i]:
            data.append('\ncd %s' % sim_idx)
            data.append(
                ('\nmpiexec -n $SLURM_NTASKS SeisSol_Release_sskx_4_elastic '
                 '%s' % par_file))
            data.append((
                '\nmpiexec -n $SLURM_NTASKS python -u '
                '/dss/dsshome1/0B/di46bak/'
                'SeisSol/postprocessing/science/GroundMotionParametersMaps/'
                'ComputeGroundMotionParametersFromSurfaceOutput_Hybrid.py '
                'output/loh1-surface.xdmf'))
            data.append('\ncd ..')
        with open(os.path.join(job_dir, 'job%s' % i), 'w') as myfile:
            myfile.writelines(data)
    logging.info('Created %s job files.' % njobs)


def sync_files(source, dest, exclude_output):
    logging.info('Syncing files. Source: %s, Destination: %s' % (
        source, dest))
    cmd = 'rsync -a %s %s --delete --progress' % (source, dest)
    if exclude_output:
        cmd += ' --exclude output/'
    print(cmd)
    while True:
        res = subprocess.call(cmd.split())
        if res == 0:
            break
        time.sleep(60)


def get_successful_indices(folder, indices):
    files = [
        'loh1-GME-surface_cell.h5',
        'loh1-GME-surface_vertex.h5',
        'loh1-GME.xdmf'
    ]
    good_indices = []
    for idx in indices:
        success = True
        for file in files:
            if not os.path.exists(
                    os.path.join(folder, 'data', str(idx), file)):
                success = False
        if success:
            good_indices.append(idx)
    return good_indices


def reorder_elements(folder, indices):
    logging.info('Correcting element ordering.')
    idir = os.path.join(folder, 'data')
    ref_file = os.path.join(idir, '0', 'loh1-GME.xdmf')  # reference file
    for idx in indices:
        idx = str(idx)
        file = os.path.join(idir, idx, 'loh1-GME.xdmf')
        p1 = subprocess.Popen(
            ['python', 'reorder_elements.py', ref_file, file,
             '--idt', '-1', '--Data', 'all'])
        fname = 'loh1-GME_corrected'
        h5name = '%s.h5' % fname
        xdmfname = '%s.xdmf' % fname

        h5new = os.path.join(idir, idx, h5name)
        xdmfnew = os.path.join(idir, idx, xdmfname)
        p1.wait()
        os.rename(h5name, h5new)
        os.rename(xdmfname, xdmfnew)


def launch_jobs(folder):
    logging.info('Launching jobs.')
    cmd = ('cd $SCRATCH/%s/jobs; for fname in job*; '
           'do sbatch $fname; done' % folder)
    res = issue_remote_command(
        build_remote_command(cmd)).splitlines()
    job_ids = [line.split('job ')[1] for line in res]

    # Wait for the jobs to finish, check status using squeue
    jobs_finished = False
    logging.info('Waiting for jobs to finish.')
    while not jobs_finished:
        time.sleep(60)
        res = issue_remote_command(
            build_remote_command('squeue -u di46bak'))
        finished = [job_id not in res for job_id in job_ids]
        if all(finished):
            jobs_finished = True
            logging.info('Jobs all finished.')


def prepare_common_files(folder):
    with open(par_file, 'rt') as f:
        data = f.read()
        data = data.replace('material_file_name', material_file)
        data = data.replace('mesh_file_name', mesh_file)
    with open(os.path.join(folder, par_file), 'wt') as f:
        f.write(data)
    for file in [mesh_file, material_file, sim_job_file]:
        shutil.copyfile(file, os.path.join(folder, file))


def write_standard_rupture_format(lon, lat, depth, strike, dip, rake, M,
                                  tini, slip1_cm, slip2_cm, slip3_cm, fname):
    dt = 0.0002
    rho = 2700.0
    vs = 3464.0
    mu = vs**2*rho
    M0 = 10**(1.5 * M + 9.1)
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


def issue_remote_command(cmd):
    done = False
    while not done:
        try:
            res = subprocess.check_output(cmd).decode('utf-8')
            done = True
        except subprocess.CalledProcessError:
            # try command again in a little bit when connection
            # is hopefully back
            print('Unable to make connection... trying again later.')
            time.sleep(60)
    return res


def build_remote_command(cmd):
    host = 'skx.supermuc.lrz.de'
    return ['ssh', '%s' % host, cmd]
