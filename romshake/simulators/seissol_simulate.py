import os
import time
import shutil
import inspect
import logging
import subprocess
import numpy as np
import netCDF4 as nc
from pyproj import Transformer
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from romshake.core.remote_controller import SLEEPY_TIME

seissol_exe = 'SeisSol_Release_dskx_4_elastic'
logging.getLogger('paramiko').setLevel(logging.WARNING)
source_params = {
    'M': 5.44,
    'tini': 0.0,
    'slip1_cm': 15.0,
    'slip2_cm': 0.0,
    'slip3_cm': 0.0,
    'lat': 33.949,
    'lon': -117.766,
    'depth': 5.0,
    'strike': 0.0,
    'dip': 90.0,
    'rake': 0.0}


class SeisSolSimulator():
    def __init__(self, par_file, sim_job_file, prefix,
                 max_jobs, t_per_sim, t_max_per_job, take_log_imt,
                 mesh_coords, remote=None, netcdf_files=[]):
        self.par_file = par_file
        self.sim_job_file = sim_job_file
        self.prefix = prefix
        self.max_jobs = max_jobs
        self.t_per_sim = t_per_sim
        self.t_max_per_job = t_max_per_job
        self.take_log_imt = take_log_imt
        self.remote = remote
        self.netcdf_files = netcdf_files
        self.gmsh_mesh_file = '%s.msh2' % self.prefix
        self.puml_mesh_file = '%s.puml.h5' % self.prefix
        self.material_file = 'material.yaml'
        self.mesh_coords = mesh_coords

    def load_data(self, folder, indices):
        logging.info('Loading data.')
        all_data = []
        for sim_idx in indices:
            data = np.array(
                os.path.join(folder, 'data', str(sim_idx), 'pgv.npy'))
            if self.take_log_imt:
                data = np.log(data)
            all_data.append(data)
        return np.array(all_data).T

    def evaluate(self, params_dict, indices, folder, **kwargs):
        self.prepare_common_files(folder)
        for i, sim_idx in enumerate(indices):
            for param_label, param_vals in params_dict.items():
                source_params[param_label] = param_vals[i]
            self.write_source_files(folder, source_params, sim_idx)
        job_indices = self.prepare_jobs(folder, indices)
        self.sync_files(folder, self.remote.full_scratch_dir, False)
        self.make_puml_file(folder)
        self.remote.run_jobs(job_indices)
        return (np.array([]), np.array([]))

    def plot_snapshot(
            self, ax, snap, vmin, vmax, title, cmap, coords, **kwargs):
        im = ax.tricontourf(
            coords.T[0],
            coords.T[1],
            snap, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='log(PGV)')

    def make_puml_file(self, folder):
        wdir = '%s%s' % (self.remote.scratch_dir, folder)
        if self.puml_mesh_file not in str(
                self.remote.issue_remote_command('ls %s' % wdir)):
            logging.info('Running pumgen to create PUML mesh file.')
            self.remote.issue_remote_command(
                'cd %s; pumgen %s -s msh2' % (
                    wdir, self.gmsh_mesh_file))

    def write_source_files(self, folder, source_params, sim_idx):
        sim_dir = os.path.join(folder, 'data', str(sim_idx))
        odir = os.path.join(sim_dir, 'output')
        for dir in [sim_dir, odir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        srf_fname = os.path.join(sim_dir, 'source.srf')
        self.write_standard_rupture_format(**source_params, fname=srf_fname)
        shutil.copyfile(
            os.path.join(folder, self.par_file),
            os.path.join(sim_dir, self.par_file))

    def prepare_jobs(self, folder, indices):
        logging.info('Preparing job files.')
        job_dir = os.path.join(folder, 'jobs')
        if os.path.exists(job_dir):
            jobfiles = [file for file in os.listdir(job_dir) if 'job' in file]
            startidx = max([int(file[-1]) for file in jobfiles]) + 1
        else:
            os.makedirs(job_dir)
            startidx = 0

        # Calculate number of jobs we should submit
        nsims = len(indices)
        njobs = max(self.max_jobs, nsims * self.t_per_sim / self.t_max_per_job)
        njobs = min(nsims, njobs)
        sims_groups = np.array_split(indices, njobs)

        # Create and populate the job files
        with open(os.path.join(folder, self.sim_job_file), 'r') as myfile:
            data_temp = myfile.readlines()

        for i in range(njobs):
            jobidx = str(startidx + i)
            data = data_temp.copy()
            job_run_time = self.t_per_sim * len(sims_groups[i])
            data = [sub.replace(
                '00:30:00',
                time.strftime('%H:%M:%S', time.gmtime(job_run_time*3600)))
                for sub in data]
            data = [sub.replace('jobidx', jobidx) for sub in data]
            for sim_idx in sims_groups[i]:
                data.append('\ncd %s' % sim_idx)
                data.append(
                    '\nrconv -i source.srf -o source.nrf -m "+proj=utm '
                    '+zone=11 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"')
                data.append('\nmpiexec -n $SLURM_NTASKS %s %s' % (
                    seissol_exe, self.par_file))
                data.append('\npython /hppfs/scratch/0B/di46bak/scripts/compute_pgv.py')
                data.append('\ncd ..')
            with open(os.path.join(
                    job_dir, 'job%s' % jobidx), 'w') as myfile:
                myfile.writelines(data)
        logging.info('Created %s job files.' % njobs)
        return range(startidx, startidx + njobs)

    def sync_files(self, source, dest, exclude_output):
        exclude_file = os.path.join(
            os.path.split(inspect.getfile(self.__class__))[0], 'exclude.txt')
        logging.info('Syncing files. Source: %s, Destination: %s' % (
            source, dest))
        cmd = ("rsync -a %s %s --progress "
               "--exclude-from=%s" % (source, dest, exclude_file))
        if exclude_output:
            cmd += ' --exclude output/'
        while True:
            res = subprocess.call(cmd.split())
            if res == 0:
                break
            time.sleep(SLEEPY_TIME)

    def get_successful_indices(self, folder, indices):
        return [idx for idx in indices if os.path.exists(os.path.join(
            folder, 'data', str(idx), 'pgv.npy'))]

    def prepare_common_files(self, folder):
        with open(self.par_file, 'rt') as f:
            data = f.read()
            data = data.replace('material_file_name', self.material_file)
            data = data.replace('mesh_file_name', self.puml_mesh_file)
        with open(os.path.join(folder, self.par_file), 'wt') as f:
            f.write(data)
        for file in self.netcdf_files + [
                self.gmsh_mesh_file, self.material_file, self.sim_job_file]:
            shutil.copyfile(file, os.path.join(folder, file))

    def write_standard_rupture_format(
            self, lon, lat, depth, strike, dip, rake, M, tini, slip1_cm,
            slip2_cm, slip3_cm, fname):

        # Area calculation
        total_slip_cm = np.sqrt(slip1_cm**2 + slip2_cm**2 + slip3_cm**2)
        total_slip_m = total_slip_cm / 100
        M0 = 10**(1.5 * M + 9.1)
        mu = self.get_local_shear_modulus(lon, lat, depth)
        area_m2 = M0 / (mu * total_slip_m)
        area_cm2 = area_m2 * 10000

        T = 0.1
        dt = 0.0002
        vtime = np.arange(0, 4, dt)
        sliprate_cm = slip1_cm * 1/T**2 * vtime*np.exp(-vtime/T)

        nt1 = vtime.shape[0]
        nt2 = 0
        nt3 = 0

        fout = open(fname, 'w')
        fout.write('1.0\n')
        fout.write('POINTS 1\n')
        fout.write("%.5e %.5e %f %f %f %.10e %f %f\n" %
                   (lon, lat, depth, strike, dip, area_cm2, tini, dt))
        fout.write("%f %f %d %f %d %f %d\n" %
                   (rake, slip1_cm, nt1, slip2_cm, nt2, slip3_cm, nt3))
        np.savetxt(fout, sliprate_cm, fmt='%.18e')
        fout.close()

    def get_local_shear_modulus(self, lat, lon, depth):
        try:
            fname = 'rhomulambda-inner.nc'
            data = nc.Dataset(fname, 'r')
            x = data.variables['x'][:]
            y = data.variables['y'][:]
            z = data.variables['z'][:]
            mu = data.variables['data'][:]['mu']
            interp = RegularGridInterpolator((z, y, x), mu)
            sProj = ('+proj=utm +zone=11 +ellps=WGS84 +datum=WGS84'
                     '+units=m +no_defs')
            transformer = Transformer.from_crs(
                'epsg:4326', sProj, always_xy=True)
            utmx, utmy = transformer.transform(lon, lat)
            return float(interp((-depth, utmy, utmx)))
        except FileNotFoundError:
            if depth <= 10.0e3:
                return 1.04e10
            else:
                return 3.23980992e10
