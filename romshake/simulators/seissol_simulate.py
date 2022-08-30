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

# Default parameters for writing standard rupture format files
DEFAULT_SOURCE = {
    'mw': 5.44,
    'tini': 0.0,
    'sd': 1.0e6,
    'lat': 33.949,
    'lon': -117.766,
    'depth': 5.0,
    'strike': 0.0,
    'dip': 90.0,
    'rake': 0.0,
    'mu': 1.0e10,
    'fname': 'source.srf',
    'plot': False,
    'vs': 3464.0,
    'dt': 0.001,
    'tmax': 10.0,
    'k': 0.32}
PGV_FILE = 'pgv.npy'


class SeisSolSimulator():
    """Class for producing PGV simulation estimates using SeisSol."""
    def __init__(self, par_file, sim_job_file, prefix,
                 max_jobs, t_per_sim, t_max_per_job, take_log_imt, filt_freq,
                 remote=None, netcdf_files=[]):
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
        self.receiver_file = 'receivers.dat'
        self.coords = np.genfromtxt(self.receiver_file)
        self.filt_freq = filt_freq

    def load_data(self, folder, indices):
        """Loads the generated data into a single numpy array.

        Args:
            folder (str): Folder path.
            indices (list): List of indices (integers) to load data.

        Returns:
            numpy.ndarray: Numpy array of the generated data.
        """
        logging.info('Loading data.')
        all_data = []
        for sim_idx in indices:
            data = np.load(
                os.path.join(folder, 'data', str(sim_idx), PGV_FILE))
            if self.take_log_imt:
                data = np.log(data)
            all_data.append(data)
        return np.array(all_data).T

    def evaluate(self, params_dict, indices, folder, **kwargs):
        """Prepares and runs the jobs to produce PGV estimates.

        Args:
            params_dict (dict): Dictionary of the parameters and values.
            indices (list): List of indices (integers).
            folder (str): Folder path.

        Returns:
            tuple: Returns empty tuple.
        """
        self.prepare_common_files(folder)
        job_indices = self.prepare_jobs(folder, indices, params_dict)
        self.sync_files(folder, self.remote.full_scratch_dir, False)
        self.make_puml_file(folder)
        self.remote.run_jobs(job_indices)
        return (np.array([]), np.array([]))

    def plot_snapshot(
            self, ax, snap, vmin, vmax, title, cmap, **kwargs):
        """Plots a single snapshot.

        Args:
            ax (matplotlib.axes): Matplotlib axis.
            snap (numpy.ndarray): Data array.
            vmin (float): Minimum value for colorbar.
            vmax (float): Maximum value for colorbar.
            title (str): Plot title.
            cmap (str): Matplotlib colormap string.
        """
        im = ax.tricontourf(
            self.coords.T[0],
            self.coords.T[1],
            snap, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='log(PGV)')

    def make_puml_file(self, folder):
        """Creates the PUML mesh file using pumgen.
        Args:
            folder (str): Folder path.
        """
        wdir = '%s%s' % (self.remote.scratch_dir, folder)
        if self.puml_mesh_file not in str(
                self.remote.issue_remote_command('ls %s' % wdir)):
            logging.info('Running pumgen to create PUML mesh file.')
            self.remote.issue_remote_command(
                'cd %s; pumgen %s -s msh2' % (
                    wdir, self.gmsh_mesh_file))

    def prepare_jobs(self, folder, indices, params_dict):
        """Prepares the job files for submissions.

        Args:
            folder (str): Folder path.
            indices (list): List of indices (ints).
            params_dict (dict): Dictionary of parameters/values.

        Returns:
            list: List of indices.
        """
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
        params_groups = np.array_split(
            np.array(list(params_dict.values())).T, njobs)
        param_keys = params_dict.keys()

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
            for j, sim_idx in enumerate(sims_groups[i]):
                data.append('\nmkdir %s' % sim_idx)
                data.append('\ncd %s' % sim_idx)
                write_cmd = '\nwrite_srf'
                sim_source = DEFAULT_SOURCE.copy()
                for k, key in enumerate(param_keys):
                    sim_source[key] = params_groups[i][j][k]
                    write_cmd += ' -%s %s' % (key, sim_source[key])
                write_cmd += ' -mu %s' % self.get_local_shear_modulus(
                    sim_source['lon'], sim_source['lat'], sim_source['depth'])
                data.append(write_cmd)
                data.append(
                    '\nrconv -i source.srf -o source.nrf -m "+proj=utm '
                    '+zone=11 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"')
                data.append('\ncp ../../%s .' % self.par_file)
                data.append('\nmkdir output')
                data.append('\nmpiexec -n $SLURM_NTASKS %s %s' % (
                    seissol_exe, '../../%s' % self.par_file))
                with open(self.par_file, 'r') as f:
                    lines = f.readlines()
                samp_line = [s for s in lines if 'pickdt' in s][0]
                samp_rate = 1 / float(samp_line.split('=')[1].split('!')[0])
                data.append(
                    '\ncompute_pgv output ../../receivers.dat %s %s -plot' % (
                        self.filt_freq, samp_rate))
                data.append('\ncd ..')
            with open(os.path.join(
                    job_dir, 'job%s' % jobidx), 'w') as myfile:
                myfile.writelines(data)
        logging.info('Created %s job files.' % njobs)
        return range(startidx, startidx + njobs)

    def sync_files(self, source, dest, exclude_output):
        """Syncs files with the remote system.

        Args:
            source (str): Source path.
            dest (str): Destintion path.
            exclude_output (bool): If True, excludes the output directory
                from the sync.
        """
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
        """Returns a list of the indices that were run successfully.

        Args:
            folder (str): Folder path
            indices (_type_): List of indices (ints).

        Returns:
            list: List of successful indices.
        """
        return [idx for idx in indices if os.path.exists(os.path.join(
            folder, 'data', str(idx), 'pgv.npy'))]

    def prepare_common_files(self, folder):
        """Prepares the common files for all simulations, including the
        material, mesh, netCDF, job, and receiver files.

        Args:
            folder (str): Folder path.
        """
        with open(self.par_file, 'rt') as f:
            data = f.read()
            data = data.replace('material_file_name', self.material_file)
            data = data.replace('mesh_file_name', self.puml_mesh_file)
        with open(os.path.join(folder, self.par_file), 'wt') as f:
            f.write(data)
        for file in self.netcdf_files + [
                self.gmsh_mesh_file, self.material_file, self.sim_job_file,
                self.receiver_file]:
            shutil.copyfile(file, os.path.join(folder, file))

        datadir = os.path.join(folder, 'data')
        if not os.path.exists(datadir):
            os.makedirs(datadir)

    def get_local_shear_modulus(self, lon, lat, depth):
        """Returns the local shear modulus at the earthquake source location.

        Args:
            lon (float): Longitude (decimal degrees).
            lat (float): Latitude (decimal degrees).
            depth (float): Depth (km)

        Returns:
            float: Local shear modulus (in Pa).
        """        
        try:
            fname = 'rhomulambda-inner.nc'
            data = nc.Dataset(fname, 'r')
            x = data.variables['x'][:]
            y = data.variables['y'][:]
            z = data.variables['z'][:]
            mu = data.variables['data'][:]['mu']
            interp = RegularGridInterpolator((z, y, x), mu)
            sProj = ('+proj=utm +zone=11 +ellps=WGS84 +datum=WGS84'
                     ' +units=m +no_defs')
            transformer = Transformer.from_crs(
                'epsg:4326', sProj, always_xy=True)
            utmx, utmy = transformer.transform(lon, lat)
            return float(interp((-depth * 1000, utmy, utmx)))
        except FileNotFoundError:
            return 3.23980992e10
