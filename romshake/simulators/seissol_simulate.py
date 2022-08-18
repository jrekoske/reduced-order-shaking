import os
import time
import h5py
import shutil
import inspect
import logging
import subprocess
import numpy as np
import netCDF4 as nc
from pyproj import Transformer
from matplotlib import pyplot as plt
from seissolxdmf import seissolxdmf as sx
from scipy.interpolate import RegularGridInterpolator
from romshake.core.remote_controller import SLEEPY_TIME
from romshake.simulators.reorder_elements import run_reordering

imt = 'PGV'
mask_file = 'mask.npy'
h5_gm_cor_file = 'loh1-GME_corrected.h5'
xdmf_cor_file = 'loh1-GME_corrected.xdmf'
seissol_exe = 'SeisSol_Release_dskx_4_elastic'
gm_exe = ('/dss/dsshome1/0B/di46bak/SeisSol/postprocessing/science/'
          'GroundMotionParametersMaps/ComputeGroundMotionParameters'
          'FromSurfaceOutput_Hybrid.py')

# Turn off excessive logging from Paramiko and matplotlib
logging.getLogger('paramiko').setLevel(logging.WARNING)

# Default earthquake source parameters to use for simulations
source_params = {
    'M': 4.54,
    'tini': 0.0,
    'slip1_cm': 15.0,
    'slip2_cm': 0.0,
    'slip3_cm': 0.0,
    'lat': 34.038,
    'lon': -118.080,
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
        mask_path = os.path.join(folder, mask_file)
        if os.path.exists(mask_path):
            elem_mask = np.load(mask_path)
        else:
            elem_mask = self.make_mask(folder, mask_path)
        for sim_idx in indices:
            h5f = h5py.File(os.path.join(
                folder, 'data', str(sim_idx), h5_gm_cor_file), 'r')
            data = np.array(h5f[imt]).flatten()[elem_mask]
            if self.take_log_imt:
                data = np.log(data)
            all_data.append(data)
        return np.array(all_data).T

    def load_coords(self, folder, idx):
        mask_path = os.path.join(folder, mask_file)
        if os.path.exists(mask_path):
            elem_mask = np.load(mask_path)
        else:
            elem_mask = self.make_mask(folder, mask_path)
        ds = sx(os.path.join(folder, 'data', str(idx), xdmf_cor_file))
        geo = ds.ReadGeometry()
        connect = ds.ReadConnect()
        coords = geo[connect].mean(axis=1)[elem_mask]
        return coords

    def make_mask(self, folder, mask_path):
        logging.info('Creating mask.')
        ref_idx = self.get_ref_idx(folder)
        h5f = h5py.File(
            os.path.join(folder, 'data', str(ref_idx), h5_gm_cor_file), 'r')
        connect = h5f['connect']
        geom = h5f['geometry']
        imts = list(h5f.keys())
        imts.remove('connect')
        imts.remove('geometry')
        elem_mask = np.zeros(connect.shape[0], dtype=bool)
        crds = self.mesh_coords
        xcenter = crds['CX']
        ycenter = crds['CY']
        aoi_l = crds['AOI_L']
        xmin = xcenter - aoi_l / 2
        xmax = xcenter + aoi_l / 2
        ymin = ycenter - aoi_l / 2
        ymax = ycenter + aoi_l / 2
        logging.info('Coordinates of the area of interest: %s %s %s %s ' % (
            xmin, xmax, ymin, ymax))
        for i, element in enumerate(connect):
            cx, cy, _ = geom[element].mean(axis=0)
            if (cx >= xmin and cx <= xmax and cy >= ymin and cy <= ymax):
                elem_mask[i] = 1.0
        np.save(mask_path, elem_mask)
        return elem_mask

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
        # Return empty arrays because we don't need the data locally
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
                data.append(
                    ('\nsrun python -u %s --periods 1 --MP 48 --lowpass 1.0'
                     ' output/loh1-surface.xdmf' % gm_exe))
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
        files = [
            'loh1_lp1.0-GME-surface_cell.h5',
            'loh1_lp1.0-GME-surface_vertex.h5',
            'loh1_lp1.0-GME.xdmf']
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

    def reorder_elements(self, folder, indices):
        logging.info('Correcting element ordering.')
        idir = os.path.join(folder, 'data')

        ref_idx = self.get_ref_idx(folder)
        ref_file = os.path.join(idir, str(ref_idx), 'loh1_lp1.0-GME.xdmf')

        for idx in indices:
            idx = str(idx)
            file = os.path.join(idir, idx, 'loh1_lp1.0-GME.xdmf')
            run_reordering(ref_file, file, [-1], ['all'])
            fname = 'loh1-GME_corrected'
            h5name = '%s.h5' % fname
            xdmfname = '%s.xdmf' % fname
            h5new = os.path.join(idir, idx, h5name)
            xdmfnew = os.path.join(idir, idx, xdmfname)
            os.rename(h5name, h5new)
            os.rename(xdmfname, xdmfnew)

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

    def get_ref_idx(self, folder):
        idir = os.path.join(folder, 'data')
        all_indices = sorted([
            int(idx) for idx in os.listdir(idir) if not idx.startswith('.')])
        ref_idx = self.get_successful_indices(folder, all_indices)[0]
        logging.info('Using index %s as the reference.' % ref_idx)
        return ref_idx

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
