import inspect
import os
import time
import h5py
import shutil
import logging
import subprocess
import numpy as np

from romshake.simulators.reorder_elements import run_reordering

imt = 'PGV'
mask_file = 'mask.npy'
h5_gm_cor_file = 'loh1-GME_corrected.h5'
remote_dir = 'di46bak@skx.supermuc.lrz.de:/hppfs/scratch/0B/di46bak/'
sleepy_time = 1 * 60  # (5 minutes)


class SeisSolSimulator():
    def __init__(self, par_file, sim_job_file, mesh_file, material_file,
                 max_jobs, t_per_sim, t_max_per_job, take_log_imt,
                 mask_bounds, netcdf_files=[]):
        self.par_file = par_file
        self.sim_job_file = sim_job_file
        self.mesh_file = mesh_file
        self.material_file = material_file
        self.max_jobs = max_jobs
        self.t_per_sim = t_per_sim
        self.t_max_per_job = t_max_per_job
        self.take_log_imt = take_log_imt
        self.mask_bounds = mask_bounds
        self.netcdf_files = netcdf_files

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
        for i, element in enumerate(connect):
            cx, cy, _ = geom[element].mean(axis=0)
            if (cx >= self.mask_bounds[0] and cx <= self.mask_bounds[1]
                    and cy >= self.mask_bounds[2]
                    and cy <= self.mask_bounds[3]):
                elem_mask[i] = 1.0
        np.save(mask_path, elem_mask)
        return elem_mask

    def evaluate(self, params_dict, indices, folder, **kwargs):
        self.prepare_common_files(folder)
        source_params = {
            'M': 6.0,
            'lon': -117.9437,
            'lat': 34.1122,
            'tini': 0.0,
            'slip1_cm': 100.0,
            'slip2_cm': 0.0,
            'slip3_cm': 0.0}
        for i, sim_idx in enumerate(indices):
            for param_label, param_vals in params_dict.items():
                source_params[param_label] = param_vals[i]
            print(source_params)
            self.write_source_files(folder, source_params, sim_idx)
        self.prepare_jobs(folder, indices)
        self.sync_files(folder, remote_dir, False)
        self.launch_jobs(folder)
        self.sync_files('%s/%s/' % (remote_dir, folder), folder, True)
        successful_indices = self.get_successful_indices(folder, indices)
        self.reorder_elements(folder, successful_indices)

        params_arr = np.array(list(params_dict.values())).T
        good_params = np.array(
            [param for param, idx in zip(params_arr, indices)
             if idx in successful_indices])

        return good_params, self.load_data(folder, successful_indices)

    def write_source_files(self, folder, source_params, sim_idx):
        logging.info('Writing source files for simulation index %s' % sim_idx)
        sim_dir = os.path.join(folder, 'data', str(sim_idx))
        odir = os.path.join(sim_dir, 'output')
        for dir in [sim_dir, odir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        srf_fname = os.path.join(sim_dir, 'source.srf')
        nrf_fname = srf_fname.replace('srf', 'nrf')
        self.write_standard_rupture_format(**source_params, fname=srf_fname)

        subprocess.call(
            ['/bin/zsh', '-i', '-c', (
                'rconv -i %s -m "+proj=utm +zone=11 +ellps=WGS84 +datum=WGS84'
                ' +units=m +no_defs" -o %s') % (srf_fname, nrf_fname)])

        shutil.copyfile(
            os.path.join(folder, self.par_file),
            os.path.join(sim_dir, self.par_file))

    def prepare_jobs(self, folder, indices):
        logging.info('Preparing job files.')
        job_dir = os.path.join(folder, 'jobs')
        # Start with a clean job directory
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        os.makedirs(job_dir)

        # Calculate number of jobs we should submit
        nsims = len(indices)
        njobs = max(self.max_jobs, nsims * self.t_per_sim / self.t_max_per_job)
        njobs = min(nsims, njobs)
        sims_groups = np.array_split(indices, njobs)

        # Create and populate the job files
        with open(os.path.join(folder, self.sim_job_file), 'r') as myfile:
            data_temp = myfile.readlines()

        for i in range(njobs):
            data = data_temp.copy()
            job_run_time = self.t_per_sim * len(sims_groups[i])
            data = [sub.replace(
                '00:30:00',
                time.strftime('%H:%M:%S', time.gmtime(job_run_time*3600)))
                for sub in data]

            for sim_idx in sims_groups[i]:
                data.append('\ncd %s' % sim_idx)
                data.append(
                    ('\nmpiexec -n $SLURM_NTASKS '
                     'SeisSol_Release_dskx_5_elastic %s' % self.par_file))
                data.append((
                    '\nmpiexec -n $SLURM_NTASKS python -u '
                    '/dss/dsshome1/0B/di46bak/'
                    'SeisSol/postprocessing/science/'
                    'GroundMotionParametersMaps/'
                    'ComputeGroundMotionParametersFromSurfaceOutput_Hybrid.py '
                    'output/loh1-surface.xdmf'))
                data.append('\ncd ..')
            with open(os.path.join(job_dir, 'job%s' % i), 'w') as myfile:
                myfile.writelines(data)
        logging.info('Created %s job files.' % njobs)

    def sync_files(self, source, dest, exclude_output):
        exclude_file = os.path.join(
            os.path.split(inspect.getfile(self.__class__))[0], 'exclude.txt')
        logging.info('Syncing files. Source: %s, Destination: %s' % (
            source, dest))
        cmd = ("rsync -a %s %s --delete --progress "
               "--exclude-from=%s" % (source, dest, exclude_file))
        if exclude_output:
            cmd += ' --exclude output/'
        print('command:', cmd)
        while True:
            res = subprocess.call(cmd.split())
            if res == 0:
                break
            time.sleep(sleepy_time)

    def get_successful_indices(self, folder, indices):
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

    def reorder_elements(self, folder, indices):
        logging.info('Correcting element ordering.')
        idir = os.path.join(folder, 'data')

        ref_idx = self.get_ref_idx(folder)
        ref_file = os.path.join(idir, str(ref_idx), 'loh1-GME.xdmf')

        for idx in indices:
            idx = str(idx)
            file = os.path.join(idir, idx, 'loh1-GME.xdmf')
            run_reordering(ref_file, file, [-1], ['all'])
            fname = 'loh1-GME_corrected'
            h5name = '%s.h5' % fname
            xdmfname = '%s.xdmf' % fname
            h5new = os.path.join(idir, idx, h5name)
            xdmfnew = os.path.join(idir, idx, xdmfname)
            os.rename(h5name, h5new)
            os.rename(xdmfname, xdmfnew)

    def launch_jobs(self, folder):
        logging.info('Launching jobs.')
        cmd = ('cd $SCRATCH/%s/jobs; for fname in job*; '
               'do sbatch $fname; done' % folder)
        res = self.issue_remote_command(
            self.build_remote_command(cmd)).splitlines()
        job_ids = [line.split('job ')[1] for line in res]

        # Wait for the jobs to finish, check status using squeue
        jobs_finished = False
        logging.info('Waiting for jobs to finish.')
        while not jobs_finished:
            time.sleep(sleepy_time)
            res = self.issue_remote_command(
                self.build_remote_command('squeue -u di46bak'))
            finished = [job_id not in res for job_id in job_ids]
            if all(finished):
                jobs_finished = True
                logging.info('Jobs all finished.')

    def prepare_common_files(self, folder):
        with open(self.par_file, 'rt') as f:
            data = f.read()
            data = data.replace('material_file_name', self.material_file)
            data = data.replace('mesh_file_name', self.mesh_file)
        with open(os.path.join(folder, self.par_file), 'wt') as f:
            f.write(data)
        for file in self.netcdf_files + [
                self.mesh_file, self.material_file, self.sim_job_file]:
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
        fout.write("%.5e %.5e %f %f %f %.10e %f %f\n" %
                   (lon, lat, depth, strike, dip, area, tini, dt))
        fout.write("%f %f %d %f %d %f %d\n" %
                   (rake, slip1_cm, nt1, slip2_cm, nt2, slip3_cm, nt3))
        np.savetxt(fout, sliprate_cm, fmt='%.18e')
        fout.close()

    def issue_remote_command(self, cmd):
        done = False
        while not done:
            try:
                res = subprocess.check_output(cmd).decode('utf-8')
                done = True
            except subprocess.CalledProcessError:
                # try command again in a little bit when connection
                # is hopefully back
                print('Unable to make connection... trying again later.')
                time.sleep(sleepy_time)
        return res

    def build_remote_command(self, cmd):
        host = 'skx.supermuc.lrz.de'
        return ['ssh', '%s' % host, cmd]
