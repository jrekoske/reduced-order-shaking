import os
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.stats import qmc

from romshake.sample import voronoi
from romshake.core.reduced_order_model import ReducedOrderModel

FNAME = 'rom_builder.pkl'


class NumericalRomBuilder():
    def __init__(self, folder, simulator, n_seeds_initial,
                 n_seeds_refine,
                 n_seeds_stop, samp_method,
                 bounds,
                 ranks=[], update_basis=False,
                 ml_regressors={}, rbf_kernels=[],
                 k_val=5, vor_kval_refine=None, vor_interp_refine=None):
        """Class for building reduced-order models from numerical simulations.

        Args:
            folder (str): Path associated with ROM data.
            simulator (object): Simulator for generating
                new data from parameters. Can be either analytic or launch
                numerical simulation jobs.
            n_seeds_initial (int): Number of seeds for the first iteration.
            n_seeds_refine (int): Number of seeds to generate with each
                iteration.
            n_seeds_stop (int): Maximum number of seeds.
            samp_method (str): Sampling refinement strategy.
            bounds (dict): Min/max values for the parameter space.
            rank (int, optional): Rank of the basis. Defaults to None.
            update_basis (bool, optional): Whether to update the basis with
                each iteration. Defaults to False.
            ml_regressors (dict, optional): Scikit-learn ML regressors.
                The keys are strings identifying the regressors and
                the values are Scikit learn regressors. Defaults to None.
            rbf_kernels (list, optional): List of scipy rbf kernels (strings).
                Defaults to [].
            k_val (int, optional): k value for k-fold errors.
                Defaults to None.
            vor_kval_refine (int, optional): k-value for Voronoi refinement.
                Defaults to None.
            vor_interp_refine (str, optional): interpolator (string) for
                Voronoi refinement. Defaults to None.
        """

        self.folder = folder
        self.simulator = simulator
        self.n_seeds_initial = n_seeds_initial
        self.n_seeds_refine = n_seeds_refine
        self.n_seeds_stop = n_seeds_stop
        self.samp_method = samp_method
        self.bounds = bounds
        self.ranks = ranks
        self.update_basis = update_basis
        self.ml_regressors = ml_regressors
        self.rbf_kernels = rbf_kernels
        self.k_val = k_val
        self.vor_kval_refine = vor_kval_refine
        self.vor_interp_refine = vor_interp_refine

        self.dim = len(self.bounds.keys())
        self.halton_sampler = qmc.Halton(d=self.dim, seed=0)

        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            raise ValueError(
                'A ROM builder has already been started in the folder %s.'
                ' You should load that instead.' % folder)

        # Set up the logging file
        logfile = os.path.join(folder, 'output.log')
        logging.basicConfig(
            filename=logfile, level=logging.DEBUG,
            format='%(asctime)s %(message)s')

        initial_params, initial_indices = self.draw_samples(
            'halton', n_seeds_initial)
        initial_params, initial_data = self.run_forward_models(
            initial_params, initial_indices)

        # Create ROM from the initial data/parameters
        self.rom = ReducedOrderModel(
            initial_params, initial_data,
            ranks, ml_regressors, rbf_kernels)

        # Get k-fold errors and store
        _, kf_error_means = self.rom.get_kfold_errors(self.k_val)
        self.error_history = {rank: {
            interp_name: [kf_error_means[rank][interp_name]]
            for interp_name in kf_error_means[rank].keys()} for rank in ranks}
        self.nsamples_history = [self.rom.P.shape[0]]

        # Iteratively update the reduced order model
        self.train()

    @classmethod
    def from_folder(cls, folder):
        with open(os.path.join(folder, FNAME), 'rb') as f:
            return pickle.load(f)

    def draw_samples(self, sampling_method, n_samps=None):
        """Draws new samples to feed into the reduced order model.

        Args:
            sampling_method (str): Sampling method.
            n_samps (int, optional): Number of samples to draw (for sampling
                methods that use it). Defaults to None.

        Returns:
            tuple: Tuple of the samples and the indices.
        """
        logging.info('Drawing new samples..')
        min_vals = np.array([val[0] for val in self.bounds.values()])
        max_vals = np.array([val[1] for val in self.bounds.values()])
        if sampling_method == 'halton':
            samples = qmc.scale(self.halton_sampler.random(
                n=n_samps), min_vals, max_vals)
        else:
            kf_errors, _ = self.rom.get_kfold_errors(self.k_val)
            samples = voronoi.voronoi_sample(
                self.rom.P, min_vals, max_vals, kf_errors, sampling_method,
                n_samps, self.vor_kval_refine, self.vor_interp_refine)

        # Discard any samples that we already have run.
        if hasattr(self, 'rom'):
            new_samples_idxs = [
                sample.tolist() not in self.rom.P.tolist()
                for sample in samples]
            samples = samples[new_samples_idxs]
        logging.info('Drew %s new samples.' % len(samples))

        # Store samples in a dataframe
        newdf = pd.DataFrame(samples, columns=list(self.bounds.keys()))
        if hasattr(self, 'df'):
            start_idx = max(self.df.index) + 1
            self.df = pd.concat([self.df, newdf]).reset_index(drop=True)
        else:
            self.df = newdf
            start_idx = 0
        indices = list(range(start_idx, start_idx + samples.shape[0]))
        return samples, indices

    def run_forward_models(self, params, indices):
        """Execute the forward models.

        Args:
            params (array): Array of the parameter values. Each row is a
                forward model and each column is a parameter.
            indices (list): List of the indices.

        Returns:
            tuple: Tuple contain the array of parameters that were succesfully
                executed and the associated data.
        """
        logging.info(
            'Running forward models for simulation indices %s' % indices)

        labels = list(self.bounds.keys())
        params_dict = {label: param for label, param in zip(labels, params.T)}
        return self.simulator.evaluate(
            params_dict, indices=indices, folder=self.folder)

    def train(self):
        """Run the training loop to build the reduced order model.
        """
        while self.rom.P.shape[0] < self.n_seeds_stop:
            logging.info(
                'Current number of simulations: %s', self.rom.P.shape[0])
            new_params, new_indices = self.draw_samples(
                self.samp_method, self.n_seeds_refine)
            new_params, new_data = self.run_forward_models(
                new_params, new_indices)
            self.rom.update(new_params, new_data, self.update_basis)
            _, kf_error_means = self.rom.get_kfold_errors(self.k_val)

            for rank in kf_error_means.keys():
                for interp_name in kf_error_means[rank].keys():
                    self.error_history[rank][interp_name].append(
                        kf_error_means[rank][interp_name])

            self.nsamples_history.append(self.rom.P.shape[0])

            # Save the updated ROM builder
            with open(os.path.join(self.folder, FNAME), 'wb') as outp:
                pickle.dump(self, outp)

        logging.info(
            'Finished training the ROM. Ended with %s simulations.' %
            self.rom.P.shape[0])
