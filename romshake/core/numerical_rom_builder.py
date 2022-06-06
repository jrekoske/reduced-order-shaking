import os
import shutil
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.stats import qmc
from matplotlib import pyplot as plt

from romshake.sample import voronoi

FNAME = 'rom_builder.pkl'
LOG_FILE = 'output.log'


class NumericalRomBuilder():
    def __init__(self, folder, simulator, rom, n_seeds_initial, n_seeds_refine,
                 n_seeds_stop, samp_method, bounds, clear,
                 desired_score, make_test_figs):
        """Class for building reduced-order models from numerical simulations.

        Args:
            folder (str): Path associated with ROM data.
            simulator (object): Simulator for generating
                new data from parameters. Can be either analytic or launch
                numerical simulation jobs.
            rom (object): Reduced order model object.
            n_seeds_initial (int): Number of seeds for the first iteration.
            n_seeds_refine (int): Number of seeds to generate with each
                iteration.
            n_seeds_stop (int): Maximum number of seeds.
            samp_method (str): Sampling refinement strategy.
            bounds (dict): Min/max values for the parameter space.
            clear (bool, optional): If True, will remove all pre-existing
                data from the folder.
            desired_score (float): Desired score (stopping condition).
            make_test_figs (bool): Plot snapshots of testing predictions.
        """

        self.folder = folder
        self.simulator = simulator
        self.rom = rom
        self.n_seeds_initial = n_seeds_initial
        self.n_seeds_refine = n_seeds_refine
        self.n_seeds_stop = n_seeds_stop
        self.samp_method = samp_method
        self.bounds = bounds
        self.dim = len(self.bounds.keys())
        self.halton_sampler = qmc.Halton(d=self.dim, seed=0)
        self.desired_score = desired_score
        self.make_test_figs = make_test_figs

        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            if clear:
                shutil.rmtree(folder)
                os.makedirs(folder)
            else:
                raise ValueError(
                    'A ROM builder has already been started in the folder %s.'
                    ' You should load that instead.' % folder)

        # Set  up the logger
        logfile = os.path.join(folder, LOG_FILE)
        logging.basicConfig(
            filename=logfile, level=logging.DEBUG,
            format='%(asctime)s %(message)s')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
            scores = [self.metric_func(pred, true) for pred, true in zip(
                self.rom.y_pred, self.rom.y_test)]
            samples = voronoi.voronoi_sample(
                self.rom.X_test, min_vals, max_vals, scores, sampling_method,
                n_samps, plot=True)

        # Discard any samples that we already have run.
        if hasattr(self.rom, 'X'):
            new_samples_idxs = [
                sample.tolist() not in self.rom.X.tolist()
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
        if hasattr(self.rom, 'X'):
            nseeds = self.rom.X.shape[0]
            best_score = np.nanmax(self.score_history)
        else:
            nseeds = 0
            best_score = -1e10  # aribtrary to make it work
        while (nseeds < self.n_seeds_stop and best_score < self.desired_score):
            logging.info(
                'Current number of simulations: %s', nseeds)
            if nseeds == 0:
                new_params, new_indices = self.draw_samples(
                    'halton', self.n_seeds_initial)
            else:
                new_params, new_indices = self.draw_samples(
                    self.samp_method, self.n_seeds_refine)
            new_params, new_data = self.run_forward_models(
                new_params, new_indices)
            self.rom.update(new_params, new_data.T)
            self.save_results()
            nseeds = self.rom.X.shape[0]
            best_score = np.nanmax(self.score_history)
        logging.info(
            'Finished training the ROM. Ended with %s simulations.' %
            self.rom.X.shape[0])

    def save_results(self):
        nseeds = self.rom.X.shape[0]
        odir = os.path.join(self.folder, str(nseeds))
        if not os.path.exists(odir):
            os.makedirs(odir)
        pd.DataFrame(
            self.rom.search.cv_results_).sort_values(
                by='rank_test_score').to_csv(os.path.join(
                    odir, 'grid_search_results.csv'), index=False)
        if hasattr(self, 'nsamples_history'):
            self.nsamples_history.append(nseeds)
            self.score_history.append(self.rom.search.best_score_)
        else:
            self.nsamples_history = [nseeds]
            self.score_history = [self.rom.search.best_score_]
        with open(os.path.join(self.folder, FNAME), 'wb') as outp:
            pickle.dump(self, outp)
        if self.make_test_figs:
            self.plot_predictions(odir)
        self.plot_score_history()

    def plot_score_history(self):
        plt.plot(self.nsamples_history, self.score_history, '-ko')
        plt.axhline(self.desired_score, c='k', ls='--', lw=0.5)
        plt.xlabel('Number of samples')
        plt.ylabel('Score')
        plt.savefig(os.path.join(self.folder, 'score_history.png'))
        plt.close()

    def plot_predictions(self, odir):
        data = [self.rom.y_test, self.rom.y_pred,
                self.rom.y_test - self.rom.y_pred]
        titles = ['Truth', 'Predicted', 'Error']
        cmaps = [None, None, 'bwr']
        for i in range(self.rom.y_test.shape[0]):
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
            for ax, ds, title, cmap in zip(axes, data, titles, cmaps):
                if ax == axes[-1]:
                    absmax = abs(ds[i]).max()
                    vmin = -absmax
                    vmax = absmax
                else:
                    vmin = min(data[0][i].min(), data[1][i].min())
                    vmax = max(data[0][i].max(), data[1][i].max())
                self.simulator.plot_snapshot(ax, ds[i], vmin=vmin, vmax=vmax,
                                             title=title, cmap=cmap)
            fig.tight_layout()
            fig.savefig(os.path.join(
                odir, 'pred_%s.png' % self.rom.X_test[i]), dpi=100)
            plt.close()
