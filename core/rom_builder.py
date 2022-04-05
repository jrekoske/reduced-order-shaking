import warnings
import itertools
import numpy as np
from scipy.stats import qmc
from sklearn import preprocessing
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold

# TODO: add functionality for starting with numerical data
# already present in a folder

# TODO: think about masks, and re-ordering elements from numerical simulation


class RomBuilder():
    def __init__(
            self, is_analytic, n_initial_samps, n_samps_refine, n_samps_stop,
            analytic_func=None, lower_bounds=None, upper_bounds=None,
            n_truth=None, rank=None, update_basis=False, ml_regressors=[],
            ml_names=[],
            rbf_kernels=[],
            samp_method=None, truth=None, data_folder=None, k_val_error=None):
        """Class for building a reduced-order model from an analytic function.

        Args:
            analytic_func (func): Analytic function, number of inputs defines
                the dimension of the ROM.
            lower_bounds (list): Minimum values for the parameter space.
            upper_bounds (list): Maximum values for the parameter space.
            n_truth (int): Number of points along each dimension that
                defines the "true" grid.
            n_initial_samps (int): Number of initial points to sample in
                the first iteration.
            regressor: Scikit-learn estimator or scipy interpolator.
            samp_method (str): Sampling refinement strategy.
            n_samps_refine (int): Number of samples to generate with each
                iteration (if the strategy needs it).
            n_samps_stop (int): Stop improving the ROM after generating
                this many samples.
            rank (int): Limit the rank to this value.
            fixed_basis (bool): Whether to update the POD basis when drawing
                new samples.
        """

        self.is_analytic = is_analytic
        self.analytic_func = analytic_func
        self.dim = len(lower_bounds)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.samp_method = samp_method
        self.n_samps_refine = n_samps_refine
        self.n_samps_stop = n_samps_stop
        self.ml_regressors = ml_regressors
        self.ml_names = ml_names
        self.rbf_kernels = rbf_kernels
        self.rank = rank
        self.error_history = []
        self.nsamples_list = []
        self.update_basis = update_basis
        self.data_folder = data_folder
        self.truth = truth
        self.k_val_error = k_val_error

        if self.is_analytic:
            self.Ptruth = np.linspace(lower_bounds, upper_bounds, n_truth)
            self.Ptruth_stacked = np.array(list(itertools.product(*self.Ptruth.T)))
            if self.truth is None:
                self.truth = self.evaluate_forward_model(self.Ptruth_stacked)

        self.halton_sampler = qmc.Halton(d=self.dim, seed=0)
        self.draw_samples(n_initial_samps, initial=True)
        self.compute_pod_coefficients()
        self.get_error(initial=True)
        self.train()

    def draw_samples(self, n_samps, initial=False):
        # TODO: add Voronoi sampling
        newP = qmc.scale(self.halton_sampler.random(
            n=n_samps), self.lower_bounds, self.upper_bounds)
        newQ = self.evaluate_forward_model(newP)
        if initial:
            self.P = newP
            self.Q = newQ
            self.compute_svd()
        else:
            self.P = np.concatenate((self.P, newP), axis=0)
            self.Q = np.concatenate((self.Q, newQ), axis=1)
            if self.update_basis:
                self.compute_svd()

    def compute_svd(self):
        self.u, self.s, self.vh = np.linalg.svd(self.Q)
        if self.rank:
            self.u_rank = self.u[:, 0:self.rank]
        else:
            self.u_rank = self.u

    def train(self):
        while self.P.shape[0] < self.n_samps_stop:
            print('nsamples:', self.P.shape[0])
            self.draw_samples(self.n_samps_refine)
            self.compute_pod_coefficients()
            self.get_error()

    def compute_pod_coefficients(self):
        self.A = self.Q.T @ self.u_rank
        self.Pscaler = preprocessing.StandardScaler().fit(self.P)
        self.Ascaler = preprocessing.StandardScaler().fit(self.A)

        self.P_scaled = self.Pscaler.transform(self.P)
        self.A_scaled = self.Ascaler.transform(self.A)

        for i in range(len(self.ml_regressors)):
            self.ml_regressors[i].fit(self.P_scaled, self.A_scaled)

        self.rbf_interps = [
            RBFInterpolator(self.P_scaled, self.A_scaled, kernel=kernel)
            for kernel in self.rbf_kernels]

    def get_error(self, initial=False):
        if self.is_analytic:
            # If we're using an analytic model, we can obtain the "true" error,
            # so let's use that
            self.pred = self.predict(self.Ptruth_stacked)
            self.error = {
                key: np.linalg.norm(self.truth, pred_interp) / np.linalg.norm(
                    self.truth)
                for key, pred_interp in self.pred.items()}
        else:
            # If numeric, then we can only get the k-fold error since
            # we don't know the "truth"
            npoints = self.P.shape[0]
            if self.k_val_error == 'loo':
                self.k_val_error = npoints
            kf = KFold(n_splits=self.k_val_error, shuffle=True)
            kf_errors = np.zeros(npoints)
            P_all, Q_all = np.copy(self.P), np.copy(self.Q)
            for train, test in kf.split(self.P):
                # Temporarily set the P and Q values for the ROM
                # to evaluate error for the subset
                self.P, self.Q = self.P[train], self.Q[train]
                self.compute_svd()
                self.compute_pod_coefficients()
                pred_dict = self.predict(P_all[test])
                true = Q_all[test]

                # TODO: check axis=1

                kf_errors[test] = np.linalg.norm(
                    pred - true, axis=1) / np.linalg.norm(true, axis=1)
                self.P = P_all, self.Q = Q_all
            # TODO: think about appropriate error measurement for k-fold (mean?)
            self.error = np.mean(kf_errors)
        if initial:
            self.error_history = {
                key: [val] for key, val in self.error.items()}
        else:
            for regr in self.error_history.keys():
                self.error_history[regr].append(self.error[regr])
        self.nsamples_list.append(self.P.shape[0])

    def evaluate_forward_model(self, P):
        if self.is_analytic:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.array(
                    [self.analytic_func(*param) for param in P]).T
        else:
            # Perform all the numerical simulation stuff and return data in
            # an array to use for the ROM
            return

    def predict(self, Ppred):
        self.Ppred_trans = self.Pscaler.transform(Ppred)

        self.Apred = {}
        for i, regr in enumerate(self.ml_regressors):
            self.Apred[self.ml_names[i]] = regr.predict(self.Ppred_trans)
        for i, regr in enumerate(self.rbf_interps):
            self.Apred[self.rbf_kernels[i]] = regr(self.Ppred_trans)
        return {
            key: self.u_rank @ self.Ascaler.inverse_transform(Apred_interp).T
            for key, Apred_interp in self.Apred.items()}
