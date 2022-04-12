import logging
import numpy as np
from sklearn import preprocessing
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold
from sklearn.base import clone


class ReducedOrderModel():
    def __init__(self, P, Q, ranks, ml_regressors, rbf_kernels):
        """Class for encapusaling reduced order model information.

        Args:
            P (array): Parameter array.
            Q (array): Data array.
            rank (int): Ranks of the POD basis.
            ml_regressors (dict, optional): Scikit-learn ML regressors.
                The keys are strings identifying the regressors and
                the values are Scikit learn regressors. Defaults to None.
            rbf_kernels (list): List of rbf kernels (strings).
                Must be a valid kernel to use with
                scipy.interpolate.RBFInterpolator.
        """
        self.P = P
        self.Q = Q
        self.ranks = ranks
        self.ml_regressors = ml_regressors
        self.rbf_kernels = rbf_kernels

        self.compute_svd()
        self.compute_pod_coefficients()
        self.fit_interpolators()

    def compute_svd(self):
        """Computes the SVD of the data.
        """
        logging.info('Performing SVD.')
        self.u, self.s, self.vh = np.linalg.svd(self.Q, full_matrices=False)

    def compute_pod_coefficients(self):
        """Computes the POD coefficients and standardizes the inputs and
        outputs for training.
        """
        logging.info('Computing POD coefficients.')
        self.A = {rank: self.Q.T @ self.u[:, 0:rank] for rank in self.ranks}

        self.Pscaler = preprocessing.StandardScaler().fit(self.P)
        self.P_scaled = self.Pscaler.transform(self.P)

        self.Ascalers = {
            rank: preprocessing.StandardScaler().fit(self.A[rank])
            for rank in self.ranks}
        self.A_scaled = {
            rank: self.Ascalers[rank].transform(self.A[rank])
            for rank in self.ranks}

    def fit_interpolators(self):
        """Fits the interpolators (rbf kernels and machine learning-based
        interpolators."""
        self.interpolators = {}
        for rank in self.ranks:
            self.interpolators[rank] = {}
            for ml_name, ml_regr in self.ml_regressors.items():
                new_regr = clone(ml_regr)
                self.interpolators[rank][ml_name] = new_regr.fit(
                    self.P_scaled, self.A_scaled[rank])
            for kernel in self.rbf_kernels:
                self.interpolators[rank][kernel] = RBFInterpolator(
                    self.P_scaled, self.A_scaled[rank], kernel=kernel)

    def update(self, newP, newQ, update_basis):
        """Updates an existing reduced order model with new parameters/data.

        Args:
            newP (array): New parameter array.
            newQ (array): New data array.
            update_basis (bool): Whether to update the POD basis with newly
                added data.
        """
        logging.info(
            'Updating the reduced order model with %s new simulations.' %
            newP.shape[0])
        self.P = np.concatenate((self.P, newP), axis=0)
        self.Q = np.concatenate((self.Q, newQ), axis=1)
        if update_basis:
            self.compute_svd()
        self.compute_pod_coefficients()
        self.fit_interpolators()

    def predict(self, Ppred):
        """Predicts the output data using the reduced order model.

        Args:
            Ppred (array): Array of prediction parameters.

        Returns:
            dict: Dictionary of predictions where the keys indicate the
                interpolator and the values are the output predictions.

        """
        Ppred_trans = self.Pscaler.transform(Ppred)
        Qpred = {}
        for rank in self.ranks:
            Qpred[rank] = {}
            for interp_name, interp in self.interpolators[rank].items():
                try:
                    Apred = interp.predict(Ppred_trans)
                except AttributeError:
                    Apred = interp(Ppred_trans)

                self.Ascalers[rank].inverse_transform(Apred).T
                Qpred[rank][interp_name] = self.u[:, 0:rank] @ self.Ascalers[
                    rank].inverse_transform(Apred).T
        return Qpred

    def get_kfold_errors(self, kval):
        """Peforms k-fold cross-validation on the reduced order model.

        Args:
            kval (int): k-value.

        Returns:
            tuple: Tuple of dictionaries. First contains the interpolators
            as the keys and the list of errors as the values. Second contains
            the interpolators as keys and the mean of errors as the values.
        """
        logging.info('Obtaining kfold errors.')
        npoints = self.P.shape[0]
        if kval == 'loo':
            kval = npoints
        interp_names = list(self.ml_regressors.keys()) + self.rbf_kernels
        kf = KFold(n_splits=kval, shuffle=True)
        kf_errors = {rank: {interp_name: np.zeros(
            npoints) for interp_name in interp_names} for rank in self.ranks}
        for train, test in kf.split(self.P):
            rom = ReducedOrderModel(
                self.P[train], self.Q[:, train], self.ranks,
                self.ml_regressors, self.rbf_kernels)
            pred_dict = rom.predict(self.P[test])
            true = self.Q[:, test]
            for rank in self.ranks:
                for interp_name, pred_vals in pred_dict[rank].items():
                    kf_errors[rank][interp_name][test] = np.linalg.norm(
                        pred_vals - true, axis=0) / np.linalg.norm(
                            true, axis=0)
        kf_error_means = {rank: {
            interp_name: np.mean(kf_errors[rank][interp_name])
            for interp_name in kf_errors[rank].keys()} for rank in self.ranks}
        return kf_errors, kf_error_means
