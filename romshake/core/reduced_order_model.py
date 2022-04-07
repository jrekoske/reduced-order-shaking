import logging
import numpy as np
from sklearn import preprocessing
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold


class ReducedOrderModel():
    def __init__(self, P, Q, rank, ml_regressors, ml_names, rbf_kernels):
        """Class for encapusaling reduced order model information.

        Args:
            P (array): Parameter array.
            Q (array): Data array.
            rank (int): Rank of the POD basis.
            ml_regressors (list): List of Scikit-learn regressors.
            ml_names (list): List of names (strings) associated with
                ml_regressors.
            rbf_kernels (list): List of rbf kernels (strings).
                Must be a valid kernel to use with
                scipy.interpolate.RBFInterpolator.
        """
        self.P = P
        self.Q = Q
        self.rank = rank
        self.ml_regressors = ml_regressors
        self.ml_names = ml_names
        self.rbf_kernels = rbf_kernels

        self.compute_svd()
        self.compute_pod_coefficients()
        self.fit_interpolators()

    def compute_svd(self):
        """Computes the SVD of the data.
        """
        logging.info('Performing SVD.')
        self.u, self.s, self.vh = np.linalg.svd(self.Q, full_matrices=False)
        if self.rank:
            self.u_rank = self.u[:, 0:self.rank]
        else:
            self.u_rank = self.u

    def compute_pod_coefficients(self):
        """Computes the POD coefficients and standardizes the inputs and
        outputs for training.
        """
        logging.info('Computing POD coefficients.')
        self.A = self.Q.T @ self.u_rank
        self.Pscaler = preprocessing.StandardScaler().fit(self.P)
        self.Ascaler = preprocessing.StandardScaler().fit(self.A)
        self.P_scaled = self.Pscaler.transform(self.P)
        self.A_scaled = self.Ascaler.transform(self.A)

    def fit_interpolators(self):
        """Fits the interpolators (rbf kernels and machine learning-based
        interpolators."""
        for i in range(len(self.ml_regressors)):
            self.ml_regressors[i].fit(self.P_scaled, self.A_scaled)
        self.rbf_interps = [
            RBFInterpolator(self.P_scaled, self.A_scaled, kernel=kernel)
            for kernel in self.rbf_kernels]

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
        self.Ppred_trans = self.Pscaler.transform(Ppred)
        self.Apred = {}
        for i, regr in enumerate(self.ml_regressors):
            self.Apred[self.ml_names[i]] = regr.predict(self.Ppred_trans)
        for i, regr in enumerate(self.rbf_interps):
            self.Apred[self.rbf_kernels[i]] = regr(self.Ppred_trans)
        return {
            key: self.u_rank @ self.Ascaler.inverse_transform(Apred_interp).T
            for key, Apred_interp in self.Apred.items()}

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
        interp_names = self.ml_names + self.rbf_kernels
        kf = KFold(n_splits=kval, shuffle=True)
        kf_errors = {interp_name: np.zeros(npoints)
                     for interp_name in interp_names}
        for train, test in kf.split(self.P):
            rom = ReducedOrderModel(
                self.P[train], self.Q[:, train], self.rank,
                self.ml_regressors, self.ml_names, self.rbf_kernels)
            pred_dict = rom.predict(self.P[test])
            true = self.Q[:, test]
            for interp_name, pred_vals in pred_dict.items():
                kf_errors[interp_name][test] = np.linalg.norm(
                    pred_vals - true, axis=0) / np.linalg.norm(true, axis=0)
        kf_error_means = {
            interp_name: np.mean(kf_errors[interp_name])
            for interp_name in kf_errors.keys()}
        # logging.info('ROM errors: %s' % kf_error_means)
        return kf_errors, kf_error_means
