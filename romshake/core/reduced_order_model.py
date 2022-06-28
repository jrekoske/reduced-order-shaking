import os
import shutil
import logging
import numpy as np
import pandas as pd
from joblib import Memory
from tensorflow import keras
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasRegressor
from dask_ml.model_selection import GridSearchCV

# CPU
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # NOQA
from sklearn.neural_network import MLPRegressor  # NOQA
from sklearn.neighbors import KNeighborsRegressor  # NOQA

# GPU
# from cuml.dask.decomposition import TruncatedSVD  # NOQA
# from cuml.neighbors import KNeighborsRegressor  # NOQA
# from cuml.dask.ensemble import RandomForestRegressor  # NOQA

# Local
from romshake.core.rbf_regressor import RBFRegressor
from romshake.simulators.remote import REMOTE_DIR, copy_file, run_jobs

# Keras neural network model
def get_nn_model(hidden_layer_dim, n_hidden_layers, meta):
    n_features_in_ = meta['n_features_in_']
    X_shape_ = meta['X_shape_']
    n_outputs_ = meta['n_outputs_']
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
    model.add(keras.layers.Activation('relu'))
    for i in range(n_hidden_layers):
        model.add(keras.layers.Dense(hidden_layer_dim))
        model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(n_outputs_))
    return model


class ReducedOrderModel():
    def __init__(
            self, regressors, svd_ncomps, test_size, scoring,
            remote_grid_search=None, folder=None, grid_search_job_file=None,
            grid_search_script=None):
        """Class for encapsulating reduced order model information.

        Args:
            parameters (dict, optional): Dictionary of parameters
                for grid search of ML hyperparameters.
            test_size (float): Fraction of data (forward models) to
                holdout from training.
            scoring (str): Scorer string (scikit-learn).
        """
        self.hyper_params = []
        for rname, hypers in regressors.items():
            if rname == 'NeuralNetwork':
                rdict = {'regressor__reg':  [KerasRegressor(
                    get_nn_model, loss='mse', optimizer='adam', **hypers)]}
            else:
                rdict = {'regressor__reg': [globals()[rname](**hypers)]}
            for hyp_name, hyp_val in hypers.items():
                rdict['regressor__reg__%s' % hyp_name] = hyp_val
            rdict['transformer__svd__n_components'] = svd_ncomps
            self.hyper_params.append(rdict)
        self.test_size = test_size
        self.scoring = scoring
        self.remote_grid_search = remote_grid_search
        self.folder = folder
        self.grid_search_job_file = grid_search_job_file
        self.grid_search_script = grid_search_script

    def update(self, newX, newy):
        """Updates an existing reduced order model with new parameters/data.

        Args:
            newX (array): New parameter array.
            newy (array): New data array.
        """
        logging.info(
            'Adding %s new simulations to the reduced order model.' %
            newX.shape[0])
        if hasattr(self, 'X'):
            self.X = np.concatenate((self.X, newX))
            self.y = np.concatenate((self.y, newy))
        else:
            self.X = newX
            self.y = newy
        if self.remote_grid_search:
            self.launch_remote_grid_search()
        else:
            self.train_search_models()

    def train_search_models(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size)
        regressor = Pipeline(
            steps=[('scaler', StandardScaler()), ('reg', RBFRegressor())])
        location = 'cachedir'
        memory = Memory(location=location, verbose=10)
        transformer = Pipeline([
            ('svd', TruncatedSVD()),
            ('yscaler', preprocessing.StandardScaler())], memory=memory)
        trans_regr = TransformedTargetRegressor(
            regressor=regressor, transformer=transformer, check_inverse=False)
        print(self.hyper_params)
        search = GridSearchCV(trans_regr, self.hyper_params,
                              scoring=self.scoring)
        logging.info('Starting grid search of model hyperparameters.')
        search.fit(self.X_train, self.y_train)
        logging.info('The best parameters are: %s' % search.best_params_)
        logging.info('The best score is: %s' % search.best_score_)
        test_score = search.score(self.X_test, self.y_test)
        logging.info('The score on the testing data is: %s' % test_score)
        self.y_pred = search.predict(self.X_test)
        self.search = search

        # Delete the temporary cache directory
        memory.clear(warn=False)
        shutil.rmtree(location)

    def launch_remote_grid_search(self):
        job_dir = os.path.join(self.folder, 'jobs')
        if os.path.exists(job_dir):
            jobfiles = [file for file in os.listdir(job_dir) if 'job' in file]
            jobidx = max([int(file[-1]) for file in jobfiles]) + 1
        else:
            os.makedirs(job_dir)
            jobidx = 0
        remote_job_file_loc = os.path.join(
            REMOTE_DIR, self.folder, 'jobs', 'job%s' % jobidx)
        copy_file(self.grid_search_job_file, remote_job_file_loc)
        copy_file(self.grid_search_script, os.path.join(
            REMOTE_DIR, self.folder))
        run_jobs([jobidx], self.folder)

        files_to_copy = ['X.npy', 'ypred.npy', 'grid_search_results.csv']
        for file in files_to_copy:
            copy_file(os.path.join(REMOTE_DIR, self.folder, file), self.folder)

        X = np.load(os.path.join(self.folder, 'X.npy'))
        ypred = np.load(os.path.join(self.folder, 'ypred.npy'))
        search_df = pd.read_csv(os.path.join(self.folder, 'grid_search_results.csv'))
        return (X, ypred, search_df)
