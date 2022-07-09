import os
import pickle
import logging
import numpy as np
from tensorflow import keras
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler
from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor

# CPU
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor  # NOQA
from sklearn.neighbors import KNeighborsRegressor  # NOQA

# Local
from romshake.core.rbf_regressor import RBFRegressor
from romshake.core.remote_controller import copy_file


class ReducedOrderModel():
    def __init__(
            self, regressors, svd_ncomps, test_size, scoring, folder,
            remote=None):
        """Class for encapsulating reduced order model information.

        Args:
            parameters (dict, optional): Dictionary of parameters
                for grid search of ML hyperparameters.
            test_size (float): Fraction of data (forward models) to
                holdout from training.
            scoring (str): Scorer string (scikit-learn).
            remote (object, optional): Remote controller object.
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
        self.remote = remote
        self.folder = folder

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
        if self.remote is not None:
            self.launch_remote_grid_search()
        else:
            self.train_search_models()

    def train_search_models(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                self.X, self.y, test_size=self.test_size)
        regressor = Pipeline(
            steps=[('scaler', StandardScaler()), ('reg', RBFRegressor())])
        transformer = Pipeline([
            ('svd', TruncatedSVD()),
            ('yscaler', preprocessing.StandardScaler())])
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

    def launch_remote_grid_search(self):
        job_dir = os.path.join(self.folder, 'jobs')
        if os.path.exists(job_dir):
            jobfiles = [file for file in os.listdir(job_dir) if 'job' in file]
            jobidx = max([int(file[-1]) for file in jobfiles]) + 1
        else:
            os.makedirs(job_dir)
            jobidx = 0
        remote_job_file_loc = os.path.join(
            self.remote.remote_wdir, 'jobs', 'job%s' % jobidx)
        copy_file(os.path.join(
            self.folder, 'index_params.csv'), self.remote.remote_wdir)
        copy_file(self.remote.grid_search_job_file, remote_job_file_loc)
        copy_file(self.remote.grid_search_script, self.remote.remote_wdir)
        copy_file('config.yml', self.remote.remote_wdir)

        self.remote.run_jobs([jobidx])
        files_to_copy = ['search_results.pkl']
        for file in files_to_copy:
            copy_file(os.path.join(
                self.remote.remote_wdir, file), self.folder)
        with open('search_results.pkl', 'rb') as inp:
            self.search = pickle.load(inp)

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
