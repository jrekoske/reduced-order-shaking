import numpy as np


class AnalyticSimulator():
    def __init__(self, func):
        self.func = func

    def evaluate(self, params_dict, **kwargs):
        nparams = len(list(params_dict.values())[0])
        data = []
        for i in range(nparams):
            sim_params = {
                param: vals[i] for param, vals in params_dict.items()}
            data.append(self.func(**sim_params))
        return np.array(list(params_dict.values())).T, np.array(data).T

    def get_successful_indices(self, folder, indices):
        return indices

    def plot_snapshot(self, ax, snap, **kwargs):
        ax.plot(snap)
