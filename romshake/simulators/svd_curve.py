import numpy as np


class SVDCurveSimulator():
    def __init__(self, base_sim, base_params_dict, p, smin, smax, fast):
        self.base_sim = base_sim
        # self.sv_curve = sv_curve(m, p, smin, smax, fast)
        self.base_data = self.base_sim.evaluate(base_params_dict)[1]
        self.p = p
        self.smin = smin
        self.smax = smax
        self.fast = fast

    def evaluate(self, params_dict, **kwargs):
        params, data = self.base_sim.evaluate(params_dict)
        nnew = data.shape[1]
        newdata = np.hstack((self.base_data, data))
        u, s, vh = np.linalg.svd(newdata, full_matrices=False)
        news = sv_curve(len(s), self.p, self.smin, self.smax, self.fast)
        newdata = u @ np.diag(news) @ vh[:, -nnew:]
        return np.array(list(params_dict.values())).T, newdata

    def plot_snapshot(self, ax, snap, **kwargs):
        self.base_sim.plot_snapshot(ax, snap, **kwargs)


def sv_curve(m, p, smin, smax, fast):
    x = np.linspace(smin, smax, m)[::-1]
    ds = smax - smin
    if fast:
        y = 10**((1 / ds**(p-1)) * (x - smin)**p + smin)
    else:
        y = 10**(-(1 / ds**(p-1)) * (smax - x)**p + smax)
    return y
