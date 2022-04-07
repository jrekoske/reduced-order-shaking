import os
import noise
import warnings
import numpy as np

from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
from openquake.hazardlib.contexts import SitesContext, DistancesContext
from openquake.hazardlib.imt import PGV
from openquake.hazardlib.const import StdDev
import pandas as pd

from shakelib.rupture.quad_rupture import QuadRupture
from shakelib.rupture.origin import Origin

warnings.filterwarnings('ignore')

N = 30
shape = (N, N)

par_csv_file = 'parameters.csv'

source_params = {
    'depth': 0,
    'strike': 0,
    'length': 10,
    'width': 10,
    'dip': 90
}


class GMPE_Simulator():
    def __init__(self, add_noise=True):
        self.add_noise = add_noise

    def evaluate(self, rb, params, indices=None, folder=None, write=True):
        gmpe = AbrahamsonEtAl2014()
        sx = SitesContext()
        dx = DistancesContext()

        all_data = []
        for i, param in enumerate(params):
            if indices:
                idx = indices[i]
            param_labels = rb.bounds.keys()
            for label, val in zip(param_labels, param):
                source_params[label] = val
            origin = Origin({
                'id': '',
                'netid': '',
                'network': '',
                'lat': 0,
                'lon': 0,
                'depth': source_params['depth'],
                'locstring': '',
                'mag': 6.0,
                'time': '',
                'mech': '',
                'reference': '',
                'productcode': ''
            })
            rup = QuadRupture.fromOrientation(
                px=[0],
                py=[0],
                pz=[source_params['depth']],
                dx=[15],
                dy=[0],
                length=[source_params['length']],
                width=[source_params['width']],
                strike=[source_params['strike']],
                dip=[source_params['dip']],
                origin=origin)

            lons = np.linspace(-0.5, 0.5, N)
            lats = np.linspace(-0.5, 0.5, N)
            X, Y = np.meshgrid(lons, lats)
            X_flat, Y_flat = X.flatten(), Y.flatten()
            Z_flat = np.zeros_like(X_flat)

            dists = rup.computeGC2(X_flat, Y_flat, Z_flat)
            dx.rjb = rup.computeRjb(X_flat, Y_flat, Z_flat)[0]
            dx.rrup = rup.computeRrup(X_flat, Y_flat, Z_flat)[0]
            dx.rx = dists['rx']
            dx.ry0 = dists['ry0']

            rx = rup.getRuptureContext([gmpe])

            sx.vs30 = np.full_like(X_flat, 760)
            sx.vs30measured = np.full_like(X_flat, False, dtype=bool)
            sx.z1pt0 = np.full_like(X_flat, 48)

            res = gmpe.get_mean_and_stddevs(
                sx, rx, dx, PGV(), [StdDev.TOTAL])[0]
            res = res.reshape(shape)
            if self.add_noise:
                res = res + self.get_noise(res)
            res = res.flatten()
            all_data.append(res)

            # Save the result
            if write:
                sim_dir = os.path.join(folder, 'data', str(idx))
                if not os.path.exists(sim_dir):
                    os.makedirs(sim_dir)
                np.save(os.path.join(sim_dir, 'result.npy'), res)
        return params, np.array(all_data).T

    def get_noise(self, res):
        scale = 100.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        world = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                world[i][j] = noise.pnoise2(
                    i/scale, j/scale, octaves=octaves, persistence=persistence,
                    lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=42)
        scale = 0.3 * res.max() / world.max()
        return scale * world

    def load_data(self, folder, indices):
        df = pd.read_csv(os.path.join(folder, par_csv_file), index_col=0)
        df = df.iloc[indices]
        params = df.values
        all_data = []
        for idx in indices:
            res = np.load(os.path.join(folder, 'data', str(idx), 'result.npy'))
            all_data.append(res)
        return params, np.array(all_data).T

    def get_successful_indices(self, folder, indices):
        good_indices = []
        for idx in indices:
            if os.path.exists(os.path.join(
                    folder, 'data', str(idx), 'result.npy')):
                good_indices.append(idx)
        return good_indices
