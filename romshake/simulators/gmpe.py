import warnings
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from openquake.hazardlib.contexts import SitesContext, DistancesContext
from openquake.hazardlib.imt import PGV
from openquake.hazardlib.const import StdDev

from shakelib.rupture.quad_rupture import QuadRupture
from shakelib.rupture.origin import Origin

warnings.filterwarnings('ignore')

N = 70
shape = (N, N)
default_source_params = {
    'x': 0,
    'y': 0,
    'depth': 0,
    'strike': 0,
    'length': 20,
    'width': 20,
    'dip': 90
}


class GMPE_Simulator():
    def __init__(self, gmpe, add_noise, noise_scale=None, sigmax=None,
                 sigmay=None):
        """Creates a GMPE Simulator object.

        Args:
            gmpe (openquake.hazardlib.gsim GMM): Openquake GMPE
            add_noise (bool): Whether to add noise to GMPE result.
            noise_scale (float): Value to scale noise.
        """
        self.gmpe = gmpe
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.sigmax = sigmax
        self.sigmay = sigmay

    def evaluate(
            self, params_dict, **kwargs):
        sx = SitesContext()
        dx = DistancesContext()

        nquakes = len(list(params_dict.values())[0])
        source_params = {}
        for param, default_val in default_source_params.items():
            if param in params_dict:
                source_params[param] = params_dict[param]
            else:
                source_params[param] = np.full(nquakes, default_val)
        all_data = []

        for i in range(nquakes):
            origin = Origin({
                'id': '',
                'netid': '',
                'network': '',
                'lat': 0,
                'lon': 0,
                'depth': source_params['depth'][i],
                'locstring': '',
                'mag': 6.0,
                'time': '',
                'mech': '',
                'reference': '',
                'productcode': ''
            })
            rup = QuadRupture.fromOrientation(
                px=[source_params['x'][i]],
                py=[source_params['y'][i]],
                pz=[source_params['depth'][i]],
                dx=[source_params['length'][i]/2],
                dy=[0],
                length=[source_params['length'][i]],
                width=[source_params['width'][i]],
                strike=[source_params['strike'][i]],
                dip=[source_params['dip'][i]],
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

            rx = rup.getRuptureContext([self.gmpe])

            sx.vs30 = np.full_like(X_flat, 760)
            sx.vs30measured = np.full_like(X_flat, False, dtype=bool)
            sx.z1pt0 = np.full_like(X_flat, 48)

            res = self.gmpe.get_mean_and_stddevs(
                sx, rx, dx, PGV(), [StdDev.TOTAL])[0]
            res = res.reshape(shape)
            if self.add_noise:
                res = res + self.get_noise(res)
            res = res.flatten()
            all_data.append(res)

        return np.array(list(params_dict.values())).T, np.array(all_data).T

    def get_noise(self, res):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(N, N))
        filt = gaussian_filter(data, sigma=[self.sigmax, self.sigmay])
        scale = self.noise_scale * res.max() / filt.max()
        return scale * filt

    def get_successful_indices(self, folder, indices):
        return indices

    def plot_snapshot(self, ax, snap, vmin, vmax, title, cmap, **kwargs):
        im = ax.imshow(snap.reshape(shape), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='log(PGV)')
