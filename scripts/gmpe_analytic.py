import numpy as np
import noise

from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
from openquake.hazardlib.contexts import SitesContext, DistancesContext
from openquake.hazardlib.imt import PGV
from openquake.hazardlib.const import StdDev

from shakelib.rupture.quad_rupture import QuadRupture
from shakelib.rupture.origin import Origin

# number of points in each dimension
N = 50


def evaluate(dep, strike=0, length=30, dip=90, width=30, add_noise=True):
    gmpe = AbrahamsonEtAl2014()
    sx = SitesContext()
    dx = DistancesContext()
    origin = Origin({
        'id': '',
        'netid': '',
        'network': '',
        'lat': 0,
        'lon': 0,
        'depth': dep,
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
        pz=[dep],
        dx=[15],
        dy=[0],
        length=[length],
        width=[width],
        strike=[strike],
        dip=[dip],
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

    res = gmpe.get_mean_and_stddevs(sx, rx, dx, PGV(), [StdDev.TOTAL])[0]
    shape = (N, N)
    res = res.reshape(shape)

    scale = 100.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    if add_noise:
        world = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                world[i][j] = noise.pnoise2(i/scale,
                                            j/scale,
                                            octaves=octaves,
                                            persistence=persistence,
                                            lacunarity=lacunarity,
                                            repeatx=1024,
                                            repeaty=1024,
                                            base=42)

        scale = 0.3 * res.max() / world.max()
        res = res + scale * world
    return res.flatten()