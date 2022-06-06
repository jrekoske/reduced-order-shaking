import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from romshake.core.numerical_rom_builder import NumericalRomBuilder
from romshake.simulators import gmpe

sim = gmpe.GMPE_Simulator(add_noise=True)

bounds_names = ['depth', 'strike', 'length', 'dip']
bounds_vals = [(0, 60), (0, 360), (20, 40), (0, 90)]

rank = 1000
rbf = 'thin_plate_spline'

n = np.linspace(100, 1000, 10)
data = []

bounds_dict = {}
for name, val in zip(bounds_names, bounds_vals):
    bounds_dict[name] = val
    folder = 'test'
    if os.path.exists(folder):
        shutil.rmtree(folder)
    rb = NumericalRomBuilder(
        'test', sim, 100, 100, 1000, 'halton', bounds_dict, ranks=[rank],
        rbf_kernels=['thin_plate_spline'])
    data.append(rb.error_history[rank][rbf])

plt.figure(figsize=(7, 4))
for i, dataval in enumerate(data):
    plt.plot(n, dataval, '-o', label='%s parameters' % (i + 1))
plt.xlabel('Number of data snapshots')
plt.ylabel('Relative $l_2$ error')
plt.yscale('log')
plt.legend(loc='upper left', bbox_to_anchor=(0.72, 0.5))
plt.savefig('gmpe_dims.pdf')
plt.close('all')
