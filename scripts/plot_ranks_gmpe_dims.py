from scipy.stats import qmc
from gmpe_analytic import get_analytic
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 13

# Bounds for depth, strike, length, dip, width
l_bound = [0, 0, 15, 0, 15]
u_bound = [60, 360, 45, 90, 45]

nsamples = 300

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)

for i, add_noise in enumerate([True, False]):
    ax = axes[i]
    for dim in range(1, 6):
        sampler = qmc.Halton(d=dim, seed=0)
        P = qmc.scale(sampler.random(n=nsamples), l_bound[:dim], u_bound[:dim])
        data = np.array([get_analytic(*param, add_noise=add_noise)
                         for param in P]).T
        u, s, vh = np.linalg.svd(data)
        ax.plot(s, label=dim)
    ax.set_xlabel('index')
    ax.set_ylabel('singular value')
    ax.set_yscale('log')
    ax.legend()
    ax.set_ylim(bottom=1e-6)
    if add_noise:
        ax.set_title('With noise')
    else:
        ax.set_title('No noise')
plt.tight_layout()
plt.savefig('pres/figs/rank_structures_gmpe.pdf')
plt.close()
