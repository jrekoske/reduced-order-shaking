import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from new_rom_builder import RomBuilder
from gmpe_analytic import get_analytic
plt.rcParams['font.size'] = 13

func = get_analytic
lower_bounds = [0, 0, 15, 0]
upper_bounds = [60, 360, 45, 90]
n_samps_initial = 3000
n_samps_refine = 100
n_samps_stop = 3000
n_truth = 10

hl_sizes = [[100, 100], [500, 500, 500]]
rank = 30

regrs = [
    KNeighborsRegressor(),
    DecisionTreeRegressor()]
ml_names = ['knn', 'dt']

for hl_size in hl_sizes:
    regrs.append(MLPRegressor(
        hidden_layer_sizes=hl_size, learning_rate='adaptive', max_iter=5000,
        verbose=True))
    ml_names.append('nn(%s)' % hl_size)

truth = np.load('dim4_truth.npy')

rb = RomBuilder(func, lower_bounds, upper_bounds,
                n_truth, n_samps_initial, n_samps_refine,
                n_samps_stop, ml_regressors=regrs, ml_names=ml_names,
                truth=truth, rank=rank)

Ppred = np.array([
    [0, 0, 30, 90],
    [60, 0, 30, 90],
    [0, 90, 30, 90],
    [0, 0, 15, 90],
    [0, 0, 30, 10]
])
pred = rb.predict(Ppred)

fig, axes = plt.subplots(nrows=len(Ppred), ncols=len(pred.keys()) + 1,
                         figsize=(14, 10))

for row_idx, predval in enumerate(Ppred):
    ax = axes[row_idx, 0]
    ax.imshow(get_analytic(*predval).reshape(50, 50))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    if row_idx == 0:
        ax.set_title('Truth')
    ax.set_ylabel('depth=%s, strike=%s \n length=%s, dip=%s' % tuple(predval))

for col_idx, key in enumerate(pred.keys()):
    for row_idx, plot_vals in enumerate(pred[key].T):
        ax = axes[row_idx, col_idx + 1]
        ax.imshow(plot_vals.reshape(50, 50))
        if row_idx == 0:
            ax.set_title(key)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

plt.tight_layout()
plt.savefig('./pres/figs/analytic_pgv_maps_3000.pdf')
