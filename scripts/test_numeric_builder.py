import os
import h5py
import numpy as np
import seissol_simulate
from mesh_plot import triplot
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from numerical_rom_builder import NumericalRomBuilder


folder = '/Users/jrekoske/Documents/topo/generic-11-27/source_files'
bounds = {
    'depth': (0, 60),
    'strike': (0, 360),
    'dip': (0, 90),
    'rake': (0, 180)}

ml_regressors = [
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    MLPRegressor(
        hidden_layer_sizes=[100, 100], learning_rate='adaptive', max_iter=5000,
        verbose=True)]

rb = NumericalRomBuilder(
    folder=folder,
    forward_model_mod=seissol_simulate,
    n_seeds_initial=10,
    n_seeds_refine=10,
    n_seeds_stop=100,
    samp_method='voronoi_vertex',
    bounds=bounds,
    rbf_kernels=['thin_plate_spline'],
    k_val=2,
    rank=50,
    ml_regressors=ml_regressors,
    ml_names=['knn', 'dt', 'nn'])

# Load in geometry for plotting
h5f = h5py.File(os.path.join(folder, 'data_0', 'loh1-GME_corrected.h5'), 'r')
connect = h5f['mesh0']['connect']
geom = h5f['mesh0']['geometry']
elem_mask = np.load('/Users/jrekoske/Documents/rom/mask.npy')
nodes = np.array(geom)
x, y = nodes[:, 0], nodes[:, 1]

Ppred = np.array([
    [0, 0, 30, 90],
    [60, 0, 30, 90],
    [0, 90, 30, 90],
    [0, 0, 15, 90],
    [0, 0, 30, 10]
])
pred_dict = rb.rom.predict(Ppred)
interp_names = list(pred_dict.keys())

fig, axes = plt.subplots(nrows=Ppred.shape[0], ncols=len(interp_names),
                         figsize=(6, 8))
for i, interp_name in enumerate(interp_names):
    pred_interp = pred_dict[interp_name]
    for j, pred in enumerate(pred_interp.T):
        ax = axes[j, i]
        triplot(
            x, y, np.array(connect)[elem_mask], pred, ax, edgecolor='face')

        for side_str in ['left', 'right', 'top', 'bottom']:
            side = ax.spines[side_str]
            side.set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        if j == 0:
            ax.set_title(interp_name)
        if i == 0:
            ax.set_ylabel(
                'depth=%s, strike=%s \n length=%s, dip=%s' % tuple(Ppred[j]))

plt.tight_layout()
plt.savefig('numerical_maps_100.pdf')
plt.close('all')
