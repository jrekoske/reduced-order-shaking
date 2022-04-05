import numpy as np
import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.tree import DecisionTreeRegressor
from new_rom_builder import RomBuilder
from gmpe_analytic import get_analytic
plt.rcParams['font.size'] = 13

func = get_analytic
lower_bounds = [0, 0, 15, 0]
upper_bounds = [60, 360, 45, 90]
n_samps_initial = 100
n_samps_refine = 500
n_samps_stop = 1000
n_truth = 10

# hl_sizes = [[20], [100], [500], [100, 100], [200, 200], [500, 500, 500]]

ranks = [5, 30, 50, 100]

# regrs = [
#     KNeighborsRegressor(),
#     DecisionTreeRegressor()]

rbf_kernels = ['thin_plate_spline', 'cubic', 'quintic']

# ml_names = ['knn', 'dt']

# for hl_size in hl_sizes:
#     regrs.append(MLPRegressor(
#         hidden_layer_sizes=hl_size, learning_rate='adaptive', max_iter=5000,
#         verbose=True))
#     ml_names.append('nn(%s)' % hl_size)

truth = np.load('dim4_truth.npy')

for rank in ranks:
    # rb = RomBuilder(func, lower_bounds, upper_bounds,
    #                 n_truth, n_samps_initial, n_samps_refine,
    #                 n_samps_stop, ml_regressors=regrs, ml_names=ml_names,
    #                 truth=truth, rank=rank)
    rb = RomBuilder(func, lower_bounds, upper_bounds,
                    n_truth, n_samps_initial, n_samps_refine,
                    n_samps_stop, rbf_kernels=rbf_kernels,
                    truth=truth, rank=rank)

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    for key, errors in rb.error_history.items():
        ax1.plot(rb.nsamples_list, errors, '-o', label=key)
    ax1.set_xlabel('n')
    ax1.set_ylabel('error')
    ax1.set_yscale('log')
    ax1.set_title('Rank: %s' % rank)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig1.savefig('pres/figs/errors_interps_rbf_%s.pdf' % rank)
