import os
import yaml
import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from mesh_plot import triplot


def get_model(n_inputs, n_outputs):
    # Function for creating keras NN model
    model = Sequential()
    model.add(Dense(20, input_dim=n_inputs,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')
    return model


def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=1000)
        # evaluate model on test set
        mae = model.evaluate(X_test, y_test, verbose=0)
        # store result
        print('>%.3f' % mae)
        results.append(mae)
    return results


# load config
with open('config.yml') as f:
    config = yaml.safe_load(f)

if not os.path.exists(config['rom_dir_ml']):
    os.makedirs(config['rom_dir_ml'])

df = pd.read_csv(os.path.join(
    config['source_dir'], 'sims_params.csv'), index_col='sim_no')

# Split training and testing
df['dataset'] = 90 * ['train'] + 10 * ['test']
df_train = df[df.dataset == 'train'].copy()
df_test = df[df.dataset == 'test'].copy()

# Load in one mesh to find the mask
h5f = h5py.File(os.path.join(
    config['source_dir'], '0', 'loh1-GME_corrected.h5'), 'r')
connect = h5f['mesh0']['connect']
geom = h5f['mesh0']['geometry']
imts = list(h5f['mesh0'].keys())
imts.remove('connect')
imts.remove('geometry')

# Only use elements in the area of interest
print('Creating mask...')
# elem_mask = np.zeros(connect.shape[0], dtype=bool)
# for i, element in enumerate(connect):
#     cx, cy, cz = geom[element].mean(axis=0)
#     if (cx >= xmin and cx <= xmax and cy >= ymin and cy <= ymax):
#         elem_mask[i] = 1.0
elem_mask = np.load('mask.npy')

# Only consider PGV for now
imts = ['PGV']

fig, axes = plt.subplots(nrows=len(config['plot_sims']),
                         ncols=len(config['alg_list'])+2, figsize=(10, 7))

for imt in imts:
    print(imt)
    with open(os.path.join(config['rom_dir_rbf'], 'pod-save-%s.pkl' % imt), 'rb') as pkl:
        rom = pickle.load(pkl)

    P = np.array(list(rom.control.values())).T
    A = rom.coeff.T

    # Code for evaluating NN model accuracy using K-fold cross validation
    # results = evaluate_model(P, A)
    # print('MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))

    n_inputs, n_outputs = P.shape[1], A.shape[1]

    for j, alg in enumerate(config['alg_list']):
        print('running %s' % alg)
        if alg == 'NN':
            model = get_model(n_inputs, n_outputs)
            model.fit(P, A, epochs=100)
        elif alg == 'KNN':
            model = KNeighborsRegressor()
            model.fit(P, A)
        elif alg == 'DT':
            model = DecisionTreeRegressor()
            model.fit(P, A)

        l2 = {key: [] for key in imts}
        linf = {key: [] for key in imts}

        # Evaluate against independent test dataset
        for sim_no, row in df_test.iterrows():
            h5f = h5py.File(os.path.join(
                config['source_dir'], str(sim_no), 'loh1-GME_corrected.h5'), 'r')
            true = np.array(h5f['mesh0'][imt]).flatten()[elem_mask]
            if config['use_log']:
                true = np.log(true)

            param_vec = [row[param] for param in config['params']]
            alpha_pred = model.predict([param_vec]).flatten()
            ml_pred = rom.phi_normalized @ alpha_pred + rom.Umean

            npoints = len(true)
            l2_error = np.linalg.norm(true - ml_pred, 2) / np.sqrt(npoints)
            linf_error = np.linalg.norm(
                true - ml_pred, np.Infinity) / np.sqrt(npoints)
            l2[imt].append(l2_error)
            linf[imt].append(linf_error)

            # if imt == 'PGV' and int(sim_no) in config['plot_sims']:
            if int(sim_no) in config['plot_sims']:

                i = config['plot_sims'].index(sim_no)
                ax = axes[i, j+2]

                nodes = np.array(geom)
                x, y = nodes[:, 0], nodes[:, 1]

                im = triplot(x, y, np.array(connect)[
                    elem_mask], ml_pred, ax, edgecolor='face', vmin=config['vmin'], vmax=config['vmax'])
                for side_str in ['left', 'right', 'top', 'bottom']:
                    side = ax.spines[side_str]
                    side.set_visible(False)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(alg)
                    # if j == 2:
                    #     fig.colorbar(im, ax=ax, label='log PGV (m/s)')

                if j == 0:
                    ax = axes[i, j]
                    triplot(x, y, np.array(connect)[
                            elem_mask], true, ax, edgecolor='face', vmin=config['vmin'], vmax=config['vmax'])
                    for side_str in ['left', 'right', 'top', 'bottom']:
                        side = ax.spines[side_str]
                        side.set_visible(False)
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title('Forward model')
                    ax.set_ylabel('Depth: %.0f km, Strike: %.0f\nRake: %.0f, Dip: %.0f' % (
                        param_vec[0], param_vec[1], param_vec[2], param_vec[3]))

                    ax = axes[i, j+1]
                    pred = rom.evaluate(param_vec)
                    triplot(x, y, np.array(connect)[
                        elem_mask], pred, ax, edgecolor='face', vmin=config['vmin'], vmax=config['vmax'])
                    for side_str in ['left', 'right', 'top', 'bottom']:
                        side = ax.spines[side_str]
                        side.set_visible(False)
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title('RBF')

                    # mesh_plot(
                    #     np.array(geom), np.array(connect)[
                    #         elem_mask], true, ml_pred,
                    #     os.path.join(config['rom_dir_ml'], 'map-%s-%s-%s.png' %
                    #                  (sim_no, imt, alg)),
                    #     l2_error, linf_error, 'POD prediction (%s interpolant)' % alg)

        df_l2 = pd.DataFrame(l2, index=df_test.index)
        df_linf = pd.DataFrame(linf, index=df_test.index)

        df_l2.to_csv(os.path.join(
            config['rom_dir_ml'], 'l2_errors_%s.csv' % alg))
        df_linf.to_csv(os.path.join(
            config['rom_dir_ml'], 'linf_errors_%s.csv' % alg))


plt.tight_layout()
print('saving figure...')
plt.savefig('ml-results-%s.png' % imt, dpi=400)
plt.close('all')
