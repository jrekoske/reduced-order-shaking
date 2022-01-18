'''
Builds the reduced order models from SeisSol output using RBF interpolant.
'''


import os
import yaml
import h5py
import pickle
import numpy as np
import pandas as pd
import pod as podtools
import matplotlib.pyplot as plt

from mesh_plot import mesh_plot

with open('config.yml') as f:
    config = yaml.safe_load(f)


if not os.path.exists(config['rom_dir_rbf']):
    os.makedirs(config['rom_dir_rbf'])

df = pd.read_csv(os.path.join(
    config['source_files'], 'sims_params.csv'), index_col='sim_no')

# Split training and testing data (90% training, 10% testing)
df['dataset'] = 90 * ['train'] + 10 * ['test']
df_train = df[df.dataset == 'train'].copy()
df_test = df[df.dataset == 'test'].copy()

# Load in one mesh to find the mask
h5f = h5py.File(os.path.join(
    config['source_files'], '0', 'loh1-GME_corrected.h5'), 'r')
connect = h5f['mesh0']['connect']
geom = h5f['mesh0']['geometry']

# Only use elements in the area of interest
print('Creating mask...')
elem_mask = np.zeros(connect.shape[0], dtype=bool)
for i, element in enumerate(connect):
    cx, cy, cz = geom[element].mean(axis=0)
    if (cx >= config['xmin'] and cx <= config['xmax'] and
            cy >= config['ymin'] and cy <= config['ymax']):
        elem_mask[i] = 1.0

# Load in the snapshot data
print('Loading snapshots...')
snapshots = {}
controls = {key: [] for key in config['params']}
for i, row in df_train.iterrows():
    h5f = h5py.File(os.path.join(
        config['source_files'], str(i), 'loh1-GME_corrected.h5'), 'r')
    imts = list(h5f['mesh0'].keys())
    imts.remove('connect')
    imts.remove('geometry')

    for imt in imts:
        imt_arr = np.array(h5f['mesh0'][imt]).flatten()[elem_mask]
        if config['use_log']:
            imt_arr = np.log(imt_arr)
        if imt in snapshots:
            snapshots[imt].append(imt_arr)
        else:
            snapshots[imt] = [imt_arr]
    for param in config['params']:
        controls[param].append(row[param])

l2 = {key: [] for key in imts}
linf = {key: [] for key in imts}
pred_dict = {key: {} for key in df_test.index}

# create a reduced order model for every imt
for imt in imts:
    print('IMT: %s' % imt)
    pod = podtools.PODMultivariate(remove_mean=True)
    # pod = podtools.PODMultivariate(remove_mean=False)
    pod.database_append(controls, snapshots[imt])
    pod.setup_basis()
    pod.setup_interpolant(rbf_type='tps', bounds_auto=True)
    print('Singular values:\n', pod.singular_values)

    e = pod.get_ric()
    if config['verbose']:
        print('Relative Information Content (RIC):\n', e)

        # LOOCV measures
        measure = podtools.rbf_loocv(pod, norm_type="linf")
        measure = np.absolute(measure)

        ordering = np.argsort(measure)
        print('m[smallest][Linf] =', ('%1.4e' % measure[ordering[0]]))
        print('m[largest ][Linf] =', ('%1.4e' % measure[ordering[-1]]))
        print('measure:\n', measure)
        print('snapshot index min/max:', ordering[0], ordering[-1])

        measure = podtools.rbf_loocv(pod, norm_type="rms")
        measure = np.absolute(measure)

        ordering = np.argsort(measure)
        print('m[smallest][rms] =', ('%1.4e' % measure[ordering[0]]))
        print('m[largest ][rms] =', ('%1.4e' % measure[ordering[-1]]))
        print('measure:\n', measure)
        print('snapshot index min/max:', ordering[0], ordering[-1])

    # Make plot of singular values and RIC
    if imt in ['PGV', 'SA05.000s']:

        plt.figure(figsize=(7, 3.5))
        plt.subplot(1, 2, 1)
        plt.plot(pod.singular_values[:-1])
        plt.xlabel('Index')
        plt.ylabel('Singular values')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        plt.plot(e)
        plt.xlabel('Index')
        plt.ylabel('RIC')
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(
            config['rom_dir_rbf'], 'singular_values_%s.png' % imt), dpi=600)

    # Test against forward models
    for i, row in df_test.iterrows():
        h5f = h5py.File(os.path.join(
            config['source_files'], str(i), 'loh1-GME_corrected.h5'), 'r')
        true = np.array(h5f['mesh0'][imt]).flatten()[elem_mask]
        if config['use_log']:
            true = np.log(true)
        param_vec = [row[param] for param in config['params']]
        pred = pod.evaluate(param_vec)
        pred_dict[i][imt] = pred

        npoints = len(true)
        l2_error = np.linalg.norm(true - pred, 2) / np.sqrt(npoints)
        linf_error = np.linalg.norm(
            true - pred, np.Infinity) / np.sqrt(npoints)
        l2[imt].append(l2_error)
        linf[imt].append(linf_error)

        if imt == 'PGV':
            mesh_plot(
                np.array(geom), np.array(connect)[elem_mask], true, pred,
                os.path.join(config['rom_dir_rbf'], 'map-%s-%s.png' % (i, imt)),
                l2_error, linf_error, 'POD prediction (RBF interpolant)')

    # Save the model to a file in case we need in later
    pickle.dump(pod, open(os.path.join(
        config['rom_dir_rbf'], 'pod-save-%s.pkl' % imt), 'wb'))

for i, row in df_test.iterrows():
    sim_pred = pred_dict[i]
    i = str(i)

    # Push POD data to a new file
    pred_file = os.path.join(config['source_files'], i, 'pod_surface_cell.h5')
    h5f_new = h5py.File(pred_file, 'w')

    grp = h5f_new.create_group('mesh0/')
    for imt in sim_pred.keys():
        pred_arr = np.full(elem_mask.shape[0], np.nan)
        pred_arr[elem_mask] = sim_pred[imt]
        grp.create_dataset(imt, data=pred_arr)
    h5f_new.close()

    # Create new xdmf file
    old_xdmf = os.path.join(
        config['source_files'], i, 'loh1-GME_corrected.xdmf')
    new_xdmf = os.path.join(config['source_files'], i, 'pod-GME.xdmf')

    with open(old_xdmf, 'r') as file:
        filedata = file.read()

    for imt in sim_pred.keys():
        filedata = filedata.replace(
            'loh1-GME_corrected.h5:/mesh0/%s' % imt,
            'pod_surface_cell.h5:/mesh0/%s' % imt)
    with open(new_xdmf, 'w') as file:
        file.write(filedata)

df_l2 = pd.DataFrame(l2, index=df_test.index)
df_linf = pd.DataFrame(linf, index=df_test.index)

df_l2.to_csv(os.path.join(config['rom_dir_rbf'], 'l2_errors_tps.csv'))
df_linf.to_csv(os.path.join(config['rom_dir_rbf'], 'linf_errors_tps.csv'))
