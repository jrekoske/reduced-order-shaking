import os
import yaml
import argparse
import numpy as np
import pandas as pd
from romshake.simulators.seissol_simulate import SeisSolSimulator
from romshake.core.reduced_order_model import ReducedOrderModel

parser = argparse.ArgumentParser()
parser.add_argument('folder')
parser.add_argument('params_dict')
parser.add_argument('hyperparameters')
parser.add_argument('test_size')
parser.add_argument('scoring')
args = parser.parse_args()

indices = os.path.listdir(os.path.join(args.folder), 'data')

with open('config.yml', 'r') as f:
    sim = SeisSolSimulator(**yaml.safe_load(f))
with open('rom_config.yml', 'r') as f:
    rom = ReducedOrderModel(**yaml.safe_load(f))

successful_indices = sim.get_successful_indices(args.folder, indices)
sim.reorder_elements(args.folder, successful_indices)
data = sim.load_data(args.folder, successful_indices)

params_arr = np.array(list(args.params_dict.values())).T
params = np.array(
    [param for param, idx in zip(params_arr, indices)
        if idx in successful_indices])

search = rom.train_search_models(
    params, data, args.test_size, args.hyperparameters, args.scoring)

# Save the results
np.save('X', params)
pd.DataFrame(search.cv_results_).to_csv('grid_search_results.csv')
