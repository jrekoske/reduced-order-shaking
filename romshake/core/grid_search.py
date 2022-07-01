import os
import yaml
import argparse
import numpy as np
import pandas as pd
from romshake.core.reduced_order_model import ReducedOrderModel
from romshake.simulators.seissol_simulate import SeisSolSimulator

parser = argparse.ArgumentParser()
parser.add_argument('folder')
parser.add_argument('params_dict')
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

# Train the models
rom.update(params, data)

# Save the results
np.save('X', params)
pd.DataFrame(rom.search.cv_results_).to_csv('grid_search_results.csv')
