import os
import yaml
import pickle
import numpy as np
from romshake.core.reduced_order_model import ReducedOrderModel
from romshake.simulators.seissol_simulate import SeisSolSimulator

with open('config.yml', 'r') as f:
    sim = SeisSolSimulator(**yaml.safe_load(f))
with open('rom_config.yml', 'r') as f:
    rom = ReducedOrderModel(**yaml.safe_load(f))
rom.remote_grid_search = False  # Make sure this is set to False, already
folder = rom.folder
indices = os.path.listdir(os.path.join(folder), 'data')
successful_indices = sim.get_successful_indices(folder, indices)
sim.reorder_elements(folder, successful_indices)
data = sim.load_data(folder, successful_indices)
params = np.load('X.npy')
rom.update(params, data)
with open('search_results', 'wb') as outp:
    pickle.dump(rom.search, outp)

# Note: need to save Xtest, ytest, ypred if we want to make prediction plots
# in addition to saving the search object
