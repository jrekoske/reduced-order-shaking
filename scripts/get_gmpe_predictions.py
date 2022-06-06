import numpy as np
from tqdm import tqdm
from gmpe_analytic import get_analytic

dep_min = 0
dep_max = 60  # km
strike_min = 0
strike_max = 360  # degrees

deps = np.linspace(dep_min, dep_max, 100)
strikes = np.linspace(strike_min, strike_max, 100)
dep_grid, strike_grid = np.meshgrid(deps, strikes)
grid = np.column_stack((dep_grid.ravel(), strike_grid.ravel()))

shakes = np.zeros((10000, 32*32))
for i in tqdm(range(10000)):
    dep = grid[i][0]
    strike = grid[i][1]
    shakes[i] = get_analytic(dep, strike)

np.save('shakes_32.npy', shakes)
