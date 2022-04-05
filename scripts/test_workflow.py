from sklearn.neighbors import KNeighborsRegressor
import seissol_simulate
from numerical_rom_builder import NumericalRomBuilder

folder = 'workflow_test'
bounds = {
    'depth': (0, 60),
    'strike': (0, 360),
    'dip': (0, 90),
    'rake': (0, 180)}

rb = NumericalRomBuilder(
    folder=folder,
    forward_model_mod=seissol_simulate,
    n_seeds_initial=15,
    n_seeds_refine=3,
    n_seeds_stop=30,
    samp_method='halton',
    bounds=bounds,
    rbf_kernels=['thin_plate_spline'],
    ml_regressors=[KNeighborsRegressor()],
    ml_names=['knn'],
    k_val=2,
    rank=50)
