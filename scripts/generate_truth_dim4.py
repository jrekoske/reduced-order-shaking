import itertools
import numpy as np
from tqdm import tqdm
from gmpe_analytic import get_analytic

func = get_analytic
lower_bounds = [0, 0, 15, 0]
upper_bounds = [60, 360, 45, 90]
n_truth = 10

Ptruth = np.linspace(lower_bounds, upper_bounds, n_truth)
Ptruth_stacked = np.array(list(itertools.product(*Ptruth.T)))

evals = []
for param in tqdm(Ptruth_stacked):
    evals.append(get_analytic(*param))
truth = np.array(evals).T
np.save('dim4_truth', truth)
