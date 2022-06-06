import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from gmpe_analytic import get_analytic
from scipy.stats import qmc
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Bounds
l_bound = [0, 0, 15]
u_bound = [60, 360, 45]

ntruth = 100
nsamples = 5000

dim = len(l_bound)

# Generate test grid
grid = np.linspace(l_bound, u_bound, ntruth)
truth = np.array([get_analytic(*param, add_noise=False) for param in grid]).T

# Halton samples for depth
sampler = qmc.Halton(d=dim, seed=0)
P = qmc.scale(sampler.random(n=nsamples), l_bound, u_bound)
data = np.array([get_analytic(*param, add_noise=False) for param in P]).T
u, s1, vh = np.linalg.svd(data)

rank = 5000
u_rank = u[:, 0:rank]

plt.plot(s1)
plt.axvline(rank, c='k', ls='--')
plt.yscale('log')
plt.show()

# Calculate and normalize the POD coefficients
A = data.T @ u_rank
Ascaler = preprocessing.StandardScaler().fit(A)
Pscaler = preprocessing.StandardScaler().fit(P)

A_scaled = Ascaler.transform(A)
P_scaled = Pscaler.transform(P)

# Pick the length scale based on the spacing
mins = np.array([
    np.diff(np.sort(P_scaled.T[i])).min() for i in range(P_scaled.shape[1])])
maxs = np.array([
    np.diff(np.sort(P_scaled.T[i])).max() for i in range(P_scaled.shape[1])])

regr = GaussianProcessRegressor(
    kernel=RBF(
        length_scale=1e-3,
        length_scale_bounds=[1e-7, 1e-2]))
# regr = DecisionTreeRegressor()
# regr = KNeighborsRegressor()

regr.fit(P_scaled, A_scaled)

# Make prediction
Apred = Ascaler.inverse_transform(regr.predict(Pscaler.transform(grid)))
pred = u_rank @ Apred.T

error = np.linalg.norm(pred - truth) / np.linalg.norm(truth)
print('Error: %.3g' % error)

# Make some predictions
p1 = u_rank @ Ascaler.inverse_transform(
    regr.predict(Pscaler.transform([[0, 0, 15]]))).T
plt.imshow(p1.reshape(50, 50))
plt.show()

# Make some predictions
p1 = u_rank @ Ascaler.inverse_transform(
    regr.predict(Pscaler.transform([[0, 90, 45]]))).T
plt.imshow(p1.reshape(50, 50))
plt.show()

# Make some predictions
p1 = u_rank @ Ascaler.inverse_transform(
    regr.predict(Pscaler.transform([[45, 0, 30]]))).T
plt.imshow(p1.reshape(50, 50))
plt.show()


# TODO: work on understanding the enforcing the rank (and if it matters)
# and the length-scale of Gaussian Process Regression