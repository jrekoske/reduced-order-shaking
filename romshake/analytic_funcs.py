import numpy as np


def gauss(t):
    x = np.linspace(-5, 5, 100)
    tmax = 1.0
    kappa = 1.0
    sigma = 1.0
    return tmax/np.sqrt(1+4*t*kappa/sigma**2)*np.exp(
        -x**2/(sigma**2+4*t*kappa))
