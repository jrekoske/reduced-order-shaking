import numpy as np


def halton(b):
    '''
    Generator function for Halton sequence.
    '''
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d


def get_halton(min, max, a, n):
    halton_a = halton(a)
    return min + (max - min) * np.array([next(halton_a) for i in range(n)])
