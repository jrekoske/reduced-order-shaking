import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from neighborhood.search import Searcher


def voronoi_sample(
        points, min_vals, max_vals, errors, n_samples_refine, n_cells_refine,
        folder):
    logging.info('Using Voronoi sampling.')
    limits = [(minval, maxval) for minval, maxval in zip(min_vals, max_vals)]
    search = Searcher(
        objective=lambda: None, limits=limits, num_samp=n_samples_refine,
        num_resamp=n_cells_refine)
    search._sample = [{'param': param, 'result': error}
                      for param, error in zip(points, errors)]
    search._sample.sort(key=lambda x: x['result'], reverse=True)
    search._neighborhood_sample()
    newX = np.array(search._queue)
    size = 3 * points.shape[1]
    try:
        search.plot(
            size=(size, size), filename=os.path.join(
                folder, 'voronoi_%s.png' % points.shape[0]))
        plt.close('all')
    except:
        pass
    return newX
