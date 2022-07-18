import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def triplot(x, y, triangles, values, ax, vmin=None, vmax=None, **kwargs):
    xy = np.c_[x, y]
    verts = xy[triangles]
    pc = matplotlib.collections.PolyCollection(verts, **kwargs)
    pc.set_array(values)
    if vmin and vmax:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.autoscale()
    ax.axis('equal')
    return pc


def mesh_plot(
        nodes, elements, true_values, pred_values, fname, l2, linf, title2):
    x = nodes[:, 0]
    y = nodes[:, 1]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    c = [true_values, pred_values, true_values - pred_values]
    titles = ['Forward model', title2,
              'Error (L2=%.2f, Linf=%.2f)' % (l2, linf)]
    vmin = min(true_values.min(), pred_values.min())
    vmax = max(true_values.max(), pred_values.max())
    for i, ax in enumerate(axes):
        if i == 2:
            vmin = None
            vmax = None
        pc = triplot(x, y, np.asarray(elements),
                     c[i], ax, vmin, vmax, edgecolor='face')
        fig.colorbar(pc, ax=ax)
        ax.axis('off')
        ax.set_title(titles[i])
    plt.savefig(fname)
    plt.close('all')
