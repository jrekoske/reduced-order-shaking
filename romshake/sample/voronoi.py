import sys
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon

eps = sys.float_info.epsilon


def voronoi_sample(points, min_vals, max_vals, kf_errors, method,
                   n_samples_refine):

    print('method:', method)
    print('nsamples:', n_samples_refine)
    dim = points.shape[1]
    points_norm = (points - min_vals) / (max_vals - min_vals)
    vor = bounded_voronoi(points_norm)
    pvertices = vor.vertices[
        vor.filtered_regions[np.argmax(kf_errors)]]
    pvertices = pvertices[
        (pvertices.T[0] >= (0 - eps)) & (pvertices.T[0] <= (1 + eps))]
    pvertices = pvertices[
        (pvertices.T[1] >= (0 - eps)) & (pvertices.T[1] <= (1 + eps))]
    pvertices = np.vstack(
        [np.clip(pvertices.T[idim], 1e-3, 1-1e-3) for idim in range(dim)]).T

    if method == 'voronoi_vertex':
        chosen_points_norm = pvertices
    elif method == 'voronoi_edge_center':
        pvertices = np.vstack((pvertices, pvertices[0]))
        chosen_points_norm = np.vstack([np.convolve(
            pvertices.T[idim], np.ones(2) / 2, mode='valid')
            for idim in range(dim)]).T
    # TODO: fix problem with voronoi_random_walk (Shapely)
    # elif method == 'voronoi_random_walk':
    #     chosen_points_norm = []
    #     cell_polygon = Polygon(pvertices)
    #     while len(chosen_points_norm) < n_samples_refine:
    #         point = np.random.uniform(
    #             np.min(pvertices, axis=0),
    #             np.max(pvertices, axis=0))
    #         shp_point = Point(point)
    #         if cell_polygon.contains(shp_point):
    #             chosen_points_norm.append(point)
    #     chosen_points_norm = np.array(chosen_points_norm)

    points = chosen_points_norm * (max_vals - min_vals) + min_vals
    return points


def bounded_voronoi(points):

    all_points = np.copy(points)
    dimension = points.shape[1]
    for dim1 in range(dimension):
        p1 = np.copy(points)
        p2 = np.copy(points)
        p1[:, dim1] = -p1[:, dim1]
        p2[:, dim1] = 1 + (1 - p2[:, dim1])
        all_points = np.append(all_points, p1, axis=0)
        all_points = np.append(all_points, p2, axis=0)

    vor = Voronoi(all_points)
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                for dim in range(dimension):
                    coord = vor.vertices[index, dim]
                    if not(coord >= -eps and coord <= 1 + eps):
                        flag = False
                        break
        if region != [] and flag:
            regions.append(region)

    vor.filtered_points = points
    vor.filtered_regions = np.array(
        vor.regions)[vor.point_region[:vor.npoints//(2*dimension + 1)]]

    return vor


def plot_voronoi_diagram(vor, points, kf_errors, chosen_points_norm, fname):
    norm = mpl.colors.Normalize(vmin=np.min(
        kf_errors), vmax=np.max(kf_errors), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        plt.plot(vertices[:, 0], vertices[:, 1], 'k-')
    sc = plt.scatter(points.T[0], points.T[1], c=kf_errors, zorder=5)
    plt.scatter(
        chosen_points_norm.T[0], chosen_points_norm.T[1], c='r', zorder=6)
    plt.scatter(points.T[0], points.T[1], c='k', zorder=7)
    for r, region in enumerate(vor.filtered_regions):
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(kf_errors[r]))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Depth (normalized)')
    plt.ylabel('Strike (normalized)')
    plt.title('$N=%s$' % len(points))
    plt.colorbar(sc, label='Relative $L_2$ error')
    plt.savefig(fname)
    plt.close('all')
