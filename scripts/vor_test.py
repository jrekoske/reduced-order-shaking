import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
eps = sys.float_info.epsilon

points = np.array([[0.1, 0.1],
                   [0.2, 0.8],
                   [0.9, 0.5]])

left = np.copy(points)
left[:, 0] *= -1

right = np.copy(left)
right[:, 0] += 2

bottom = np.copy(points)
bottom[:, 1] *= -1

top = np.copy(bottom)
top[:, 1] += 2

all_points = np.vstack((points, left, right, top, bottom))
vor = Voronoi(all_points)

regions = []
for region in vor.regions:   
    flag = True
    for index in region:
        if index == -1:
            flag = False
            break
        else:
            x = vor.vertices[index, 0]
            y = vor.vertices[index, 1]
            if not(0 - eps <= x and x <= 1 + eps and
                0 - eps <= y and y <= 1 + eps):
                flag = False
                break
    if region and flag:
        regions.append(region)
vor.filtered_points = points
vor.filtered_regions = np.array(vor.regions)[vor.point_region[:vor.npoints//5]]

voronoi_plot_2d(vor)
plt.scatter(all_points.T[0], all_points.T[1])
plt.savefig('vor_test.png')