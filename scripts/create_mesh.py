import os
import glob
import gmsh
import math
import yaml
import numpy as np
from pyproj import Transformer
from rasterio.merge import merge
from scipy.interpolate import RegularGridInterpolator

with open('config.yml') as f:
    config = yaml.safe_load(f)
mconfig = config['mesh']

# Load and merge topography
fnames = glob.glob(os.path.join('socal_topo_1_28', '*dem.tif'))
a = merge(fnames)

topo = a[0][0]
topo = np.flip(topo, axis=0)

# Get lons/lats associated with topography
x_lon, y_lat = a[1] * (np.arange(0, topo.shape[0]),
                       np.arange(0, topo.shape[1]))


# Conversion from lat/lon to mesh coordinate system (SW corner is 0, 0)
sProj = '+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=%s +lat_0=%s +axis=enu' % (
    mconfig['sw_lon'], mconfig['sw_lat'])
transformer_inv = Transformer.from_crs(sProj, 'epsg:4326', always_xy=True)
transformer = Transformer.from_crs('epsg:4326', sProj, always_xy=True)

xmin = 0
xmax = mconfig['outer_le']
ymin = 0
ymax = mconfig['outer_le']
zmax = -200e3

# Initialize Gmsh API
gmsh.initialize()
gmsh.model.add('Northridge')

# Resample original topographic data
x, y = transformer.transform(x_lon, y_lat)
N = math.ceil(max(np.ptp(x), np.ptp(y)) / (mconfig['inner_dx']))


def tag(i, j):
    # Helper function to return a node tag given two indices i and j
    return (N + 1) * i + j + 1


# The x, y, z coordinates of all the nodes:
coords = []

# The tags of the corresponding nodes:
nodes = []

# The connectivities of the triangle elements (3 node tags per triangle) on the
# terrain surface:
triangles_connect = []

# The connectivities of the line elements on the 4 boundaries (2 node tags
# for each line element):
lines_connect = [[], [], [], []]

# The connectivities of the point elements on the 4 corners (1 node tag for
# each point element):
points_connect = [tag(0, 0), tag(N, 0), tag(N, N), tag(0, N)]

# Adding topography point by point
x_grid = np.linspace(xmin, xmax, N+1)
y_grid = np.linspace(ymin, ymax, N+1)
xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

# Interpolate grid topography
ftopo = RegularGridInterpolator((x_lon, np.flip(y_lat)), topo.T)
lon_lat_grid = transformer_inv.transform(xx_grid.flatten(), yy_grid.flatten())
topo_grid = ftopo(lon_lat_grid).reshape(N+1, N+1)

for i in range(N + 1):
    for j in range(N + 1):
        nodes.append(tag(i, j))
        coords.extend([x_grid[i], y_grid[j], topo_grid[j, i]])
        if i > 0 and j > 0:
            triangles_connect.extend(
                [tag(i - 1, j - 1), tag(i, j - 1), tag(i - 1, j)])
            triangles_connect.extend([tag(i, j - 1), tag(i, j), tag(i - 1, j)])
        if (i == 0 or i == N) and j > 0:
            lines_connect[3 if i == 0 else 1].extend(
                [tag(i, j - 1), tag(i, j)])
        if (j == 0 or j == N) and i > 0:
            lines_connect[0 if j == 0 else 2].extend(
                [tag(i - 1, j), tag(i, j)])

# Create 4 discrete points for the 4 corners of the terrain surface:
for i in range(4):
    gmsh.model.addDiscreteEntity(0, i + 1)
gmsh.model.setCoordinates(1, xmin, ymin, coords[3 * tag(0, 0) - 1])
gmsh.model.setCoordinates(2, xmax, ymin, coords[3 * tag(N, 0) - 1])
gmsh.model.setCoordinates(3, xmax, ymax, coords[3 * tag(N, N) - 1])
gmsh.model.setCoordinates(4, xmin, ymax, coords[3 * tag(0, N) - 1])

# Create 4 discrete bounding curves, with their boundary points:
for i in range(4):
    gmsh.model.addDiscreteEntity(1, i + 1, [i + 1, i + 2 if i < 3 else 1])

# Create one discrete surface, with its bounding curves:
gmsh.model.addDiscreteEntity(2, 1, [1, 2, -3, -4])

# Add all the nodes on the surface:
gmsh.model.mesh.addNodes(2, 1, nodes, coords)
gmsh.model.addPhysicalGroup(2, [1], 101)  # Free-surface boundary label

# Add point elements on the 4 points, line elements on the 4 curves, and
# triangle elements on the surface:
for i in range(4):
    # Type 15 for point elements:
    gmsh.model.mesh.addElementsByType(i + 1, 15, [], [points_connect[i]])
    # Type 1 for 2-node line elements:
    gmsh.model.mesh.addElementsByType(i + 1, 1, [], lines_connect[i])
# Type 2 for 3-node triangle elements:
gmsh.model.mesh.addElementsByType(1, 2, [], triangles_connect)

# Reclassify the nodes on the curves and the points
gmsh.model.mesh.reclassifyNodes()

# Create a geometry for the discrete curves and surfaces, so that we can
# remesh them later on:
gmsh.model.mesh.createGeometry()

# Create other entities to form one volume below the terrain surface:
p1 = gmsh.model.geo.addPoint(xmin, ymin, zmax)
p2 = gmsh.model.geo.addPoint(xmax, ymin, zmax)
p3 = gmsh.model.geo.addPoint(xmax, ymax, zmax)
p4 = gmsh.model.geo.addPoint(xmin, ymax, zmax)
c1 = gmsh.model.geo.addLine(p1, p2)
c2 = gmsh.model.geo.addLine(p2, p3)
c3 = gmsh.model.geo.addLine(p3, p4)
c4 = gmsh.model.geo.addLine(p4, p1)
c10 = gmsh.model.geo.addLine(p1, 1)
c11 = gmsh.model.geo.addLine(p2, 2)
c12 = gmsh.model.geo.addLine(p3, 3)
c13 = gmsh.model.geo.addLine(p4, 4)
ll1 = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
s1 = gmsh.model.geo.addPlaneSurface([ll1])  # bot
ll3 = gmsh.model.geo.addCurveLoop([c1, c11, -1, -c10])  # fro
s3 = gmsh.model.geo.addPlaneSurface([ll3])  # fro
ll4 = gmsh.model.geo.addCurveLoop([c2, c12, -2, -c11])
s4 = gmsh.model.geo.addPlaneSurface([ll4])  # rig
ll5 = gmsh.model.geo.addCurveLoop([c3, c13, 3, -c12])
s5 = gmsh.model.geo.addPlaneSurface([ll5])  # bac
ll6 = gmsh.model.geo.addCurveLoop([c4, c10, 4, -c13])
s6 = gmsh.model.geo.addPlaneSurface([ll6])  # lef
# Absorbing boundary label
gmsh.model.addPhysicalGroup(2, [s1, s3, s4, s5, s6], 105)
sl1 = gmsh.model.geo.addSurfaceLoop([s1, s3, s4, s5, s6, 1])
v1 = gmsh.model.geo.addVolume([sl1])
gmsh.model.addPhysicalGroup(3, [v1], 1)
gmsh.model.geo.synchronize()

# setup key mesh parameters
gmsh.model.mesh.field.add('Distance', 1)
gmsh.model.mesh.field.setNumbers(1, 'EdgesList', [1, 2, 3, 4])
gmsh.model.mesh.field.setNumbers(1, 'FacesList', [1])

gmsh.model.mesh.field.add('Box', 2)
gmsh.model.mesh.field.setNumber(2, 'VIn', mconfig['inner_dx'])
gmsh.model.mesh.field.setNumber(2, 'VOut', mconfig['outer_dx'])
gmsh.model.mesh.field.setNumber(
    2, 'XMin', (mconfig['outer_le'] - mconfig['inner_le']) / 2)
gmsh.model.mesh.field.setNumber(
    2, 'XMax', (mconfig['outer_le'] + mconfig['inner_le']) / 2)
gmsh.model.mesh.field.setNumber(
    2, 'YMin', (mconfig['outer_le'] - mconfig['inner_le']) / 2)
gmsh.model.mesh.field.setNumber(
    2, 'YMax', (mconfig['outer_le'] + mconfig['inner_le']) / 2)
gmsh.model.mesh.field.setNumber(2, 'ZMin', -mconfig['inner_lz'])
gmsh.model.mesh.field.setNumber(2, 'ZMax', mconfig['inner_lz'])
gmsh.model.mesh.field.setNumber(2, 'Thickness', mconfig['thickness'])

# Use the minimum of all the fields as the background mesh field
gmsh.model.mesh.field.add('Min', 5)
gmsh.model.mesh.field.setNumbers(5, 'FieldsList', [2])
gmsh.model.mesh.field.setAsBackgroundMesh(5)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write('mesh_socal_topo.msh2')  # type 2 Gmsh file

gmsh.finalize()
