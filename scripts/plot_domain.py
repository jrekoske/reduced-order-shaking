import glob
import yaml
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from rasterio.merge import merge
import cartopy.feature as cfeature
from pyproj import Transformer


with open('config.yml') as f:
    config = yaml.safe_load(f)


ll_utm_e = 0
ll_utm_n = 0
ur_utm_e = config['mesh']['outer_le']
ur_utm_n = config['mesh']['outer_le']

sProj = '+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=%s +lat_0=%s +axis=enu' % (
    config['mesh']['sw_lon'], config['mesh']['sw_lat'])
transformer_inv = Transformer.from_crs(sProj, 'epsg:4326', always_xy=True)
transformer = Transformer.from_crs('epsg:4326', sProj, always_xy=True)

ll_lon, ll_lat = transformer_inv.transform(ll_utm_e, ll_utm_n)
ur_lon, ur_lat = transformer_inv.transform(ur_utm_e, ur_utm_n)


# Load and merge topography
fnames = glob.glob('socal_topo_1_28/*dem.tif')
a = merge(fnames)

topo = a[0][0]
x_lon, y_lat = a[1] * (np.arange(0, topo.shape[0]),
                       np.arange(0, topo.shape[1]))

# apply downsampling
topo = topo[::10, ::10]
x_lon = x_lon[::10]
y_lat = y_lat[::10]

X, Y = np.meshgrid(x_lon, y_lat)

la_x, la_y = (-118.2437, 34.0522)

ax = plt.axes(projection=ccrs.PlateCarree())
im = ax.pcolormesh(X, Y, topo, cmap='terrain', shading='auto')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.text(la_x, la_y, 'Los Angeles')
ax.set_xlim(ll_lon, ur_lon)
ax.set_ylim(ll_lat, ur_lat)

xmin, xmax, ymin, ymax = -118.6328, -117.8649, 33.7286, 34.3516

dlon = ur_lon - ll_lon
dlat = ur_lat - ll_lat

ax.axhline(ymin, (xmin-ll_lon)/dlon, (xmax-ll_lon)/dlon, c='r')
ax.axhline(ymax, (xmin-ll_lon)/dlon, (xmax-ll_lon)/dlon, c='r')
ax.axvline(xmin, (ymin-ll_lat)/dlat, (ymax-ll_lat)/dlat, c='r')
ax.axvline(xmax, (ymin-ll_lat)/dlat, (ymax-ll_lat)/dlat, c='r')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='k', alpha=0.2, linestyle='--')
gl.top_labels = False
gl.right_labels = False

plt.colorbar(im, label='Elevation (m)')
plt.savefig('topo_domain.png', dpi=300)
plt.close('all')
