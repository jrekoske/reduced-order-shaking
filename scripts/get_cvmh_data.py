import numpy as np
import subprocess
from netCDF4 import Dataset


def write_cvhm_input(x, y, z, name):
    print('Writing CVM-H infile for %s' % name)
    with open('infile-%s' % name, 'w') as infile:
        for xval in x:
            for yval in y:
                for zval in z:
                    infile.write('%s %s %s\n' % (xval, yval, zval))


def write_netcdf_file(x, y, z, name):

    vp, vs, rho = np.genfromtxt('outfile-%s' % name, usecols=[-3, -2, -1]).T
    
    # Enforce minimum velocities
    np.clip(vp, 2984.0, None, out=vp)
    np.clip(vs, 1400.0, None, out=vs)
    np.clip(rho, 2220.34, None, out=rho)

    # convert wavespeeds to Lame parameters
    mu = vs**2 * rho
    lam = vp**2 * rho - 2*mu

    print('Writing NetCDF file for %s' % name)
    nc = Dataset('rhomulambda-%s.nc' % name, 'w', format='NETCDF4')
    nc.createDimension('x', len(x))
    nc.createDimension('y', len(y))
    nc.createDimension('z', len(z))

    vx = nc.createVariable('x', 'f4', ('x',))
    vx[:] = x
    vy = nc.createVariable('y', 'f4', ('y',))
    vy[:] = y
    vz = nc.createVariable('z', 'f4', ('z',))
    vz[:] = z

    mattype4 = np.dtype([('rho', 'f4'), ('mu', 'f4'), ('lambda', 'f4')])
    mattype8 = np.dtype([('rho', 'f8'), ('mu', 'f8'), ('lambda', 'f8')])
    mattype = nc.createCompoundType(mattype4, 'material')

    # transform to an array of tuples
    arr = np.stack((rho, mu, lam), axis=1)
    arr = arr.view(dtype=mattype8)
    mat = nc.createVariable('data', mattype, ('z', 'y', 'x'))
    mat[:] = np.reshape(arr, mat.shape)
    nc.close()


# TODO: read these values from the config

# lower left coordinates (of outer region)
easting = 287960
northing = 3525948

# Units (km)
side_outer = 250e3
zmax_outer = 200e3

side_inner = 70e3
zmax_inner = 60e3

spacing_outer = 10e3
spacing_inner = 1e3

# highest elevation
elev_max = 4e3

# Spacing between inner and outer zones
d = ((side_outer - side_inner) / 2)

x_i = np.arange(easting + d, easting + d + side_inner +
                spacing_inner, spacing_inner)
y_i = np.arange(northing + d, northing + d +
                side_inner + spacing_inner, spacing_inner)
z_i = np.arange(-zmax_inner, elev_max + spacing_inner, spacing_inner)

x_o = np.arange(easting, easting + side_outer + spacing_outer, spacing_outer)
y_o = np.arange(northing, northing + side_outer + spacing_outer, spacing_outer)
z_o = np.arange(-zmax_outer, elev_max + spacing_outer, spacing_outer)

write_cvhm_input(x_i, y_i, z_i, 'inner')
write_cvhm_input(x_o, y_o, z_o, 'outer')

print('Running CVM-H vx_lite')
subprocess.run(['./run_cvhm.sh'])

# Write netcdf using mesh coordinate system
x_o -= easting
y_o -= northing

x_i -= easting
y_i -= northing

write_netcdf_file(x_i, y_i, z_i, 'inner')
write_netcdf_file(x_o, y_o, z_o, 'outer')
