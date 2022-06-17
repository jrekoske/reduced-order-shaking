#!/bin/bash
VENV=romshake
package_list=(
      "cartopy"
      "gmsh"
      "h5py"
      "joblib"
      "matplotlib"
      "netcdf4"
      "numpy"
      "openquake.engine"
      "pandas"
      "paramiko"
      "pyproj"
      "pyyaml"
      "rasterio"
      "scikit_learn"
      "scipy"
      "setuptools"
      "tqdm"
      "trimesh"
)
eval "$(micromamba shell hook --shell=bash)"
micromamba activate base
micromamba remove -y --name $VENV --all
micromamba create -y -vvv --name $VENV ${package_list[*]}
micromamba activate $VENV
pip install seissolxdmf
pip install seissolxdmfwriter
pip install --no-deps -e .
