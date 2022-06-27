#!/bin/bash
VENV=romshake
eval "$(conda shell.bash hook)"
conda activate base
conda remove -y --name $VENV --all
mamba create -n $VENV -y --file conda_requirements.txt -c conda-forge -vvv
conda activate $VENV
pip install -r pip_requirements.txt
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install --no-deps -e .
