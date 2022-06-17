#!/bin/bash
VENV=romshake
eval "$(conda shell.bash hook)"
conda activate base
conda remove -y --name $VENV --all
mamba create -n $VENV -y --file conda_requirements.txt -c conda-forge -vvv
conda activate $VENV
pip install -r pip_requirements.txt
pip install --no-deps -e .
