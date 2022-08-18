#!/bin/bash
# Name of virtual environment
VENV=romshake
eval "$(conda shell.bash hook)"
conda activate base

# Remove virtual environmnent if it exists
conda remove -y --name $VENV --all

# Install mamba
conda install -c conda-forge mamba -y

# Create environment and install conda packages
if  [[ $1 = "-gpu" ]]; then
    mamba create -n $VENV -y --file=conda_requirements.txt --file=gpu_requirements.txt -c conda-forge -c nvidia -c rapidsai -vvv
else
    mamba create -n $VENV -y --file conda_requirements.txt -c conda-forge
fi

# End if conda create command fails.
if [ $? -ne 0 ]; then
    echo "Failed to create conda environment. Resolve any conflicts, then try again."
    exit 1
fi

# Activate the new environment
echo "Activating the $VENV virtual environment"
conda activate $VENV

# End if conda activate fails
if [ $? -ne 0 ];then
    echo "Failed to activate ${VENV} conda environment. Exiting."
    exit 1
fi

# Activate virtual environment
mamba activate $VENV

# Install pip packages
pip install -r pip_requirements.txt

# Configure the system paths (for tensorflow)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Instal this repository
pip install --no-deps -e .

# End if pip install fails
if [ $? -ne 0 ];then
    echo "Failed to pip install this package. Exiting."
    exit 1
fi
