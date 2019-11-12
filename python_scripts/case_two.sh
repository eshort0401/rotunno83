#!/bin/bash
#PBS -q expressbw
#PBS -P w40
#PBS -l ncpus=16
#PBS -l mem=128GB
#PBS -l jobfs=1GB
#PBS -l walltime=24:00:00
#PBS -l wd

# Run python program

# Load default modules
module use /g/data3/hh5/public/modules
module load conda

unset CONDA_PKGS_DIRS
unset CONDA_ENVS_PATH

conda activate analysis3-19.07

python3 ./rotunnoScript.py
