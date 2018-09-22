#!/bin/bash
#PBS -q normalbw 
#PBS -P w40
#PBS -l ncpus=2240
#PBS -l mem=5120GB
#PBS -l jobfs=200MB
#PBS -l walltime=05:00:00
#PBS -l software=python
#PBS -l wd
 
# Load modules.
module use /g/data3/hh5/public/modules
module load conda/analysis3
 
# Run Python applications
python3 rotunnoScript.py
