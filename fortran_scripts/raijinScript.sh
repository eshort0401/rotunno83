#!/bin/bash
#PBS -q express
#PBS -P w40
#PBS -l ncpus=16
#PBS -l mem=32GB
#PBS -l jobfs=100B
#PBS -l walltime=00:30:00
#PBS -l software=fortran
#PBS -l wd
 
# Run fortran program
./rotunnoCaseTwo 0.20 10.0
