#!/bin/bash
#PBS -q express
#PBS -P w40
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l jobfs=100MB
#PBS -l walltime=02:00:00
#PBS -l software=fortran
#PBS -l wd
 
# Run fortran program
./rotunnoCaseTwoTest
