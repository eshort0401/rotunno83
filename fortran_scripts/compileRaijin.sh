#!/bin/bash
# Load required modules
module unload intel-cc
module unload intel-fc
module unload netcdf
module load intel-cc/2018.3.222
module load intel-fc/2018.3.222
module load netcdf/4.6.1

# Compile fortran
ifort -g -check all -warn all -i8 -r8 \
-I/usr/local/include \
../fortran_libraries/math.f90  \
../fortran_libraries/nc_tools.f90 \
$1 -o $2 \
-lnetcdff -fopenmp
