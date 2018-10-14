#!/bin/bash
gfortran -g -ffpe-trap=invalid,overflow,underflow \
 -fcheck=all -Wall -fdefault-integer-8 -fdefault-real-8 \
-I/usr/local/include \
../fortran_libraries/math.f90  \
../fortran_libraries/nc_tools.f90 \
$1 -o $2 \
-L/usr/local/lib -lnetcdff -fopenmp
