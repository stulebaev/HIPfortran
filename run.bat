@echo off

gfortran -c hipfort_enums.F90
gfortran -c hipfort_types.F90
gfortran -c hipfort_auxiliary.F90
gfortran -c hipfort_hipmemcpy.F90
gfortran -c hipfort_hipmalloc.F90
gfortran -c hipfort.F90

gfortran -c hipblas_module.f90
gfortran -c gemm_mod.f90 -fopenmp
gfortran gemm_prog.f90 gemm_mod.o -fopenmp -lhipblas -lamdhip64 -L%HIP_PATH%\lib -o gemm_prog

set PATH=%PATH%;%HIP_PATH%\bin
gemm_prog
