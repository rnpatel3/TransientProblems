
PLANE_STRESS_SRC = precision.f90 \
	quicksort.f90 \
	plane_stress.f90

F90_FLAGS=
F90=gfortran

default: ${PLANE_STRESS_SRC}
	f2py -L/usr/lib -llapack -lblas ${F90_FLAGS} -c -m plane_stress ${PLANE_STRESS_SRC}

debug: F90_FLAGS=--noopt --debug --f90flags="-fcheck=all"
debug: default

clean:
	rm -f *.o *.so
