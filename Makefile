CXX ?= g++

MKLROOT ?= /opt/intel/composer_xe_2015.2.164/mkl
MKL ?= ${MKLROOT}/lib/intel64
MKLFLAGS = -L${MKL} -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

LDFLAGS=${MKLFLAGS} -ldl -lpthread

CFLAGS=-std=c++14 -I${MKLROOT}/include -Ofast -g -DNDEBUG

flags ?= # runtime flags

.PHONY: rand

rand: rand.cpp *.h
	${CXX} ${CFLAGS} -o rand rand.cpp ${LDFLAGS}

run:
	@LD_LIBRARY_PATH=${MKL} ./rand ${flags}
