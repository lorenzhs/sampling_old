CXX ?= g++

MKLROOT ?= /opt/intel/composer_xe_2015.2.164/mkl
MKL ?= ${MKLROOT}/lib/intel64
MKLFLAGS = -L${MKL} -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

LDFLAGS=-Wl,-Bstatic ${MKLFLAGS} -Wl,-Bdynamic -ldl -lpthread

CFLAGS=-std=c++14 -I${MKLROOT}/include -Wall -Wextra -Werror -g
OPT=-Ofast -DNDEBUG -march=native -flto=8
DEBUG=-O0 -march=native
CFLAGS+=-IDistributedSampling/lib -IDistributedSampling/lib/tools -IDistributedSampling/extern/stocc -IDistributedSampling/extern/dSFMT
MPATH=DistributedSampling/optimized/extern
LDFLAGS+=${MPATH}/mersenne/*.o ${MPATH}/stocc/stoc1.o ${MPATH}/stocc/wnchyppr.o ${MPATH}/dSFMT/dSFMT.o

flags ?= # runtime flags

.PHONY: rand

rand: rand.cpp *.h
	${CXX} ${CFLAGS} ${OPT} -o rand rand.cpp ${LDFLAGS}

debug:
	${CXX} ${CFLAGS} ${DEBUG} -o rand rand.cpp ${LDFLAGS}

run:
	@LD_LIBRARY_PATH=${MKL} ./rand ${flags}
