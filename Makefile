CXX ?= g++

MKLROOT ?= /opt/intel/compilers_and_libraries/linux/mkl
MKL ?= ${MKLROOT}/lib/intel64
MKLFLAGS = -L${MKL} -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

LDFLAGS=-Wl,-Bstatic ${MKLFLAGS} -Wl,-Bdynamic -ldl -lpthread

CFLAGS=-std=c++14 -I${MKLROOT}/include -Wall -Wextra -Werror -g
OPT=-Ofast -DNDEBUG -march=native -flto=8
DEBUG=-O0 -march=native
CFLAGS+=-IDistributedSampling/lib -IDistributedSampling/lib/tools -IDistributedSampling/extern/stocc -IDistributedSampling/extern/dSFMT
MPATH=DistributedSampling/optimized/extern
LDFLAGS+=libstocc.a

flags ?= # runtime flags

# use 64-bit integers
# explicit 32-bit integer suffix
ifeq ($(B64),0)
SUFF:=32
CFLAGS+=-DNOSTD
else
ifneq ($(B64),)
CFLAGS+=-DUSE64BIT
SUFF:=64
endif
endif

# use stable fixer
ifneq ($(STABLE),)
CFLAGS+=-DFIX_STABLE
SUFF:=${SUFF}S
endif

.PHONY: rand

rand: rand.cpp *.h
ifneq ($(SUFF),)
	@echo "Output name is rand${SUFF}"
endif
	${CXX} ${CFLAGS} ${OPT} -o rand${SUFF} rand.cpp ${LDFLAGS}

debug:
ifneq ($(SUFF),)
	@echo "Output name is rand${SUFF}-dbg"
endif
	${CXX} ${CFLAGS} ${DEBUG} -o rand${SUFF}-dbg rand.cpp ${LDFLAGS}

run:
	@LD_LIBRARY_PATH=${MKL} ./rand ${flags}

buildall:
	B64=0 make rand
	B64=0 STABLE=1 make rand
	B64=1 make rand
	B64=1 STABLE=1 make rand
