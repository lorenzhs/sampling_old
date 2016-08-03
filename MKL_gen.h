#pragma once

#include <cassert>
#include <random>

#include <mkl.h>
#include <mkl_vsl.h>

#include "util.h"
#include "timer.h"

#include "errcheck.inc"

struct MKL_gen {
    static void generate_block(int *dest, size_t size, double p,
                               unsigned int seed = 0) {
        VSLStreamStatePtr stream;
        if (seed == 0) {
            std::random_device rd;
            seed = rd();
        }
        vslNewStream(&stream, VSL_BRNG_SFMT19937, seed);

        if (size >= std::numeric_limits<int>::max()) {
            SERR << "Error: MKL_Sampler block size exceeds value range of int:"
                 << size << " >= " << std::numeric_limits<int>::max()
                 << std::endl;
        }

        int count = static_cast<int>(size);
        int status = viRngGeometric(VSL_RNG_METHOD_GEOMETRIC_ICDF, stream,
                                    count, dest, p);

        vslDeleteStream(&stream);
        CheckVslError(status);
    }
};
