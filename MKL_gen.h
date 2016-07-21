#pragma once

#include <cassert>
#include <random>

#include <mkl.h>
#include <mkl_vsl.h>

#include "util.h"
#include "timer.h"

#include "errcheck.inc"

struct sampling_stats {
    double t_gen, t_sample;
    explicit sampling_stats(double gen = 0, double sample = 0)
        : t_gen(gen)
        , t_sample(sample)
        {}
};

struct MKL_gen {
    enum gen_method { bernoulli, geometric };

    static void generate_block(int *dest, size_t size, double p,
                               gen_method method = gen_method::geometric,
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

        int status;
        if (method == gen_method::bernoulli) {
            status = viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, count, dest, p);
        } else if (method == gen_method::geometric) {
            status = viRngGeometric(VSL_RNG_METHOD_GEOMETRIC_ICDF, stream, count, dest, p);
        } else {
            std::cerr << "invalid generation method";
            status = VSL_ERROR_BADARGS;
        }

        vslDeleteStream(&stream);
        CheckVslError(status);
    }
};
