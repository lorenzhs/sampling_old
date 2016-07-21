#pragma once

#include <limits>
#include <random>
#include <thread>
#include <vector>

#include "util.h"

struct std_gen {
    template <typename OutputIterator>
    static void generate_block(OutputIterator begin, OutputIterator end,
                               double p, unsigned int seed = 0) {
        assert(p >= 0 && p <= 1);
        // handle degenerate cases
        /*
        if (1.0 - p < nearly_zero) {
            for (auto it = begin; it < end; ++it) {
                *it = 1;
            }
        } else if (p < nearly_zero) {
            return;
        }
        */

        if (seed == 0) {
            seed = std::random_device{}();
        }
        std::mt19937 gen(seed);
        std::geometric_distribution<long> dist(p);

        for (auto it = begin; it < end; ++it) {
            *it = dist(gen);
        }
    }

    template <typename It>
    static void generate_block(It dest, size_t size, double p,
                               unsigned int seed = 0) {
        generate_block(dest, dest+size, p, seed);
    }
};
