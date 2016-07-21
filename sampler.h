#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

#include "timer.h"

struct sampler {
    // Formulas from "Sequential Random Sampling" by Ahrens and Dieter, 1985
    static auto calc_params(size_t universe, size_t k /* samples */) {
        double r = sqrt(k);
        double a = sqrt(log(1+k/(2*M_PI)));
        a = a + a*a/(3.0 * r);
        size_t b = k + size_t(4 * a * r);
        double p = (k + a * r) / universe;
        return std::make_pair(p, b);
    }

    template <typename It>
    static void inplace_prefix_sum(It begin, It end) {
        using value_type = typename std::iterator_traits<It>::value_type;

        if (begin == end) return;
        value_type sum = *begin;

        while (++begin != end) {
            sum += *begin;
            *begin = sum;
        }
    }

    template <size_t unroll = 8, typename It>
    static void inplace_prefix_sum_unroll(It begin, It end) {
        using value_type = typename std::iterator_traits<It>::value_type;
        value_type sum = 0;

        while(begin + unroll < end) {
            faux_unroll<unroll>::call(
                [&](size_t i){
                    sum += *(begin + i);
                    *(begin + i) = sum;
                });
            begin += unroll;
        }

        while (begin < end) {
            sum += *begin;
            *begin++ = sum;
        }
    }


    template <typename It>
    static auto fix(It begin, It end, size_t k, unsigned int seed = 0) {
        assert(size > k);
        if (seed == 0) {
            seed = std::random_device{}();
        }
        std::mt19937 gen(seed);
        std::uniform_int_distribution<size_t> dist(0, end-begin);

        size_t to_remove = (end-begin) - k;

        while (to_remove > 0) {
            size_t pos = dist(gen);
            if (*(begin + pos) > -1) {
                *(begin + pos) = -1;
                to_remove--;
            }
        }

        // compact
        return std::remove(begin, end, -1);
    }

    /**
     * @param dest output iterator
     * @param size size of array (last valid pos is dest + size - 1)
     * @param k number of samples to draw
     * @param universe value range of samples [0..universe)
     */
    template <typename It, typename F, typename G>
    static auto sample(It dest, size_t ssize, size_t k, double p,
                       size_t universe, F&& gen_block, G&& prefsum,
                       const bool verbose = false, unsigned int seed = 0)
    {
        It pos;
        double t_gen(0.0), t_pref(0.0), t_check(0.0), t_fix(0.0);
        size_t its = 0; // how many iterations it took
        timer t;

        do {
            t.reset();
            gen_block(dest, dest+ssize, p, seed);
            t_gen += t.get_and_reset();

            prefsum(dest, dest+ssize);
            t_pref += t.get_and_reset();

            // find out how many samples are within the universe.
            // pos is the first one that's too large.
            pos = std::lower_bound(dest, dest+ssize, universe);
            t_check += t.get_and_reset();

            ++its;
            if (verbose)
                std::cout << "\tIt " << its << ": got " << pos-dest
                          << " samples in range (" << k << " required) => "
                          << (pos - dest) - k << " to delete, "
                          << dest + ssize - pos << " outside universe ignored"
                          << std::endl;
        } while (pos < (dest + ssize) && (pos - dest) < (long)k);
        // first condition is that the whole universe is covered (i.e. ssize was
        // big enough) and second condition is that >= k samples lie within it

        t.reset();
        if (pos - dest > (long)k) {
            // pick k out of the pos-dest-1 elements
            pos = fix(dest, pos, k, seed);
            // pos is now the past-the-end iterator of the sample indices
            assert(pos - dest == (long)k);
        }
        t_fix += t.get_and_reset();

        std::stringstream stream;
        stream << "INFO"
               << " time=" << t_gen + t_pref + t_check + t_fix
               << " k=" << k
               << " b=" << ssize
               << " restarts=" << its-1
               << " t_gen=" << t_gen
               << " t_prefsum=" << t_pref
               << " t_check=" << t_check
               << " t_fix=" << t_fix;
        return stream.str();
    }
};
