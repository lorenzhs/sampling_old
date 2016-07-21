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

struct MKL_sampler {
    enum gen_method { bernoulli, geometric };
    static const bool debug = true;

    template <size_t blocksize = (1 << 24),
              typename InputIterator, typename F, typename G>
    static sampling_stats sample(InputIterator begin, InputIterator end,
                                 double p, F &&callback, G &&seedfn,
                                 const bool debug_all = false) {
        assert(p >= 0 && p <= 1);
        // Handle degenerate cases directly
        if (1.0 - p < nearly_zero) { // p == 1
            timer t;
            while (begin != end) {
                callback(begin++);
            }
            return sampling_stats(0, t.get());
        } else if (p < nearly_zero) { // p == 0
            return sampling_stats(0, 0);
        }
        // Actually geometric seems to always be faster, even for large p
        //if (p > 0.33) {
        //    return sample_bernoulli(begin, end, p, callback, seedfn, blocksize);
        //} else {
        return sample_geometric(begin, end, p, callback, seedfn, blocksize, debug_all);
        //}
    }

    //protected:
    template <typename InputIterator, typename F, typename G>
    static sampling_stats sample_bernoulli(InputIterator begin, InputIterator end,
        double p, F &&callback, G &&seedfn, const size_t blocksize)
    {
        sampling_stats stats;

        const size_t num_elems = end - begin + 1;
        const size_t num_blocks = (num_elems + blocksize - 1) / blocksize;

        timer t;
        size_t curr_blocksize = (num_blocks == 1) ? num_elems : blocksize;
        int *randblock = new int[curr_blocksize];
        stats.t_gen += t.get_and_reset();  // time for allocation

        for (size_t block = 0; block < num_blocks; ++block) {
            InputIterator it = begin + block * blocksize;
            InputIterator last = it + blocksize;
            if (block + 1 == num_blocks) {  // last block
                last = end;
                curr_blocksize = (end - it + 1);
            }
            // generate random data
            stats.t_sample += t.get_and_reset();
            generate_block(randblock, curr_blocksize, p, gen_method::bernoulli, seedfn());
            stats.t_gen += t.get_and_reset();

            for (size_t i = 0; i < curr_blocksize; ++i, ++it) {
                if (randblock[i] == 1) {
                    callback(it);
                }
            }
        }

        delete[] randblock;

        stats.t_sample += t.get();
        return stats;
    }

    template <typename InputIterator, typename F, typename G>
    static sampling_stats sample_geometric(InputIterator begin, InputIterator end,
        double p, F &&callback, G&& seedfn, const size_t blocksize,
        const bool debug_all = false)
    {
        sampling_stats stats;
        const size_t num_elems = static_cast<size_t>(end - begin) + 1;
        // (Over-)estimate #samples
        size_t exp_samples = static_cast<size_t>(static_cast<double>(num_elems) * p * 1.2);
        size_t randblock_size = std::min(exp_samples, blocksize);
        randblock_size = std::max(randblock_size, 256UL);

        timer t;
        int *randblock = new int[randblock_size];
        generate_block(randblock, randblock_size, p, gen_method::geometric, seedfn());
        stats.t_gen += t.get_and_reset();

        if (debug_all)
            SLOG << "MKL Sampler: expecting " << exp_samples
                 << " samples per PE (p = " << p << ", #elems = " << num_elems
                 << "), randblock_size = " << randblock_size << std::endl;

        size_t block_idx(1);
        // Do the first step explicitly. This allows us to restructure the loop
        // below to call the callback first, and avoids an if (it < end) check.
        for (auto it = begin + randblock[0]; it < end; ++it) {
            callback(it);
            // There is a ++it above and a skip down here. That is correct:
            // the geometric distribution can also be 0 (i.e. don't skip any
            // element), which means sampling the next element. This could
            // also be achieved by adding 1 here, but that would require extra
            // handling at the beginning...
            //assert(block_idx < randblock_size && block_idx < orig_randblock_size);
            it += randblock[block_idx++];

            // Check wether we need a new block
            if (unlikely(block_idx == randblock_size)) {
                // Abort if we just made it
                if (it >= end) break;

                block_idx = 0;
                exp_samples = static_cast<size_t>(
                    static_cast<double>(end-it) * p * 1.2);
                if (exp_samples <= 0)
                    SLOG << "WTF exp_samples ≤ 0: " << exp_samples << std::endl;

                // generating blocks that are too small isn't worth it
                randblock_size = std::min(std::max(exp_samples, 128UL),
                                          randblock_size);

                SLOG << "MKL Sampler: Generating another block of size "
                     << randblock_size << " (expecting " << exp_samples
                     << " samples)" << std::endl;

                stats.t_sample += t.get_and_reset();

                // this is safe as randblock_size can only decrease
                generate_block(randblock, randblock_size, p, gen_method::geometric, seedfn());
                stats.t_gen += t.get_and_reset();
            }
        }

        delete[] randblock;

        stats.t_sample += t.get_and_reset();
        return stats;
    }


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
