#pragma once

#include <limits>
#include <random>
#include <thread>
#include <vector>

#include "util.h"

struct std_sampler {
    template <typename InputIterator, typename F>
    static void sample(InputIterator begin, InputIterator end, double p, F &&callback, unsigned int seed = 0) {
        assert(p >= 0 && p <= 1);
        // handle degenerate cases
        if (p < nearly_zero) return;
        if (1.0 - p < nearly_zero) {
            InputIterator it(begin);
            while (it < end)
                callback(it++);
            return;
        }
        if (seed == 0) {
            std::random_device rd;
            seed = rd();
        }
        std::mt19937 gen(seed);
        std::bernoulli_distribution dist(p);

        InputIterator it(begin);
        while (it < end) {
            if (dist(gen)) {
                callback(it);
            }
            ++it;
        }
    }

    template <typename InputIterator, typename F>
    static void sample_skip_value(InputIterator begin, InputIterator end, double p, F &&callback, unsigned int seed = 0) {
        assert(p >= 0 && p <= 1);
        // handle degenerate cases
        if (1.0 - p < nearly_zero) {
            InputIterator it(begin);
            while (it < end)
                callback(it++);
            return;
        } else if (p < nearly_zero) {
            return;
        }

        if (seed == 0) {
            seed = std::random_device{}();
        }
        std::mt19937 gen(seed);
        std::geometric_distribution<long> dist(p);

        for (auto it = begin + dist(gen); it < end; ++it) {
            callback(it);
            it += dist(gen);
        }
    }

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

    static void generate_block(int *dest, size_t size, double p,
                                     unsigned int seed = 0) {
        generate_block(dest, dest+size, p, seed);
    }


    template <typename InputIterator, typename F,
              typename value = typename InputIterator::value_type>
    static void sample_skip_value_parallel(InputIterator begin, InputIterator end,
                                           double p, int num_threads, F &&callback) {
        assert(p >= 0 && p <= 1);
        int hardware_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads == -1) num_threads = hardware_threads;
        num_threads = std::min(num_threads, hardware_threads);

        unsigned int elems_per_thread((end - begin) / num_threads);
        auto leftovers = (end - begin) - elems_per_thread * num_threads;
        std::vector<InputIterator> *samples = new std::vector<InputIterator>[num_threads];

        // Worker function, simulating a number of PEs
        auto worker = [&p,&samples,&begin,&end](const size_t thread, const size_t firstIdx, const size_t lastIdx) {
            // Simulate local aggregations for PEs minPE to maxPE
            // Random number generator for each worker
            // TODO SEED
            std::mt19937 gen(std::random_device{}());
            std::geometric_distribution<> dist(p);
            const InputIterator last = begin + lastIdx;
            assert(last <= end);

            for (auto it = begin + firstIdx; it < last; ++it) {
                auto skip = dist(gen);
                it += skip;
                if (it < end) {
                    samples[thread].push_back(it);
                }
            }
        };

        std::vector<std::thread> workers;
        size_t min(0), max(elems_per_thread);
        for (auto thread = 0; thread < num_threads; ++thread) {
            if (thread < leftovers) max++;
            workers.push_back(std::thread(worker, thread, min, max));
            min = max;
            max += elems_per_thread;
        }

        // Wait for computations to finish
        for (std::thread &worker_thread : workers) {
            worker_thread.join();
        }

        for (auto thread = 0; thread < num_threads; ++thread) {
            for (const auto &sample : samples[thread]) {
                callback(sample);
            }
        }

        delete[] samples;
    }
};
