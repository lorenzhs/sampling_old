#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include "arg_parser.h"
#include "benchmark.h"
#include "util.h"
#include "sampler.h"
#include "MKL_gen.h"
#include "std_gen.h"


template <typename T, typename F>
void run(F&& runner, const std::vector<std::unique_ptr<T[]>> &data,
         int num_threads, int iterations, std::string name,
         const std::string extra = "", const bool verbose = false) {

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    statistics stats;
    timer t; // start the clock

    for (int iteration = 0; iteration < iterations; ++iteration) {
        threads.clear();

        t.reset();
        for (int thread = 0; thread < num_threads; ++thread) {
            threads.emplace_back(runner, data[thread].get(), thread, iteration);
        }

        for (auto &thread : threads) {
            thread.join();
        }

        double time = t.get();
        stats.push(time);
        // we can't print the other info because it's encapsulated in the runner
        if (verbose)
            std::cout << "RESULT runner=" << name << " time=" << time
                      << " num_threads=" << num_threads
                      << " iteration=" << iteration
                      << extra << std::endl;
    }

    std::cout << "RESULT runner=" << name
              << " time=" << stats.avg()
              << " stddev=" << stats.stddev()
              << " numthreads=" << num_threads
              << " iterations=" << iterations
              << extra << std::endl;
}


// Formulas from "Sequential Random Sampling" by Ahrens and Dieter, 1985
static constexpr auto calc_params(size_t universe, size_t k /* samples */) {
    double r = sqrt(k);
    double a = sqrt(log(1+k/(2*M_PI)));
    a = a + a*a/(3.0 * r);
    size_t b = k + size_t(4 * a * r);
    double p = (k + a * r) / universe;
    return std::make_pair(p, b);
}


int main(int argc, char** argv) {
    arg_parser args(argc, argv);
    size_t universe = args.get<size_t>("n", 1<<30);
    size_t k = args.get<size_t>("k", 1<<20); // sample size

    int num_threads = args.get<int>("t", 1);
    int iterations = args.get<int>("i", 1);
    static std::mutex cout_mutex;
    const bool verbose = args.is_set("v");

    double p; size_t ssize;
    std::tie(p, ssize) = calc_params(universe, k);

    std::cout << "Geometric sampler, " << k << " samples per thread "
              << "(p = " << p << ") from universe of size " << universe
              << ", using " << num_threads << " thread(s), "
              << iterations << " iteration(s)." << std::endl;

    auto data = std::vector<std::unique_ptr<int[]>>(num_threads);
    // initialize in parallel
    run([ssize, &data](auto /* dataptr */, int thread, int /* iteration */) {
            data[thread] = std::make_unique<int[]>(ssize); // weak scaling
            // ensure that the memory is initialized
            std::fill(data[thread].get(), data[thread].get() + ssize, 0);
        }, data, num_threads, 1, "init");

    // warmup
    size_t warmup_size = std::min<size_t>(1024*1024, ssize);
    std::cout << "Running warmup (size " << warmup_size << ")" << std::endl;
    run([warmup_size, p](auto data, int /* thread */, int /* iteration */) {
            MKL_sampler::generate_block(data, warmup_size, p);
            /* Fuck GCC and Intel.
             *
             * No seriously, fuck them. Without this call here, wildly
             * inefficient code is generated for std_sampler. If I call
             * inplace_prefix_sum here, std_sampler will be ~20% slower and
             * inplace_prefix_sum will be weird, too, and MKL behaves oddly too.
             */
            sampler::inplace_prefix_sum_unroll<12>(data, data + warmup_size);

            std_sampler::generate_block(data, warmup_size, p);
            sampler::inplace_prefix_sum(data, data + warmup_size);
        }, data, num_threads, 1, "warmup");

    std::stringstream extra_stream;
    extra_stream << " ssize=" << ssize << " p=" << p;
    auto extra = extra_stream.str();

    // Measure MKL_sampler
    std::cout << "Running measurements..." << std::endl;
    run([universe, k, p, ssize, num_threads, verbose]
        (auto data, int thread_id, int iteration){

            auto msg = sampler::sample(
                data, ssize, k, p, universe,
                [](auto begin, auto end, double p, unsigned int seed)
                { return MKL_sampler::generate_block(
                        begin, end-begin, p,
                        MKL_sampler::gen_method::geometric, seed); });

            //MKL_sampler::generate_block(data, ssize, p);

            if (verbose) {
                cout_mutex.lock();
                // timer output is in milliseconds
                std::cout << msg << " method=mkl"
                          << " thread_id=" << thread_id
                          << " num_threads=" << num_threads
                          << " iteration=" << iteration << std::endl;
                cout_mutex.unlock();
            }
        }, data, num_threads, iterations, "mkl", extra);


    // Measure std_sampler
    run([universe, k, p, ssize, num_threads, verbose]
        (auto data, int thread_id, int iteration){
            auto msg = sampler::sample(
                data, ssize, k, p, universe,
                [](auto begin, auto end, double p, unsigned int seed)
                { return std_sampler::generate_block(begin, end, p, seed); });

            if (verbose) {
                cout_mutex.lock();
                // timer output is in milliseconds
                std::cout << msg << " method=std"
                          << " thread_id=" << thread_id
                          << " num_threads=" << num_threads
                          << " iteration=" << iteration << std::endl;
                cout_mutex.unlock();
            }
        }, data, num_threads, iterations, "std", extra);
}
