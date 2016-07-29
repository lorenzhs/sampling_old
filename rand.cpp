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
    global_stats stats;
    // stats for multi-threaded version including sync / load imbalance
    statistics mt_stats;
    timer t;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        threads.clear();

        t.reset();
        if (num_threads > 1) {
            for (int thread = 0; thread < num_threads; ++thread) {
                threads.emplace_back(runner, data[thread].get(), thread,
                                     iteration, &stats);
            }

            for (auto &thread : threads) {
                thread.join();
            }
        } else {
            runner(data[0].get(), 0, iteration, &stats);
        }

        double time = t.get();
        mt_stats.push(time);

        // we can't print the other info because it's encapsulated in the runner
        if (verbose)
            std::cout << "RESULT runner=" << name << " time=" << time
                      << " num_threads=" << num_threads
                      << " iteration=" << iteration
                      << extra << std::endl;
    }

    std::cout << "RESULT runner=" << name
              << " time=" << stats.s_sum.avg()
              << " stddev=" << stats.s_sum.stddev();
    if (num_threads > 1)
        std::cout << " mt_time=" << mt_stats.avg()
                  << " mt_dev=" << mt_stats.stddev();
    std::cout << " t_gen=" << stats.s_gen.avg()
              << " t_gen_dev=" << stats.s_gen.stddev()
              << " t_fix=" << stats.s_fix.avg()
              << " t_fix_dev=" << stats.s_fix.stddev()
              << " t_pref=" << stats.s_prefsum.avg()
              << " t_pref_dev=" << stats.s_prefsum.stddev()
              << " numthreads=" << num_threads
              << " iterations=" << iterations
#ifdef FIX_STABLE
              << " fixer=stable"
#else
              << " fixer=fast"
#endif
              << extra << std::endl;
}

#ifndef USE64BIT
using T = int32_t;
#else
using T = int64_t;
#endif
static std::mutex cout_mutex;

int main(int argc, char** argv) {
    arg_parser args(argc, argv);
    size_t universe = args.get<size_t>("n", 1<<30);
    size_t k = args.get<size_t>("k", 1<<20); // sample size

    int num_threads = args.get<int>("t", 1);
    int iterations = args.get<int>("i", 1);
    const bool verbose = args.is_set("v") || args.is_set("vv");
    const bool very_verbose = args.is_set("vv");

    double p; size_t ssize;
    std::tie(p, ssize) = sampler::calc_params(universe, k);

    std::cout << "Geometric sampler, " << k << " samples per thread "
              << "(p = " << p << ") from universe of size " << universe
              << ", using " << num_threads << " thread(s), "
              << iterations << " iteration(s)." << std::endl;

    auto data = std::vector<std::unique_ptr<T[]>>(num_threads);
    // initialize in parallel
    run([ssize, &data]
        (auto /* dataptr */, int thread, int /* iteration */, auto /*stats*/) {
            data[thread] = std::make_unique<T[]>(ssize); // weak scaling
            // ensure that the memory is initialized
            std::fill(data[thread].get(), data[thread].get() + ssize, 0);
        }, data, num_threads, 1, "init");

    // warmup
    size_t k_warmup = std::min<size_t>(1<<16, k);
    std::cout << "Running warmup (" << k_warmup << " samples)" << std::endl;
    run([k_warmup, universe]
        (auto data, int /*thread*/, int /*iteration*/, auto stats) {
            double p_warmup; size_t ssize_warmup;
            std::tie(p_warmup, ssize_warmup) =
                sampler::calc_params(universe, k_warmup);
            // MKL_gen
#ifndef USE64BIT
            sampler::sample(
                data, ssize_warmup, k_warmup, p_warmup, universe,
                [](auto begin, auto end, double p, unsigned int seed)
                { return MKL_gen::generate_block(
                        begin, end-begin, p,
                        MKL_gen::gen_method::geometric, seed); },
                [](auto begin, auto end)
                { return sampler::inplace_prefix_sum(begin, end); }, stats);
#endif

            // std_gen
            sampler::sample(
                data, ssize_warmup, k_warmup, p_warmup, universe,
                [](auto begin, auto end, double p, unsigned int seed)
                { return std_gen::generate_block(begin, end, p, seed); },
                [](auto begin, auto end)
                { return sampler::inplace_prefix_sum(begin, end); }, stats);
        }, data, num_threads, 100, "warmup");

    std::stringstream extra_stream;
    extra_stream << " k=" << k << " b=" << ssize
                 << " p=" << p << " N=" << universe;
    auto extra = extra_stream.str();

    // Measure MKL_gen
    std::cout << "Running measurements..." << std::endl;
#ifndef USE64BIT
    run([universe, k, p, ssize, num_threads, verbose, very_verbose]
        (auto data, int thread_id, int iteration, auto stats){
            auto msg = sampler::sample(
                data, ssize, k, p, universe,
                [](auto begin, auto end, double p, unsigned int seed)
                { return MKL_gen::generate_block(
                        begin, end-begin, p,
                        MKL_gen::gen_method::geometric, seed); },
                [](auto begin, auto end)
                { return sampler::inplace_prefix_sum(begin, end); },
                stats, very_verbose);

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
#endif


    // Measure std_gen
    run([universe, k, p, ssize, num_threads, verbose, very_verbose]
        (auto data, int thread_id, int iteration, auto stats){
            auto msg = sampler::sample(
                data, ssize, k, p, universe,
                [](auto begin, auto end, double p, unsigned int seed)
                { return std_gen::generate_block(begin, end, p, seed); },
                [](auto begin, auto end)
                { return sampler::inplace_prefix_sum(begin, end); },
                stats, very_verbose);

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
