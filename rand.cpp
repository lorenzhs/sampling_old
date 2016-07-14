#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include "arg_parser.h"
#include "benchmark.h"
#include "util.h"
#include "MKL_sampler.h"
#include "std_sampler.h"

template <typename It>
void inplace_prefix_sum(It begin, It end) {
    using value_type = typename std::iterator_traits<It>::value_type;

    if (begin == end) return;
    value_type sum = *begin;

    while (++begin != end) {
        sum += *begin;
        *begin = sum;
    }
}

template <size_t unroll = 8, typename It>
void inplace_prefix_sum_unroll(It begin, It end) {
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

template <typename F>
void run(F&& runner, const std::vector<std::unique_ptr<int[]>> &data,
         int num_threads, int iterations, std::string name,
         const bool verbose = false) {

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
                      << " iteration=" << iteration << std::endl;
    }

    std::cout << "RESULT runner=" << name
              << " time=" << stats.avg()
              << " stddev=" << stats.stddev()
              << " numthreads=" << num_threads
              << " iterations=" << iterations
              << std::endl;
}

int main(int argc, char** argv) {
    arg_parser args(argc, argv);
    size_t size = args.get<size_t>("s", 1<<28);
    double p = args.get<double>("p", 0.1);
    int num_threads = args.get<int>("t", 1);
    int iterations = args.get<int>("i", 1);
    static std::mutex cout_mutex;
    const bool verbose = args.is_set("v");

    auto data = std::vector<std::unique_ptr<int[]>>(num_threads);
    // initialize in parallel
    run([size, &data](auto /* dataptr */, int thread, int /* iteration */) {
            data[thread] = std::make_unique<int[]>(size); // weak scaling
            // ensure that the memory is initialized
            std::fill(data[thread].get(), data[thread].get() + size, 0);
        }, data, num_threads, 1, "init");

    // warmup
    size_t warmup_size = std::min<size_t>(1024*1024, size);
    std::cout << "Running warmup (size " << warmup_size << ")" << std::endl;
    run([warmup_size, p](auto data, int /* thread */, int /* iteration */) {
            MKL_sampler::generate_block(data, warmup_size, p);
            inplace_prefix_sum_unroll(data, data + warmup_size);
            std_sampler::generate_block(data, warmup_size, p);
            inplace_prefix_sum(data, data + warmup_size);
        }, data, num_threads, 1, "warmup");

    // Measure MKL_sampler
    std::cout << "Running measurements..." << std::endl;
    run([size, p, num_threads, verbose]
        (int *data, int thread_id, int iteration){
            timer t;
            MKL_sampler::generate_block(data, size, p);
            double t_sample = t.get_and_reset();
            // unrolling 12 times is fastest on my machine
            inplace_prefix_sum_unroll<12>(data, data + size);
            double t_prefsum = t.get();

            // timer output is in milliseconds
            if (verbose) {
                cout_mutex.lock();
                std::cout
                    << "RESULT size=" << size << " p=" << p
                    << " time=" << t_sample + t_prefsum
                    << " throughput=" << size*1000.0 / (t_sample + t_prefsum)
                    << " method=mkl t_sample=" << t_sample
                    << " t_prefsum=" << t_prefsum
                    << " thread_id=" << thread_id
                    << " num_threads=" << num_threads
                    << " iteration=" << iteration << std::endl;
                cout_mutex.unlock();
            }
        }, data, num_threads, iterations, "mkl");


    // Measure std_sampler
    run([size, p, num_threads, verbose]
        (int *data, int thread_id, int iteration){
            timer t;
            std_sampler::generate_block(data, size, p);
            double t_sample = t.get_and_reset();
            inplace_prefix_sum(data, data + size);
            double t_prefsum = t.get();

            if (verbose) {
                cout_mutex.lock();
                // timer output is in milliseconds
                std::cout
                    << "RESULT size=" << size << " p=" << p
                    << " time=" << t_sample + t_prefsum
                    << " throughput=" << size*1000.0 / (t_sample + t_prefsum)
                    << " method=std t_sample=" << t_sample
                    << " t_prefsum=" << t_prefsum
                    << " thread_id=" << thread_id
                    << " num_threads=" << num_threads
                    << " iteration=" << iteration << std::endl;
                cout_mutex.unlock();
            }
        }, data, num_threads, iterations, "std");
}
