#include <algorithm>
#include <iostream>
#include <memory>

#include "arg_parser.h"
#include "util.h"
#include "MKL_sampler.h"
#include "std_sampler.h"

int main(int argc, char** argv) {
    arg_parser args(argc, argv);
    size_t size = args.get<size_t>("s", 1<<28);
    double p = args.get<double>("p", 0.1);

    auto data = std::make_unique<int[]>(size);
    // ensure that the memory is initialized
    std::fill(data.get(), data.get() + size, 0);

    // warmup
    size_t warmup_size = std::min<size_t>(1024*1024, size);
    std::cout << "Running warmup (size " << warmup_size << ")" << std::endl;
    MKL_sampler::generate_block(data.get(), warmup_size, p);
    std_sampler::generate_block(data.get(), warmup_size, p);

    std::cout << "Running measurement..." << std::endl;
    timer t;
    MKL_sampler::generate_block(data.get(), size, p);
    double time = t.get();

    // timer output is in milliseconds
    std::cout << "RESULT size=" << size << " p=" << p
              << " time=" << time << " throughput=" << size * 1000.0 / time
              << " method=mkl" << std::endl;

    t.reset();
    std_sampler::generate_block(data.get(), size, p);
    time = t.get();

    // timer output is in milliseconds
    std::cout << "RESULT size=" << size << " p=" << p
              << " time=" << time << " throughput=" << size * 1000.0 / time
              << " method=std" << std::endl;

}
