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

    // warmup
    size_t warmup_size = std::min<size_t>(1024*1024, size);
    std::cout << "Running warmup (size " << warmup_size << ")" << std::endl;
    MKL_sampler::generate_block(
        data.get(), warmup_size, p, MKL_sampler::gen_method::geometric);

    std::cout << "Running measurement..." << std::endl;
    timer t;
    MKL_sampler::generate_block(
        data.get(), size, p, MKL_sampler::gen_method::geometric);

    // timer output is in milliseconds
    std::cout << "RESULT size=" << size << " p=" << p
              << " time=" << t.get() << std::endl;
}
