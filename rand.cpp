#include <algorithm>
#include <iostream>
#include <memory>

#include "arg_parser.h"
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

// template-based loop unrolling
template <size_t N> struct faux_unroll {
    template <typename F> static void call(F &&f) {
        faux_unroll<N-1>::call(f);
        f(N-1);
    }
};

template <> struct faux_unroll<0> {
    template <typename F> static void call(F &&) {}
};

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
    inplace_prefix_sum_unroll(data.get(), data.get() + warmup_size);
    std_sampler::generate_block(data.get(), warmup_size, p);
    inplace_prefix_sum(data.get(), data.get() + warmup_size);

    std::cout << "Running measurement..." << std::endl;
    timer t;
    MKL_sampler::generate_block(data.get(), size, p);
    double t_sample = t.get_and_reset();
    // unrolling 12 times is fastest on my machine
    inplace_prefix_sum_unroll<12>(data.get(), data.get() + size);
    double t_prefsum = t.get();

    // timer output is in milliseconds
    std::cout << "RESULT size=" << size << " p=" << p
              << " time=" << t_sample + t_prefsum
              << " throughput=" << size * 1000.0 / (t_sample + t_prefsum)
              << " method=mkl t_sample=" << t_sample
              << " t_prefsum=" << t_prefsum << std::endl;

    t.reset();
    std_sampler::generate_block(data.get(), size, p);
    t_sample = t.get_and_reset();
    inplace_prefix_sum(data.get(), data.get() + size);
    t_prefsum = t.get();

    // timer output is in milliseconds
    std::cout << "RESULT size=" << size << " p=" << p
              << " time=" << t_sample + t_prefsum
              << " throughput=" << size * 1000.0 / (t_sample + t_prefsum)
              << " method=std t_sample=" << t_sample
              << " t_prefsum=" << t_prefsum << std::endl;

}
