#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <random>
#include <immintrin.h>

#include "timer.h"
#include <sampling/methodR.h>
#include <sampling/methodH.h>
#include <sampling/methodSH.h>

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

    // MKL generates 0-based deviates, std::geometric_distribution is 1-based
    template <bool addone, typename It>
    static void inplace_prefix_sum(It begin, It end) {
        using value_type = typename std::iterator_traits<It>::value_type;

        if (begin == end) return;
        value_type sum = *begin;

        while (++begin != end) {
            sum += *begin;
            if (addone) sum++;
            *begin = sum;
        }
    }


    // Based on http://stackoverflow.com/a/32501562/3793885 by Peter Cordes
    // In-place prefix sum, optionally incrementing the sum between elements
    // (required for MKL geometric distribution...)
    template <bool addone, typename T>
    static void inplace_prefix_sum_sse(T* data, size_t n) {
        static_assert(std::is_same<T, int>::value, "can only do int");

        __m128i *datavec = (__m128i*)data;
        const int vec_elems = sizeof(*datavec)/sizeof(T);

        if (addone) data[0]--; // fix first element

        // don't start an iteration beyond this
        const __m128i *endp = (__m128i*) (data + n - 2*vec_elems);
        __m128i carry = _mm_setzero_si128();
        const __m128i ones = _mm_set1_epi32(1);

        for(; datavec <= endp ; datavec += 2) {
            __m128i x0 = _mm_load_si128(datavec + 0);
            __m128i x1 = _mm_load_si128(datavec + 1); // unroll / pipeline by 1

            if (addone) {
                x0 = _mm_add_epi32(x0, ones);
                x1 = _mm_add_epi32(x1, ones);
            }

            x0 = _mm_add_epi32(x0, _mm_slli_si128(x0, 4)); // sizeof(T)));
            x1 = _mm_add_epi32(x1, _mm_slli_si128(x1, 4)); // sizeof(T)));

            x0 = _mm_add_epi32(x0, _mm_slli_si128(x0, 8)); // 2*sizeof(T)));
            x1 = _mm_add_epi32(x1, _mm_slli_si128(x1, 8)); // 2*sizeof(T)));

            // more shifting if vec_elems is larger
            // this has to go after the byte-shifts, to avoid double-counting the carry.
            x0 = _mm_add_epi32(x0, carry);
            // store first to allow destructive shuffle (non-avx pshufb if needed)
            _mm_store_si128(datavec +0, x0);

            x1 = _mm_add_epi32(_mm_shuffle_epi32(x0, _MM_SHUFFLE(3,3,3,3)), x1);
            _mm_store_si128(datavec +1, x1);

            // broadcast the high element for next vector
            carry = _mm_shuffle_epi32(x1, _MM_SHUFFLE(3,3,3,3));
        }

        // handle the leftover elements
        T *ptr = (T*)datavec;
        if (ptr < data + n) {
            *ptr += *(ptr-1);
            inplace_prefix_sum<addone>(ptr, data+n);
        }
    }

    // AVX adaptation of the SSE2 prefix sum above.
    // Due to inter-lane shuffling, it's actually slower.
    template <bool addone>
    static void inplace_prefix_sum_avx(int* data, size_t n) {
        using T = int;
        const T* end = data + n;

        if (addone) data[0]--; // fix first element
        T sum = 0;
        // process first few elements manually to fulfill the 32-byte alignment
        // requirement of _mm256_load_si256
        while (((size_t)data & 31) && data < end) {
            sum += *data;
            if (addone) ++sum;
            *data++ = sum;
            --n;
        }

        __m256i *datavec = (__m256i*)data;
        const int vec_elems = sizeof(__m256i)/sizeof(T);

        // don't start an iteration beyond this
        const __m256i *endp = (__m256i*) (data + n - 2*vec_elems);
        __m256i carry = _mm256_set1_epi32(sum);

        for(; datavec <= endp ; datavec += 2) {
            __m256i x0 = _mm256_load_si256(datavec + 0);
            __m256i x1 = _mm256_load_si256(datavec + 1); // unroll / pipeline by 1

            if (addone) {
                x0 = _mm256_add_epi32(x0, _mm256_set1_epi32(1));
                x1 = _mm256_add_epi32(x1, _mm256_set1_epi32(1));
            }

            x0 = _mm256_add_epi32(x0,_mm256_alignr_epi8(x0,
                 _mm256_permute2x128_si256(x0, x0, _MM_SHUFFLE(0, 0, 2, 0)), 12));
            x1 = _mm256_add_epi32(x1, _mm256_alignr_epi8(x1,
                 _mm256_permute2x128_si256(x1, x1, _MM_SHUFFLE(0, 0, 2, 0)), 12));

            x0 = _mm256_add_epi32(x0, _mm256_alignr_epi8(x0,
                 _mm256_permute2x128_si256(x0, x0, _MM_SHUFFLE(0, 0, 2, 0)), 8));
            x1 = _mm256_add_epi32(x1, _mm256_alignr_epi8(x1,
                 _mm256_permute2x128_si256(x1, x1, _MM_SHUFFLE(0, 0, 2, 0)), 8));

            x0 = _mm256_add_epi32(x0, _mm256_permute2x128_si256(x0, x0, _MM_SHUFFLE(0, 0, 2, 0)));
            x1 = _mm256_add_epi32(x1, _mm256_permute2x128_si256(x1, x1, _MM_SHUFFLE(0, 0, 2, 0)));

            // more shifting if vec_elems is larger
            // this has to go after the byte-shifts, to avoid double-counting the carry.
            x0 = _mm256_add_epi32(x0, carry);
            // store first to allow destructive shuffle (non-avx pshufb if needed)
            _mm256_store_si256(datavec, x0);

            x1 = _mm256_add_epi32(_mm256_permutevar8x32_epi32(x0, _mm256_set1_epi32(7)), x1);
            _mm256_store_si256(datavec +1, x1);

            // broadcast the high element for next vector
            carry = _mm256_permutevar8x32_epi32(x1, _mm256_set1_epi32(7));
        }

        // handle the leftover elements
        if (data < end) {
            data = (T*)datavec;
            *data += *(data-1);
            inplace_prefix_sum<addone>(data, (T*)end);
        }
    }

    // Dispatch prefix sum to vectorized implementation if possible
    template <bool addone, typename It,
              typename value_type = typename std::iterator_traits<It>::value_type>
    static typename std::enable_if_t<std::is_same<value_type, int>::value>
    inplace_prefix_sum_disp(It begin, It end) {
        inplace_prefix_sum_sse<addone>(begin, end-begin);
    }

    // Fallback to non-vectorized implementation
    template <bool addone, typename It,
              typename value_type = typename std::iterator_traits<It>::value_type>
    static typename std::enable_if_t<!std::is_same<value_type, int>::value>
    inplace_prefix_sum_disp(It begin, It end) {
        inplace_prefix_sum<addone>(begin, end);
    }

    template <bool addone, size_t unroll = 8, typename It>
    static void inplace_prefix_sum_unroll(It begin, It end) {
        using value_type = typename std::iterator_traits<It>::value_type;
        value_type sum = 0;

        while(begin + unroll < end) {
            faux_unroll<unroll>::call(
                [&](size_t i){
                    sum += *(begin + i);
                    if (addone) ++sum;
                    *(begin + i) = sum;
                });
            begin += unroll;
        }

        while (begin < end) {
            sum += *begin;
            *begin++ = sum;
        }
    }

    // assumes holes[0] == -1 for simplified edge case handling
    template <typename It>
    static auto compact(It begin, It end, ssize_t* holes, size_t to_remove) {
        using value_type = typename std::iterator_traits<It>::value_type;

        It dest = begin;
        for (size_t i = 0; i < to_remove; ++i) {
            assert(holes[i+1] > holes[i]);
            size_t size = holes[i+1] - holes[i] - 1;
            //std::cout << "Moving " << size << " elements from index "
            //          << holes[i] + 1 << " to " << dest-begin << std::endl;
            memmove(dest, begin + holes[i] + 1, size * sizeof(value_type));
            dest += size;
        }
        // do the last gap
        auto srcpos = begin + holes[to_remove] + 1;
        if (srcpos < end) {
            // std::cout << "Moving last " << end-srcpos << " elements from "
            //           << srcpos - begin << " to pos " << dest - begin
            //           << " => last: " << dest + (end-srcpos) - begin
            //           << " vs " << k << std::endl;
            memmove(dest, srcpos, (end - srcpos) * sizeof(value_type));
            dest += (end-srcpos);
        }
        return dest;
    }

    template <typename It>
    static auto pick_holes(It begin, It end, size_t k, unsigned int seed,
                           bool sorted) {
        assert(end - begin > (long)k);
        const size_t to_remove = (end-begin) - k;

        if (seed == 0) {
            seed = std::random_device{}();
        }
        auto holes = std::make_unique<int64_t[]>(to_remove + 1);
        holes[0] = -1; // dummy
        size_t hole_idx = 1;

        const size_t basecase = 1024;
        // Only use for k up to 4MM, it gets slow after that
        if (sorted && k < (1<<22)) {
            // SORTED hash sampling
            SortedHashSampling<> hs((ULONG)seed, to_remove);
            SeqDivideSampling<StochasticLib1,SortedHashSampling<>> s(
                hs, basecase, (ULONG)seed);
            s.sample(end-begin, to_remove, [&](auto pos) {
                    holes[hole_idx++] = pos;
                });
        } else {
            HashSampling<> hs((ULONG)seed, to_remove);
            SeqDivideSampling<> s(hs, basecase, (ULONG)seed);
            s.sample(end-begin, to_remove, [&](auto pos) {
                    holes[hole_idx++] = pos;
                });
            if (sorted)
                std::sort(holes.get() + 1, holes.get() + to_remove + 1);
        }
        assert(hole_idx == to_remove + 1);

        if (sorted) {
            assert(std::is_sorted(holes.get(), holes.get() + to_remove + 1));
        }

        return holes;
    }

    template <typename It>
    static auto fix_stable(It begin, It end, size_t k, unsigned int seed = 0) {
        assert(end - begin > (long)k);

        // Get holes (sorted), then apply memmove compactor
        auto holes = pick_holes(begin, end, k, seed, true);

        size_t to_remove = (end-begin) - k;
        return compact(begin, end, holes.get(), to_remove);
    }


    template <typename It>
    static auto fix(It begin, It end, size_t k, unsigned int seed = 0) {
        assert(end-begin > (long)k);
        size_t to_remove = (end-begin) - k;

        // Holes should be sorted so we can process them in one sweep
        auto indices = pick_holes(begin, end, k, seed, true);

        auto last = end - 1;
        ssize_t pos = to_remove;
        // handle case where the last element is to be removed
        // revert last postincrement even if loop doesn't match
        if (begin + indices[pos] == last) { --last; --pos; }
        while (pos > 0) {
            //std::iter_swap(begin + indices[pos--], last--);
            *(begin + indices[pos--]) = std::move(*last--);
        }

        return last + 1;
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
                       global_stats *stats = nullptr,
                       const bool verbose = false, unsigned int seed = 0)
    {
        It pos;
        double t_gen(0.0), t_pref(0.0), t_fix(0.0);
        size_t its = 0; // how many iterations it took
        timer t, t_total;
        size_t usable_samples = 0;

        do {
            t.reset();
            gen_block(dest, dest+ssize, p, seed);
            // ensure a new seed is used in every iteration
            // (0 = generate a random seed)
            if (seed != 0) seed++;
            t_gen += t.get_and_reset();

            prefsum(dest, dest+ssize);
            t_pref += t.get_and_reset();

            // find out how many samples are within the universe.
            // pos is the first one that's too large.
            pos = std::lower_bound(dest, dest+ssize, universe);
            usable_samples = pos - dest;

            ++its;
            if (verbose)
                std::cout << "\tIt " << its << ": got " << usable_samples
                          << " samples in range (" << k << " of " << ssize
                          << " required) => "
                          << (long long)usable_samples - (long long)k
                          << " to delete, "  << ssize - usable_samples
                          << " outside universe ignored (largest: "
                          << *(dest + ssize - 1) << ")" << std::endl;
        } while (pos == (dest + ssize) || usable_samples < k);
        // first condition is that the whole universe is covered (i.e. ssize was
        // big enough) and second condition is that >= k samples lie within it

        t.reset();
        if (usable_samples > k) {
            // pick k out of the pos-dest-1 elements
#ifdef FIX_STABLE
            pos = fix_stable(dest, pos, k, seed);
#else
            pos = fix(dest, pos, k, seed);
#endif
            // pos is now the past-the-end iterator of the sample indices
            assert(pos - dest == (long)k);
        }
        t_fix += t.get_and_reset();

        double t_sum = t_total.get();
        if (stats != nullptr) {
            stats->push_sum(t_sum);
            stats->push_gen(t_gen);
            stats->push_prefsum(t_pref);
            stats->push_fix(t_fix);
        }

        std::stringstream stream;
        stream << "INFO"
               << " time=" << t_sum
               << " k=" << k
               << " b=" << ssize
               << " restarts=" << its-1
               << " t_gen=" << t_gen
               << " t_prefsum=" << t_pref
               << " t_fix=" << t_fix
#ifdef FIX_STABLE
               << " fixer=stable";
#else
               << " fixer=fast";
#endif
        return stream.str();
    }
};
