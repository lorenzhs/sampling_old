#pragma once

#include <csignal>

#define likely(x)   __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

static double nearly_zero = 1e-10;

void wait_for_gdb() {
    // attaching gdb and cont'ing will make it continue
    raise(SIGSTOP);
}

template <typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params) {
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

// find most siginificant one bit
template <typename int_or_long = unsigned int>
int highest_set_bit(int_or_long num) {
    int msb;
    asm("bsrl %1,%0" : "=r"(msb) : "r"(num));
    return msb;
}


// Some logging
#define COUT std::cout << "PE " << comm.rank() << " "
#define LOG  if (debug) std::cout << "PE " << comm.rank() << " "
#define INFO if (debug) std::cout << "PE " << comm.rank() << " INFO @ " << __FILE__ ":" << __LINE__ << " (" << __PRETTY_FUNCTION__ << "): "
#define ERR  std::cerr << "PE " << comm.rank() << " ERROR @ " << __FILE__ ":" << __LINE__ << " (" << __PRETTY_FUNCTION__ << "): "
#define WARN std::cerr << "PE " << comm.rank() << " WARNING @ " << __FILE__ ":" << __LINE__ << " (" << __PRETTY_FUNCTION__ << "): "
// Sequential versions (no communicator)
#define SLOG  if (debug) std::cout
#define SINFO if (debug) std::cout << "INFO @ " << __FILE__ ":" << __LINE__ << " (" << __PRETTY_FUNCTION__ << "): "
#define SERR  std::cerr << "ERROR @ " << __FILE__ ":" << __LINE__ << " (" << __PRETTY_FUNCTION__ << "): "
#define SWARN std::cerr << "WARNING @ " << __FILE__ ":" << __LINE__ << " (" << __PRETTY_FUNCTION__ << "): "
