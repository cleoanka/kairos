// bench_rings.cpp — MCTS-style synthesis of SPSC ring variants (Component C).
//
// Synthesises THREE lock-free single-producer/single-consumer ring designs that
// differ only in cache-line padding of the indices and in memory ordering,
// benchmarks each with a producer/consumer thread pair, and declares the fastest
// the "champion". This empirically validates the production ring's design choices
// (cache-line-padded head/tail + acquire/release) — the spec's Part IV Milestone 1
// ("write 3 implementations, benchmark, pick the fastest").
//
// Standalone and NOT wired into production; the verified production ring lives in
// lockfree_ring.hpp. CONSTITUTION: no memcpy on the hot path (slot writes are
// single-struct stores); no technical-analysis identifiers.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include <pthread.h>
#include <pthread/qos.h>

namespace {

// Bias the calling thread onto the performance cluster — without this the
// producer/consumer can land on P+E cores and the comparison becomes scheduler
// noise rather than a measure of the ring design.
void pin_to_p_cores() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

struct Item { double v[8]; };   // 64-byte payload

// One SPSC ring, parameterised by index alignment and the acquire/release-style
// memory order used to publish/observe the other thread's index.
template <std::size_t ALIGN, std::memory_order LOAD, std::memory_order STORE>
struct Ring {
    static constexpr std::size_t CAP = 1u << 12;   // 4096, power of two
    static constexpr std::size_t MASK = CAP - 1;
    Item buf[CAP];
    alignas(ALIGN) std::atomic<std::size_t> head{0};
    alignas(ALIGN) std::atomic<std::size_t> tail{0};

    bool push(const Item& x) {
        const std::size_t t = tail.load(std::memory_order_relaxed);
        const std::size_t next = t + 1;
        if (next - head.load(LOAD) > CAP) return false;   // back-pressure when full
        buf[t & MASK] = x;                                 // single-struct store
        tail.store(next, STORE);
        return true;
    }
    bool pop(Item& x) {
        const std::size_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(LOAD)) return false;            // empty
        x = buf[h & MASK];
        head.store(h + 1, STORE);
        return true;
    }
};

template <class R>
double bench_throughput(std::size_t total) {
    R ring;
    const auto t0 = std::chrono::steady_clock::now();
    std::thread producer([&] {
        pin_to_p_cores();
        Item x{};
        for (std::size_t i = 0; i < total;) {
            x.v[0] = static_cast<double>(i);
            if (ring.push(x)) ++i;
        }
    });
    std::thread consumer([&] {
        pin_to_p_cores();
        Item x{};
        for (std::size_t got = 0; got < total;) {
            if (ring.pop(x)) ++got;
        }
    });
    producer.join();
    consumer.join();
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    return secs > 0 ? static_cast<double>(total) / secs : 0.0;
}

template <class R>
double best_of(std::size_t total, int rounds) {
    double best = 0.0;
    for (int r = 0; r < rounds; ++r) best = std::max(best, bench_throughput<R>(total));
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t total = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 20000000ULL;
    const int rounds = 9;

    // Apple-Silicon cache lines are 128 bytes, so alignas(64) may NOT separate the
    // two indices onto distinct lines — V4 tests true 128-byte separation.
    using V1 = Ring<64, std::memory_order_acquire, std::memory_order_release>;
    using V2 = Ring<8, std::memory_order_acquire, std::memory_order_release>;
    using V3 = Ring<64, std::memory_order_seq_cst, std::memory_order_seq_cst>;
    using V4 = Ring<128, std::memory_order_acquire, std::memory_order_release>;

    struct Res { const char* name; double mps; };
    Res res[4] = {
        {"pad64  + acquire/release   (current production)", best_of<V1>(total, rounds) / 1e6},
        {"unpadded + acquire/release (false-share)", best_of<V2>(total, rounds) / 1e6},
        {"pad64  + seq_cst           (strong order)", best_of<V3>(total, rounds) / 1e6},
        {"pad128 + acquire/release   (true line split)", best_of<V4>(total, rounds) / 1e6},
    };

    std::printf("MCTS ring synthesis — SPSC throughput (P-core pinned, best of %d, %zu items)\n",
                rounds, total);
    int champ = 0;
    for (int i = 0; i < 4; ++i) {
        std::printf("  %-46s %8.1f M items/s\n", res[i].name, res[i].mps);
        if (res[i].mps > res[champ].mps) champ = i;
    }
    std::printf("CHAMPION: %s  (%.1f M items/s)\n", res[champ].name, res[champ].mps);
    return res[champ].mps > 0.0 ? 0 : 1;
}
