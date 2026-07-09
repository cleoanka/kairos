// producer_demo.cpp — standalone C++ producer for the LOB-Core zero-copy ring.
//
// Project AURA / LOB-Core, Component C. Creates (or attaches to) the shared
// memory ring and pushes synthetic 66-double LobSnapshots so the Python red-team
// stress harness can consume them over the SAME shared pages (true zero-copy).
//
// Usage:
//   producer_demo [name] [capacity] [n_snapshots] [--attach]
//   e.g.  ./producer_demo aura_lob 4096 1000000
//
// PERFORMANCE-CORE PINNING (Apple Silicon, macOS):
//   macOS does not expose hard CPU affinity to a specific core index the way
//   Linux's sched_setaffinity does. The supported levers are:
//     1. pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
//        -> tells the scheduler to prefer the P-cluster (performance cores) and
//           run at the highest non-realtime QoS. This is the primary, reliable
//           knob for steering a thread onto P-cores on M-series silicon.
//     2. thread_policy_set(THREAD_AFFINITY_POLICY) with a non-zero affinity tag
//        -> a *hint* that threads sharing a tag should share an L2; on Apple
//           Silicon affinity tags are largely a no-op but we set it anyway as a
//           portable hint and to document intent.
//   Together these bias the busy producer thread onto a performance core.
//
// CONSTITUTION:
//   * Rule 1: pushes are single-struct in-place stores via the ring; no memcpy.
//   * Rule 2: no technical-analysis identifiers.

#include <pthread.h>
#include <pthread/qos.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <mach/mach.h>
#include <mach/thread_policy.h>

#include "shm_lob.hpp"

namespace {

// Steer the calling thread onto the performance cluster (see header comment).
void pin_to_performance_cores(int affinity_tag) {
    // (1) Highest interactive QoS — the reliable P-core bias on macOS.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    // (2) Affinity-tag hint via the Mach thread policy API. Best-effort: on
    // Apple Silicon this is advisory and may be ignored, hence we don't fail.
    thread_affinity_policy_data_t policy = {affinity_tag};
    mach_port_t mt = pthread_mach_thread_np(pthread_self());
    kern_return_t kr = thread_policy_set(
        mt, THREAD_AFFINITY_POLICY,
        reinterpret_cast<thread_policy_t>(&policy), THREAD_AFFINITY_POLICY_COUNT);
    if (kr != KERN_SUCCESS) {
        // Advisory only — log and continue.
        std::fprintf(stderr,
                     "[producer] affinity hint not applied (kr=%d) — QoS still "
                     "biases P-cores.\n",
                     kr);
    }
}

// Fill one synthetic snapshot with a plausible 10-level book around `mid`.
void make_synthetic(lob::LobSnapshot& s, double ts, double mid) {
    s.f.ts = ts;
    s.f.mid = mid;
    for (std::size_t i = 0; i < lob::N_LEVELS; ++i) {
        const double off = static_cast<double>(i + 1);  // tick offset from mid
        s.f.bid_px[i] = off;
        s.f.ask_px[i] = off;
        s.f.bid_sz[i] = 100.0 - off * 5.0;
        s.f.ask_sz[i] = 100.0 - off * 4.0;
        s.f.bid_cxl[i] = std::fmod(ts + off, 13.0);
        s.f.ask_cxl[i] = std::fmod(ts + off, 11.0);
    }
    s.f.trade_buy = std::fmod(ts, 5.0);
    s.f.trade_sell = std::fmod(ts, 3.0);
    s.f.trade_n = std::fmod(ts, 4.0);
    s.f.regime = 0.0;  // eval-only; producer leaves it neutral
}

}  // namespace

int main(int argc, char** argv) {
    std::string name = (argc > 1) ? argv[1] : "aura_lob";
    std::size_t capacity = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 4096;
    std::size_t n = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 1000000;
    bool do_attach = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--attach") == 0) do_attach = true;
    }

    pin_to_performance_cores(/*affinity_tag=*/1);

    lob::ShmLob seg;
    lob::SnapshotRing* ring = nullptr;
    try {
        ring = do_attach ? seg.attach(name) : seg.create(name, capacity);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[producer] segment error: %s\n", e.what());
        return 1;
    }

    std::printf(
        "[producer] %s '%s' cap=%zu bytes=%zu ring@%p slots@%p — pushing %zu "
        "snapshots\n",
        do_attach ? "attached" : "created", name.c_str(), seg.capacity(),
        seg.bytes(), static_cast<void*>(ring),
        static_cast<void*>(ring->data()), n);

    lob::LobSnapshot snap{};
    std::size_t pushed = 0, backoff = 0;
    for (std::size_t i = 0; i < n; ++i) {
        make_synthetic(snap, static_cast<double>(i), 100.0 + (i % 17) * 0.01);
        // SPSC: if the consumer is slow, the ring fills; spin briefly then retry.
        while (!ring->push(snap)) {
            ++backoff;
            std::this_thread::yield();
        }
        ++pushed;
    }

    std::printf("[producer] done: pushed=%zu backoff_spins=%zu\n", pushed,
                backoff);
    // Keep the segment mapped a moment so a consumer can finish draining.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return 0;
}
