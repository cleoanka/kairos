// shm_lob.hpp — POSIX shared-memory segment (mmap MAP_SHARED) that hosts the
// SPSC lock-free ring of LobSnapshot. The C++ producer and the Python consumer
// map the SAME physical pages (Apple-Silicon unified memory), so the whole
// pipeline is genuinely zero-copy: no serialization, no memcpy across the IPC
// boundary, just shared bytes viewed through the buffer protocol.
//
// Project AURA / LOB-Core, Component C.
//
// CONSTITUTION:
//   * Rule 1: the only memcpy in this file is a ONE-TIME bulk zero-fill at
//     segment creation, explicitly waived with 'soul:allow-memcpy'. The hot
//     path (push/pop) does single-struct stores via the ring, never memcpy.
//   * Rule 2: no technical-analysis identifiers.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "lockfree_ring.hpp"

namespace lob {

// --- Data contract (must match src/python/lob_core/schema.py byte-for-byte) --
//
// N_LEVELS = 10 price levels per side. One snapshot is exactly 64 doubles, laid
// out in this precise column order so a (n, 64) float64 numpy view over the
// shared pages reproduces schema.column_names() position-for-position:
//
//   index 0      : ts
//   index 1      : mid
//   index 2..11  : bid_px_0..9
//   index 12..21 : bid_sz_0..9
//   index 22..31 : ask_px_0..9
//   index 32..41 : ask_sz_0..9
//   index 42..51 : bid_cxl_0..9
//   index 52..61 : ask_cxl_0..9
//   index 62     : trade_buy
//   index 63     : trade_sell
//   index 64     : trade_n
//   index 65     : regime       (eval-only; carried for completeness, never a target)
//
// The POD width matches the full schema.column_names() contract exactly:
// 2 + 40 depth + 20 cxl + 4 (trade_buy, trade_sell, trade_n, regime) = 66 doubles.
// A (n, 66) float64 view over the shared pages reproduces the contract
// position-for-position, so Python runs schema.featurize_from_raw() over it with
// zero copy (only the derived feature matrix is allocated).

inline constexpr std::size_t N_LEVELS = 10;
inline constexpr std::size_t N_DOUBLES = 66;  // doubles per snapshot (POD width)

// Exactly 66 contiguous doubles. We use a union so the same storage is reachable
// either as named microstructure fields (C++ ergonomics) or as a flat raw[66]
// (the buffer-protocol view Python wraps). Both members alias the same bytes;
// the struct is a trivial, standard-layout POD with no padding (8-byte aligned,
// 66*8 = 528 bytes).
struct LobSnapshot {
    union {
        struct {
            double ts;                     // 1
            double mid;                    // 1
            double bid_px[N_LEVELS];       // 10
            double bid_sz[N_LEVELS];       // 10
            double ask_px[N_LEVELS];       // 10
            double ask_sz[N_LEVELS];       // 10
            double bid_cxl[N_LEVELS];      // 10
            double ask_cxl[N_LEVELS];      // 10
            double trade_buy;              // 1
            double trade_sell;             // 1
            double trade_n;                // 1
            double regime;                 // 1  -> total = 2 + 60 + 4 = 66
        } f;
        double raw[N_DOUBLES];             // flat alias for the buffer protocol
    };
};

static_assert(sizeof(LobSnapshot) == N_DOUBLES * sizeof(double),
              "LobSnapshot must be exactly 66 contiguous doubles (no padding)");
static_assert(sizeof(double) == 8, "ABI assumes IEEE-754 64-bit double");

// Convenience alias: the concrete ring specialised for LobSnapshot.
using SnapshotRing = RingControl<LobSnapshot>;

// --- Shared-memory segment manager ------------------------------------------
//
// Lifecycle:
//   ShmLob seg; seg.create("aura_lob", 4096);   // producer: shm_open O_CREAT
//   ShmLob seg; seg.attach("aura_lob");          // consumer: shm_open existing
//   seg.detach();                                 // munmap (+ shm_unlink if creator)
//
// `name` is a POSIX shared-memory object name (a leading '/' is added if absent).
class ShmLob {
public:
    ShmLob() = default;
    ~ShmLob();

    ShmLob(const ShmLob&) = delete;
    ShmLob& operator=(const ShmLob&) = delete;

    // Create a NEW segment sized for `capacity` snapshots (rounded up to a power
    // of two) and initialise the ring control block. Throws std::runtime_error
    // on failure. Returns the ring pointer (lives on the shared pages).
    SnapshotRing* create(const std::string& name, std::size_t capacity);

    // Attach to an EXISTING segment created by another process. Does NOT
    // re-initialise the control block. Throws if the segment is missing.
    SnapshotRing* attach(const std::string& name);

    // Unmap (and, if we created it, unlink) the segment.
    void detach();

    SnapshotRing* ring() const noexcept { return ring_; }
    void* base() const noexcept { return base_; }
    std::size_t bytes() const noexcept { return bytes_; }
    std::size_t capacity() const noexcept { return capacity_; }
    bool is_creator() const noexcept { return creator_; }

private:
    SnapshotRing* map_existing(const std::string& name, bool creator,
                               std::size_t capacity);

    std::string name_;
    void* base_ = nullptr;       // mmap base of the whole segment
    SnapshotRing* ring_ = nullptr;  // == base_, typed
    std::size_t bytes_ = 0;
    std::size_t capacity_ = 0;
    int fd_ = -1;
    bool creator_ = false;
};

// Round up to the next power of two (>= 1).
std::size_t next_pow2(std::size_t v) noexcept;

}  // namespace lob
