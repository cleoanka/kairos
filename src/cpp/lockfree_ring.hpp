// lockfree_ring.hpp — header-only SPSC lock-free ring buffer of LobSnapshot.
//
// Project AURA / LOB-Core, Component C (zero-copy IPC bridge).
//
// Design:
//   * Single-Producer / Single-Consumer only. This is the ONLY contract under
//     which the acquire/release scheme below is correct without a mutex.
//   * Power-of-two capacity so the modulo reduces to a bitmask (& mask).
//   * std::atomic<size_t> head_ (consumer index) and tail_ (producer index)
//     with acquire/release ordering — no locks, no CAS loops needed for SPSC.
//   * The buffer itself is a flexible storage region that lives *inside* the
//     shared-memory segment (see shm_lob.hpp). This header only defines the
//     control structure that is overlaid on those shared pages, so the producer
//     (C++) and consumer (Python via pybind11) operate on the SAME bytes.
//
// CONSTITUTION:
//   * Rule 1 (no memcpy on hot path): push()/pop() write/read a single struct by
//     direct assignment into a pre-allocated slot — a placement, not a bulk
//     memcpy. The Python side never copies either: it wraps the slot region with
//     the buffer protocol (np.frombuffer over the same physical pages).
//   * Rule 2 (no TA identifiers): none used.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

namespace lob {

// Forward-declared here, fully defined in shm_lob.hpp. The ring is generic over
// the POD slot type so the control header can be unit-tested independently.
struct LobSnapshot;

// Cache line size on Apple Silicon (M-series): 128 bytes. We pad head/tail onto
// separate lines to avoid false sharing between producer and consumer cores.
inline constexpr std::size_t kCacheLine = 128;

// A control block that is meant to be *overlaid* (placement-constructed) on the
// front of a shared-memory segment. It contains the atomic indices and the
// capacity; the snapshot slots follow immediately after, in the same segment.
//
// Layout (one contiguous mmap region):
//   [ RingHeader ][ slot 0 ][ slot 1 ] ... [ slot capacity-1 ]
//
// `capacity` MUST be a power of two. `mask = capacity - 1`.
template <typename Slot>
struct alignas(kCacheLine) RingControl {
    // Producer-owned write index. Released by producer, acquired by consumer.
    alignas(kCacheLine) std::atomic<std::size_t> tail_;
    // Consumer-owned read index. Released by consumer, acquired by producer.
    alignas(kCacheLine) std::atomic<std::size_t> head_;
    // Immutable after creation.
    alignas(kCacheLine) std::size_t capacity_;
    std::size_t mask_;

    // One-time initialisation by the CREATOR of the segment only. The attaching
    // side must NOT call this (it would clobber a live producer/consumer).
    void init(std::size_t capacity) noexcept {
        capacity_ = capacity;
        mask_ = capacity - 1;
        tail_.store(0, std::memory_order_relaxed);
        head_.store(0, std::memory_order_relaxed);
    }

    // The slot array begins right after this control block in the same mapping.
    // We compute it via pointer arithmetic so it works on the shared pages.
    Slot* slots() noexcept {
        return reinterpret_cast<Slot*>(reinterpret_cast<std::byte*>(this) +
                                       slots_offset());
    }
    const Slot* slots() const noexcept {
        return reinterpret_cast<const Slot*>(
            reinterpret_cast<const std::byte*>(this) + slots_offset());
    }

    // Offset of the slot array, aligned up to a cache line after the header.
    static constexpr std::size_t slots_offset() noexcept {
        constexpr std::size_t hdr = sizeof(RingControl<Slot>);
        return (hdr + (kCacheLine - 1)) & ~(kCacheLine - 1);
    }

    // Total bytes the whole ring (header + slots) needs for a given capacity.
    static std::size_t bytes_for(std::size_t capacity) noexcept {
        return slots_offset() + capacity * sizeof(Slot);
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    std::size_t size() const noexcept {
        const std::size_t t = tail_.load(std::memory_order_acquire);
        const std::size_t h = head_.load(std::memory_order_acquire);
        return t - h;  // unsigned wrap-around is well defined and correct here.
    }

    // PRODUCER side. Returns false if the ring is full (consumer is behind).
    // No memcpy: we assign the single POD into its pre-allocated slot in place.
    bool push(const Slot& value) noexcept {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        const std::size_t next = t + 1;
        // Full if writing would catch up to the consumer's last-read index.
        if (next - head_.load(std::memory_order_acquire) > capacity_) {
            return false;
        }
        slots()[t & mask_] = value;  // in-place single-struct store (no bulk copy)
        // Publish the new tail; release pairs with consumer's acquire on head/tail.
        tail_.store(next, std::memory_order_release);
        return true;
    }

    // CONSUMER side. Returns false if the ring is empty. `out` receives the slot
    // by direct assignment (single struct, not a bulk buffer copy).
    bool pop(Slot& out) noexcept {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        if (h == tail_.load(std::memory_order_acquire)) {
            return false;  // empty
        }
        out = slots()[h & mask_];  // in-place single-struct load
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    // Zero-copy peek for the buffer-protocol path: returns a pointer to the slot
    // array base. Python wraps this directly — never copies it out.
    Slot* data() noexcept { return slots(); }
    const Slot* data() const noexcept { return slots(); }
};

}  // namespace lob
