// shm_lob.cpp — implementation of the mmap-backed shared-memory LOB segment.
//
// Project AURA / LOB-Core, Component C.
//
// CONSTITUTION:
//   * Rule 1: the sole bulk copy is a one-time zero-fill of fresh pages at
//     creation, waived inline with 'soul:allow-memcpy'. There is NO memcpy on
//     the push/pop hot path.
//   * Rule 2: no technical-analysis identifiers.

#include "shm_lob.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace lob {

std::size_t next_pow2(std::size_t v) noexcept {
    if (v < 2) return 1;
    --v;
    for (std::size_t i = 1; i < sizeof(std::size_t) * 8; i <<= 1) {
        v |= v >> i;
    }
    return v + 1;
}

static std::string normalize_name(const std::string& name) {
    // POSIX shared-memory names must start with a single '/'.
    if (!name.empty() && name.front() == '/') return name;
    return "/" + name;
}

ShmLob::~ShmLob() { detach(); }

SnapshotRing* ShmLob::map_existing(const std::string& name, bool creator,
                                   std::size_t capacity) {
    name_ = normalize_name(name);
    creator_ = creator;

    const int oflag = creator ? (O_CREAT | O_RDWR) : O_RDWR;
    fd_ = ::shm_open(name_.c_str(), oflag, 0600);
    if (fd_ < 0) {
        throw std::runtime_error("shm_open('" + name_ + "') failed: " +
                                 std::strerror(errno));
    }

    if (creator) {
        capacity_ = next_pow2(capacity);
        bytes_ = SnapshotRing::bytes_for(capacity_);
        if (::ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) {
            const int e = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(std::string("ftruncate failed: ") +
                                     std::strerror(e));
        }
    } else {
        // Discover the existing size via fstat so we don't need the capacity.
        struct stat st {};
        if (::fstat(fd_, &st) != 0) {
            const int e = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(std::string("fstat failed: ") +
                                     std::strerror(e));
        }
        bytes_ = static_cast<std::size_t>(st.st_size);
    }

    base_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (base_ == MAP_FAILED) {
        const int e = errno;
        ::close(fd_);
        fd_ = -1;
        base_ = nullptr;
        throw std::runtime_error(std::string("mmap(MAP_SHARED) failed: ") +
                                 std::strerror(e));
    }

    // The ring control block is overlaid on the front of the mapped region; the
    // snapshot slots follow it in the same contiguous mapping (true shared pages).
    ring_ = reinterpret_cast<SnapshotRing*>(base_);

    if (creator) {
        // One-time bulk zero-fill of brand-new pages so the control block and
        // all slots start clean. This is initialisation, not a hot-path copy.
        std::memset(base_, 0, bytes_);  // soul:allow-memcpy (one-time bulk init)
        // Placement-init the atomics/capacity in the shared header.
        ring_->init(capacity_);
    } else {
        capacity_ = ring_->capacity_;
    }
    return ring_;
}

SnapshotRing* ShmLob::create(const std::string& name, std::size_t capacity) {
    return map_existing(name, /*creator=*/true, capacity);
}

SnapshotRing* ShmLob::attach(const std::string& name) {
    return map_existing(name, /*creator=*/false, /*capacity=*/0);
}

void ShmLob::detach() {
    if (base_ && base_ != MAP_FAILED) {
        ::munmap(base_, bytes_);
    }
    base_ = nullptr;
    ring_ = nullptr;
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    if (creator_ && !name_.empty()) {
        // The creator owns the lifetime of the named object.
        ::shm_unlink(name_.c_str());
    }
    creator_ = false;
    bytes_ = 0;
    capacity_ = 0;
}

}  // namespace lob
