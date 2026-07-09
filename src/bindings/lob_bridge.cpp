// lob_bridge.cpp — pybind11 module 'lob_bridge'.
//
// Project AURA / LOB-Core, Component C: the zero-copy IPC bridge between the
// C++ producer (matching engine) and the Python consumer (MLX featurizer).
//
// The module attaches to the mmap MAP_SHARED ring (shm_lob) and exposes the
// snapshot storage to Python via the BUFFER PROTOCOL (py::buffer_info). The
// returned memoryview is a (capacity, 66) float64 window directly over the
// shared pages — np.frombuffer / np.asarray wrap it WITHOUT copying.
//
// ZERO-COPY PROOF (also asserted in the Python check):
//   The buffer_info.ptr we hand to Python equals ring->data() (the slot base in
//   the shared mapping). So:
//       np.asarray(memoryview).ctypes.data  ==  bridge.base_addr()
//   i.e. the numpy array's data pointer is literally the mmap'd shared page
//   address. No bytes are copied; the producer's writes appear in numpy live.
//
// CONSTITUTION:
//   * Rule 1: no memcpy on the hot path. push_snapshot writes one struct via the
//     ring's in-place store; latest_view() returns a pointer view (no copy).
//   * Rule 2: no technical-analysis identifiers.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../cpp/shm_lob.hpp"

namespace py = pybind11;

namespace {

// One module-global segment handle. A given Python process is either the
// creator (test producer) or an attacher (consumer); either way it holds one.
class Bridge {
public:
    void create(const std::string& name, std::size_t capacity) {
        seg_ = std::make_unique<lob::ShmLob>();
        ring_ = seg_->create(name, capacity);
    }

    void attach(const std::string& name) {
        seg_ = std::make_unique<lob::ShmLob>();
        ring_ = seg_->attach(name);
    }

    void detach() {
        ring_ = nullptr;
        seg_.reset();
    }

    lob::SnapshotRing* ring() {
        if (!ring_) throw std::runtime_error("bridge not create()/attach()-ed");
        return ring_;
    }

    lob::ShmLob* seg() {
        if (!seg_) throw std::runtime_error("bridge not create()/attach()-ed");
        return seg_.get();
    }

private:
    std::unique_ptr<lob::ShmLob> seg_;
    lob::SnapshotRing* ring_ = nullptr;
};

// Push one snapshot from a Python sequence/array of exactly N_DOUBLES values.
// We accept a contiguous float64 buffer (zero-copy read of the *input*) or any
// iterable of 66 numbers. Returns false if the ring is full.
bool push_snapshot(Bridge& b, py::object obj) {
    lob::LobSnapshot snap{};

    // Fast path: a contiguous float64 numpy array — read its raw doubles.
    if (py::isinstance<py::array>(obj)) {
        auto arr = py::cast<py::array_t<double, py::array::c_style |
                                                   py::array::forcecast>>(obj);
        if (static_cast<std::size_t>(arr.size()) != lob::N_DOUBLES) {
            throw std::runtime_error("snapshot must have exactly 66 doubles");
        }
        const double* src = arr.data();
        for (std::size_t i = 0; i < lob::N_DOUBLES; ++i) snap.raw[i] = src[i];
    } else {
        // Generic path: any iterable of numbers.
        auto seq = py::cast<std::vector<double>>(obj);
        if (seq.size() != lob::N_DOUBLES) {
            throw std::runtime_error("snapshot must have exactly 66 doubles");
        }
        for (std::size_t i = 0; i < lob::N_DOUBLES; ++i) snap.raw[i] = seq[i];
    }
    return b.ring()->push(snap);
}

// Return a (capacity, 66) float64 memoryview directly over the shared slot
// region. This is the zero-copy window: numpy wraps it without copying.
py::memoryview latest_view(Bridge& b) {
    lob::SnapshotRing* r = b.ring();
    double* base = reinterpret_cast<double*>(r->data());
    const std::size_t rows = r->capacity_;
    const std::size_t cols = lob::N_DOUBLES;

    // 2-D buffer: shape (rows, 64), C-contiguous double strides.
    return py::memoryview::from_buffer(
        base,                                  // pointer into shared pages
        {static_cast<py::ssize_t>(rows),       // shape
         static_cast<py::ssize_t>(cols)},
        {static_cast<py::ssize_t>(cols * sizeof(double)),  // strides
         static_cast<py::ssize_t>(sizeof(double))});
}

// Drain helper: pop up to `max_n` snapshots and return how many were consumed.
// (Used by the red-team stress harness to verify producer/consumer liveness.)
std::size_t drain(Bridge& b, std::size_t max_n) {
    lob::SnapshotRing* r = b.ring();
    lob::LobSnapshot tmp;
    std::size_t n = 0;
    while (n < max_n && r->pop(tmp)) ++n;
    return n;
}

// Raw base address of the shared slot region (for the zero-copy address proof).
std::uintptr_t base_addr(Bridge& b) {
    return reinterpret_cast<std::uintptr_t>(b.ring()->data());
}

std::size_t ring_size(Bridge& b) { return b.ring()->size(); }
std::size_t ring_capacity(Bridge& b) { return b.ring()->capacity_; }

// Benchmark helper: hammer push() with synthetic snapshots and report
// throughput. The consumer (Python or producer_demo) must be draining or the
// ring fills; we count successful pushes for a fixed iteration budget.
py::dict benchmark(Bridge& b, std::size_t iters) {
    lob::SnapshotRing* r = b.ring();
    lob::LobSnapshot snap{};
    std::size_t ok = 0, full = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
        snap.f.ts = static_cast<double>(i);
        snap.f.mid = 100.0 + static_cast<double>(i % 7);
        if (r->push(snap)) {
            ++ok;
        } else {
            ++full;
            // Self-drain one so a single-process bench can make progress.
            lob::LobSnapshot tmp;
            r->pop(tmp);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    py::dict d;
    d["iters"] = iters;
    d["pushed"] = ok;
    d["full_events"] = full;
    d["seconds"] = secs;
    d["pushes_per_sec"] = secs > 0 ? static_cast<double>(ok) / secs : 0.0;
    return d;
}

}  // namespace

PYBIND11_MODULE(lob_bridge, m) {
    m.doc() =
        "LOB-Core zero-copy IPC bridge (Component C): mmap MAP_SHARED SPSC ring "
        "of 66-double LobSnapshots, exposed to Python via the buffer protocol.";

    m.attr("N_DOUBLES") = py::int_(static_cast<int>(lob::N_DOUBLES));
    m.attr("N_LEVELS") = py::int_(static_cast<int>(lob::N_LEVELS));

    py::class_<Bridge>(m, "Bridge")
        .def(py::init<>())
        .def("create", &Bridge::create, py::arg("name"), py::arg("capacity"),
             "Create + initialise a new shared-memory ring.")
        .def("attach", &Bridge::attach, py::arg("name"),
             "Attach to an existing shared-memory ring.")
        .def("detach", &Bridge::detach)
        .def("push_snapshot", &push_snapshot, py::arg("snapshot"),
             "Push one 66-double snapshot; returns False if the ring is full.")
        .def("latest_view", &latest_view,
             "Zero-copy (capacity, 66) float64 memoryview over the shared pages.")
        .def("drain", &drain, py::arg("max_n") = 1024,
             "Pop up to max_n snapshots; returns the number consumed.")
        .def("size", &ring_size, "Number of unread snapshots in the ring.")
        .def("capacity", &ring_capacity)
        .def("base_addr", &base_addr,
             "Raw address of the shared slot base (zero-copy proof).")
        .def("benchmark", &benchmark, py::arg("iters") = 1000000,
             "Push throughput micro-benchmark.");

    // Module-level convenience constructors so callers can do:
    //   b = lob_bridge.create('aura_lob', 4096)
    m.def("create",
          [](const std::string& name, std::size_t capacity) {
              auto b = std::make_unique<Bridge>();
              b->create(name, capacity);
              return b;
          },
          py::arg("name"), py::arg("capacity"));
    m.def("attach",
          [](const std::string& name) {
              auto b = std::make_unique<Bridge>();
              b->attach(name);
              return b;
          },
          py::arg("name"));
}
