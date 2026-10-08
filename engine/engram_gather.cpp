// CPU-only, byte-exact row gather. Persistent sleeping workers, no Python calls.
//
// Two sources. Mapped: memcpy out of the memmapped tables, so page faults do the I/O through the
// page cache. Direct: pread of the 512-byte sectors covering each row from an O_DIRECT descriptor
// into per-worker aligned scratch -- no page cache and no mapping (see engine/engram.py).
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
constexpr int64_t SECTOR = 512;
constexpr size_t SCRATCH = 4096;  // >= the widest span (256 bytes over two sectors), page aligned

uint8_t *aligned_scratch() {
    void *p = nullptr;
    return posix_memalign(&p, SCRATCH, SCRATCH) == 0 ? static_cast<uint8_t *>(p) : nullptr;
}

// Copies bytes [off, off + len) of fd into dst through the covering sectors.
bool read_span(int fd, int64_t off, size_t len, uint8_t *dst, uint8_t *scratch) {
    const int64_t start = off & ~(SECTOR - 1);
    const size_t want = static_cast<size_t>(((off + static_cast<int64_t>(len) + SECTOR - 1) & ~(SECTOR - 1)) - start);
    const size_t need = static_cast<size_t>(off - start) + len;
    size_t done = 0;
    while (done < need) {
        const ssize_t got = pread(fd, scratch + done, want - done, start + static_cast<int64_t>(done));
        if (got < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (got == 0) return false;  // end of file before the row
        done += static_cast<size_t>(got);
    }
    std::memcpy(dst, scratch + (off - start), len);
    return true;
}
}  // namespace

class Gather {
    const uint8_t *weights_ = nullptr, *scales_ = nullptr;
    int fd_ = -1;
    int64_t w_off_ = 0, s_off_ = 0;
    int64_t rows_;
    std::mutex call_, mutex_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> workers_;
    std::vector<uint8_t *> scratch_;  // [0] is the calling thread's, [1 + k] worker k's
    bool stop_ = false;
    uint64_t generation_ = 0;
    size_t remaining_ = 0;
    const int64_t *ids_ = nullptr;
    uint8_t *output_ = nullptr;
    size_t count_ = 0;
    std::atomic<size_t> next_{0};
    std::atomic<bool> failed_{false};

    void copy(size_t i, uint8_t *scratch) {
        const auto row = ids_[i];
        uint8_t *out = output_ + i * 264;
        if (fd_ < 0) {
            std::memcpy(out, weights_ + row * 256, 256);
            std::memcpy(out + 256, scales_ + row * 8, 8);
        } else if (!read_span(fd_, w_off_ + row * 256, 256, out, scratch) ||
                   !read_span(fd_, s_off_ + row * 8, 8, out + 256, scratch)) {
            failed_.store(true, std::memory_order_relaxed);
        }
    }
    void work(uint8_t *scratch) {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            ready_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            lock.unlock();
            for (size_t i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < count_;)
                copy(i, scratch);
            lock.lock();
            if (--remaining_ == 0) done_.notify_one();
        }
    }
    void start(int threads) {
        // One-worker mode runs directly on the calling thread, still without GIL.
        try {
            const int n = threads > 1 ? threads : 0;
            for (int i = 0; i <= n; ++i) {
                uint8_t *p = fd_ < 0 ? nullptr : aligned_scratch();
                if (fd_ >= 0 && !p) throw std::bad_alloc();
                scratch_.push_back(p);
            }
            for (int i = 0; i < n; ++i) workers_.emplace_back([this, i] { work(scratch_[1 + i]); });
        } catch (...) {
            shutdown();
            throw;
        }
    }
    void shutdown() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        ready_.notify_all();
        for (auto &thread : workers_) thread.join();
        workers_.clear();
        for (auto *p : scratch_) std::free(p);
        scratch_.clear();
    }
public:
    Gather(const uint8_t *w, const uint8_t *s, int64_t rows, int threads)
        : weights_(w), scales_(s), rows_(rows) { start(threads); }
    Gather(int fd, int64_t w_off, int64_t s_off, int64_t rows, int threads)
        : fd_(fd), w_off_(w_off), s_off_(s_off), rows_(rows) { start(threads); }
    // Python wrapper serializes destruction against calls.
    ~Gather() { shutdown(); }
    int run(const int64_t *ids, size_t count, uint8_t *output) {
        std::lock_guard<std::mutex> call(call_);
        // Check every ID before launching workers or writing any output.
        for (size_t i = 0; i < count; ++i)
            if (ids[i] < 0 || ids[i] >= rows_) return 1;
        std::unique_lock<std::mutex> lock(mutex_);
        ids_ = ids; output_ = output; count_ = count;
        failed_.store(false, std::memory_order_relaxed);
        if (workers_.empty() || count <= 1) {
            for (size_t i = 0; i < count; ++i) copy(i, scratch_[0]);
        } else {
            next_.store(0, std::memory_order_relaxed);
            remaining_ = workers_.size();
            ++generation_;
            ready_.notify_all();
            done_.wait(lock, [&] { return remaining_ == 0; });
        }
        return failed_.load(std::memory_order_relaxed) ? 4 : 0;
    }
};

extern "C" {
void *engram_gather_create(const uint8_t *w, const uint8_t *s, int64_t rows, int threads) noexcept {
    if (!w || !s || rows < 1 || threads < 1 || threads > 128) return nullptr;
    try { return new Gather(w, s, rows, threads); } catch (...) { return nullptr; }
}
void *engram_gather_create_direct(int fd, int64_t w_off, int64_t s_off, int64_t rows, int threads) noexcept {
    if (fd < 0 || w_off < 0 || s_off < 0 || rows < 1 || threads < 1 || threads > 128) return nullptr;
    try { return new Gather(fd, w_off, s_off, rows, threads); } catch (...) { return nullptr; }
}
int engram_gather_run(void *ctx, const int64_t *ids, size_t count, uint8_t *output) noexcept {
    if (!ctx || (!ids && count) || (!output && count)) return 2;
    try { return static_cast<Gather *>(ctx)->run(ids, count, output); } catch (...) { return 3; }
}
void engram_gather_destroy(void *ctx) noexcept { delete static_cast<Gather *>(ctx); }
}
