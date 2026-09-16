// CPU-only, byte-exact row gather. Persistent sleeping workers, no Python calls.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

class Gather {
    const uint8_t *weights_, *scales_;
    int64_t rows_;
    std::mutex call_, mutex_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
    uint64_t generation_ = 0;
    size_t remaining_ = 0;
    const int64_t *ids_ = nullptr;
    uint8_t *output_ = nullptr;
    size_t count_ = 0;
    std::atomic<size_t> next_{0};

    void copy(size_t i) {
        const auto row = ids_[i];
        std::memcpy(output_ + i * 264, weights_ + row * 256, 256);
        std::memcpy(output_ + i * 264 + 256, scales_ + row * 8, 8);
    }
    void work() {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            ready_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            lock.unlock();
            for (size_t i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < count_;)
                copy(i);
            lock.lock();
            if (--remaining_ == 0) done_.notify_one();
        }
    }
public:
    Gather(const uint8_t *w, const uint8_t *s, int64_t rows, int threads)
        : weights_(w), scales_(s), rows_(rows) {
        // One-worker mode runs directly on the calling thread, still without GIL.
        try {
            if (threads > 1)
                for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { work(); });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
            ready_.notify_all();
            for (auto &thread : workers_) thread.join();
            throw;
        }
    }
    ~Gather() {
        // Python wrapper serializes destruction against calls.
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        ready_.notify_all();
        for (auto &thread : workers_) thread.join();
    }
    int run(const int64_t *ids, size_t count, uint8_t *output) {
        std::lock_guard<std::mutex> call(call_);
        // Check every ID before launching workers or writing any output.
        for (size_t i = 0; i < count; ++i)
            if (ids[i] < 0 || ids[i] >= rows_) return 1;
        std::unique_lock<std::mutex> lock(mutex_);
        ids_ = ids; output_ = output; count_ = count;
        if (workers_.empty() || count <= 1) {
            for (size_t i = 0; i < count; ++i) copy(i);
            return 0;
        }
        next_.store(0, std::memory_order_relaxed);
        remaining_ = workers_.size();
        ++generation_;
        ready_.notify_all();
        done_.wait(lock, [&] { return remaining_ == 0; });
        return 0;
    }
};

extern "C" {
void *engram_gather_create(const uint8_t *w, const uint8_t *s, int64_t rows, int threads) noexcept {
    if (!w || !s || rows < 1 || threads < 1 || threads > 128) return nullptr;
    try { return new Gather(w, s, rows, threads); } catch (...) { return nullptr; }
}
int engram_gather_run(void *ctx, const int64_t *ids, size_t count, uint8_t *output) noexcept {
    if (!ctx || (!ids && count) || (!output && count)) return 2;
    try { return static_cast<Gather *>(ctx)->run(ids, count, output); } catch (...) { return 3; }
}
void engram_gather_destroy(void *ctx) noexcept { delete static_cast<Gather *>(ctx); }
}
