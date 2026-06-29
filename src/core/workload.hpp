#pragma once
// =============================================================================
// workload.hpp — Request lifecycle + KV-Cache behavioural model
// =============================================================================
//
// • RequestState / Request   – state-machine for a single inference job.
// • EvictionPolicy hierarchy – pluggable KV-cache eviction strategies.
// • KVCache                  – tracks cache size, hit/miss, and delegates
//                              eviction to the configured policy.
// =============================================================================

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace llmsimbench {

// ═══════════════════════════════════════════════════════════════════════════
//  Request
// ═══════════════════════════════════════════════════════════════════════════

enum class RequestState : uint8_t {
    PENDING,
    IN_PREFILL,
    IN_DECODE,
    PREEMPTED,
    COMPLETED
};

inline std::string request_state_to_string(RequestState s) {
    switch (s) {
        case RequestState::PENDING:    return "PENDING";
        case RequestState::IN_PREFILL: return "IN_PREFILL";
        case RequestState::IN_DECODE:  return "IN_DECODE";
        case RequestState::PREEMPTED:  return "PREEMPTED";
        case RequestState::COMPLETED:  return "COMPLETED";
        default:                       return "UNKNOWN";
    }
}

/// Models a single LLM inference request with full lifecycle tracking.
class Request {
public:
    Request(int id, int prompt_len, int gen_len, double arrival)
        : request_id_(id)
        , prompt_length_(prompt_len)
        , total_generation_length_(gen_len)
        , arrival_time_(arrival)
    {}

    // ── State transitions ────────────────────────────────────────────────
    void start_prefill(double sim_time) {
        assert(state_ == RequestState::PENDING);
        state_ = RequestState::IN_PREFILL;
        prefill_start_time_ = sim_time;
    }

    void transition_to_decode(double sim_time) {
        if (state_ != RequestState::IN_PREFILL) {
            std::cerr << "Assertion failed in transition_to_decode: req_id=" << request_id_ << ", state is " << request_state_to_string(state_) << std::endl;
        }
        assert(state_ == RequestState::IN_PREFILL);
        state_ = RequestState::IN_DECODE;
        prefill_end_time_  = sim_time;
        decode_start_time_ = sim_time;
    }

    void complete(double sim_time) {
        assert(state_ == RequestState::IN_DECODE);
        state_ = RequestState::COMPLETED;
        completion_time_ = sim_time;
    }

    /// Preempt a running request (from IN_PREFILL or IN_DECODE).
    /// Saves current progress so it can be resumed later.
    void preempt(double /*sim_time*/) {
        assert(state_ == RequestState::IN_PREFILL || state_ == RequestState::IN_DECODE);
        preempted_from_decode_ = (state_ == RequestState::IN_DECODE);
        state_ = RequestState::PREEMPTED;
        ++preemption_count_;
    }

    /// Resume a preempted request into the prefill phase.
    void resume_prefill(double sim_time) {
        assert(state_ == RequestState::PREEMPTED);
        state_ = RequestState::IN_PREFILL;
        prefill_start_time_ = sim_time; // restart prefill timing
    }

    /// Resume a preempted request into the decode phase.
    void resume_decode(double sim_time) {
        assert(state_ == RequestState::PREEMPTED);
        state_ = RequestState::IN_DECODE;
        decode_start_time_ = sim_time; // restart decode timing
    }

    /// Increment generated-token counter; returns true when all tokens done.
    bool generate_token() {
        ++tokens_generated_;
        return tokens_generated_ >= total_generation_length_;
    }

    // ── Partial prefill progress (for chunked prefill) ───────────────────
    void set_prefill_tokens_processed(int n) { prefill_tokens_processed_ = n; }
    [[nodiscard]] int  get_prefill_tokens_processed() const noexcept { return prefill_tokens_processed_; }
    [[nodiscard]] int  get_remaining_prefill_tokens() const noexcept {
        return prompt_length_ - prefill_tokens_processed_;
    }

    // ── Queries ──────────────────────────────────────────────────────────
    [[nodiscard]] bool is_completed() const noexcept {
        return state_ == RequestState::COMPLETED;
    }

    [[nodiscard]] int          get_id()               const noexcept { return request_id_; }
    [[nodiscard]] int          get_prompt_length()     const noexcept { return prompt_length_; }
    [[nodiscard]] int          get_generation_length() const noexcept { return total_generation_length_; }
    [[nodiscard]] int          get_tokens_generated()  const noexcept { return tokens_generated_; }
    [[nodiscard]] double       get_arrival_time()      const noexcept { return arrival_time_; }
    [[nodiscard]] RequestState get_state()             const noexcept { return state_; }
    [[nodiscard]] std::string  get_state_str()         const { return request_state_to_string(state_); }
    [[nodiscard]] int          get_preemption_count()   const noexcept { return preemption_count_; }
    [[nodiscard]] bool         was_preempted_from_decode() const noexcept { return preempted_from_decode_; }

    // Latency helpers (valid only after corresponding phase completes)
    [[nodiscard]] double get_prefill_latency() const {
        return prefill_end_time_ - prefill_start_time_;
    }
    [[nodiscard]] double get_decode_latency() const {
        return completion_time_ - decode_start_time_;
    }
    [[nodiscard]] double get_end_to_end_latency() const {
        return completion_time_ - arrival_time_;
    }
    [[nodiscard]] double get_prefill_start()  const noexcept { return prefill_start_time_; }
    [[nodiscard]] double get_decode_start()   const noexcept { return decode_start_time_; }
    [[nodiscard]] double get_completion_time() const noexcept { return completion_time_; }

private:
    int          request_id_;
    int          prompt_length_;
    int          total_generation_length_;
    int          tokens_generated_{0};
    int          prefill_tokens_processed_{0};
    int          preemption_count_{0};
    bool         preempted_from_decode_{false};
    double       arrival_time_;
    RequestState state_{RequestState::PENDING};

    double prefill_start_time_{0.0};
    double prefill_end_time_{0.0};
    double decode_start_time_{0.0};
    double completion_time_{0.0};
};


// ═══════════════════════════════════════════════════════════════════════════
//  KV-Cache Eviction Policies
// ═══════════════════════════════════════════════════════════════════════════

// Forward declaration
class KVCache;

/// Abstract base for cache eviction strategies.
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    /// Remove entries from `cache` until it is within capacity.
    virtual void evict(KVCache& cache) = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

/// Evicts random entries.
class RandomEvictionPolicy : public EvictionPolicy {
public:
    void evict(KVCache& cache) override;
    [[nodiscard]] std::string name() const override { return "Random"; }
};

/// Evicts the least-recently-used entries.
class LRUEvictionPolicy : public EvictionPolicy {
public:
    void evict(KVCache& cache) override;
    [[nodiscard]] std::string name() const override { return "LRU"; }
};

/// Simplified attention-guided eviction: evicts entries with lowest
/// simulated attention score.
class AttentionGuidedEvictionPolicy : public EvictionPolicy {
public:
    void evict(KVCache& cache) override;
    [[nodiscard]] std::string name() const override { return "AttentionGuided"; }
};


// ═══════════════════════════════════════════════════════════════════════════
//  KV Cache
// ═══════════════════════════════════════════════════════════════════════════

/// Behavioural model of an LLM key-value cache.
///
/// Each "entry" represents one token's cached K/V state.  We track size
/// (in number of entries) rather than actual tensor data, since this is a
/// *behavioural* simulator — the goal is to model hit/miss dynamics and
/// eviction behaviour, not to store real weights.
struct KVCacheEntry {
    int     token_index{0};
    double  attention_score{1.0};   // Simulated importance (higher = keep)
    double  last_access_time{0.0};  // For LRU
    size_t  memory_bytes{0};        // Estimated per-entry footprint
};

class KVCache {
public:
    explicit KVCache(size_t capacity = 4096, size_t bytes_per_entry = 2048)
        : capacity_(capacity)
        , bytes_per_entry_(bytes_per_entry)
    {}

    // ── Mutation ─────────────────────────────────────────────────────────

    /// Simulate prefill: populate cache with `num_tokens` entries.
    void generate_kv_cache(int num_tokens, double sim_time) {
        std::mt19937 rng(42 + num_tokens);                    // reproducible
        std::uniform_real_distribution<double> score_dist(0.1, 1.0);

        entries_.reserve(static_cast<size_t>(num_tokens));
        for (int i = 0; i < num_tokens; ++i) {
            KVCacheEntry e;
            e.token_index     = i;
            e.attention_score = score_dist(rng);
            e.last_access_time = sim_time;
            e.memory_bytes     = bytes_per_entry_;
            entries_.push_back(e);
        }
        current_size_ = entries_.size();

        // Trigger eviction if over capacity
        if (current_size_ > capacity_ && eviction_policy_) {
            eviction_policy_->evict(*this);
        }
    }

    /// Access a token's KV state (for decode).  Returns false on miss.
    bool access_token(int token_idx, double sim_time) {
        for (auto& e : entries_) {
            if (e.token_index == token_idx) {
                e.last_access_time = sim_time;
                ++cache_hits_;
                return true;
            }
        }
        ++cache_misses_;
        return false;
    }

    /// Append a single new decode-token entry to the cache.
    void add_decode_token(int token_idx, double sim_time) {
        KVCacheEntry e;
        e.token_index      = token_idx;
        e.attention_score  = 1.0;   // newly generated → high importance
        e.last_access_time = sim_time;
        e.memory_bytes     = bytes_per_entry_;
        entries_.push_back(e);
        current_size_ = entries_.size();

        if (current_size_ > capacity_ && eviction_policy_) {
            eviction_policy_->evict(*this);
        }
    }

    void set_eviction_policy(std::unique_ptr<EvictionPolicy> policy) {
        eviction_policy_ = std::move(policy);
    }

    void clear() {
        entries_.clear();
        current_size_ = 0;
    }

    // ── Queries ──────────────────────────────────────────────────────────
    [[nodiscard]] size_t current_size()    const noexcept { return current_size_; }
    [[nodiscard]] size_t capacity()        const noexcept { return capacity_; }
    [[nodiscard]] int    cache_hits()      const noexcept { return cache_hits_; }
    [[nodiscard]] int    cache_misses()    const noexcept { return cache_misses_; }
    [[nodiscard]] size_t memory_bytes()    const noexcept { return current_size_ * bytes_per_entry_; }
    [[nodiscard]] double hit_rate()        const noexcept {
        int total = cache_hits_ + cache_misses_;
        return total > 0 ? static_cast<double>(cache_hits_) / total : 0.0;
    }
    [[nodiscard]] const std::string eviction_policy_name() const {
        return eviction_policy_ ? eviction_policy_->name() : "None";
    }

    // Direct access for eviction policies
    std::vector<KVCacheEntry>& entries() { return entries_; }
    const std::vector<KVCacheEntry>& entries() const { return entries_; }
    void update_size() { current_size_ = entries_.size(); }

private:
    size_t capacity_;
    size_t bytes_per_entry_;
    size_t current_size_{0};
    int    cache_hits_{0};
    int    cache_misses_{0};

    std::vector<KVCacheEntry>                entries_;
    std::unique_ptr<EvictionPolicy>          eviction_policy_;
};

// ── Eviction policy implementations ─────────────────────────────────────────

inline void RandomEvictionPolicy::evict(KVCache& cache) {
    auto& entries = cache.entries();
    std::mt19937 rng(static_cast<unsigned>(entries.size()));
    while (entries.size() > cache.capacity()) {
        auto idx = std::uniform_int_distribution<size_t>(0, entries.size() - 1)(rng);
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(idx));
    }
    cache.update_size();
}

inline void LRUEvictionPolicy::evict(KVCache& cache) {
    auto& entries = cache.entries();
    // Sort by last access time (ascending) so oldest-accessed are first
    std::sort(entries.begin(), entries.end(),
              [](const KVCacheEntry& a, const KVCacheEntry& b) {
                  return a.last_access_time < b.last_access_time;
              });
    // Remove oldest until within capacity
    while (entries.size() > cache.capacity()) {
        entries.erase(entries.begin());
    }
    cache.update_size();
}

inline void AttentionGuidedEvictionPolicy::evict(KVCache& cache) {
    auto& entries = cache.entries();
    // Sort by attention score (ascending) so least-important are first
    std::sort(entries.begin(), entries.end(),
              [](const KVCacheEntry& a, const KVCacheEntry& b) {
                  return a.attention_score < b.attention_score;
              });
    while (entries.size() > cache.capacity()) {
        entries.erase(entries.begin());
    }
    cache.update_size();
}

} // namespace llmsimbench


// ═════════════════════════════════════════════════════════════════════════
//  GlobalKVMemoryManager — cross-request GPU memory budget tracking
// ═════════════════════════════════════════════════════════════════════════

namespace llmsimbench {

/// Tracks total KV cache memory usage across all active requests on a
/// single GPU node.  Used to trigger preemption when memory is tight.
class GlobalKVMemoryManager {
public:
    explicit GlobalKVMemoryManager(size_t budget_bytes = 0)
        : budget_bytes_(budget_bytes) {}

    /// Try to allocate `bytes`.  Returns false if it would exceed budget.
    bool try_allocate(size_t bytes) {
        if (budget_bytes_ == 0) return true;  // unlimited
        if (used_bytes_ + bytes > budget_bytes_) return false;
        used_bytes_ += bytes;
        return true;
    }

    /// Force allocate `bytes`, allowing used memory to exceed the budget.
    void force_allocate(size_t bytes) {
        if (!is_enabled()) return;
        used_bytes_ += bytes;
    }

    /// Release `bytes` back to the pool.
    void release(size_t bytes) {
        used_bytes_ = (bytes > used_bytes_) ? 0 : used_bytes_ - bytes;
    }

    /// Current memory pressure as a fraction [0.0, 1.0].
    [[nodiscard]] double memory_pressure() const {
        if (budget_bytes_ == 0) return 0.0;
        return static_cast<double>(used_bytes_) / static_cast<double>(budget_bytes_);
    }

    [[nodiscard]] size_t used_bytes()   const noexcept { return used_bytes_; }
    [[nodiscard]] size_t budget_bytes() const noexcept { return budget_bytes_; }
    [[nodiscard]] bool   is_enabled()   const noexcept { return budget_bytes_ > 0; }

    void set_budget(size_t bytes) { budget_bytes_ = bytes; }
    void reset() { used_bytes_ = 0; }

private:
    size_t budget_bytes_{0};
    size_t used_bytes_{0};
};

} // namespace llmsimbench
