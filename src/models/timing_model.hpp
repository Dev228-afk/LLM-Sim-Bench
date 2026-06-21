#pragma once
// =============================================================================
// timing_model.hpp — Engine components and the main SimulationEngine (DES)
// =============================================================================
//
// • EngineComponent (abstract)  – common interface for processing units.
// • PrefillEngine               – compute-bound parallel prefill model.
// • DecodeEngine                – bandwidth-bound sequential decode model.
// • SimulationEngine            – top-level discrete-event simulation loop.
// =============================================================================

#include "../core/event.hpp"
#include "../core/event_queue.hpp"
#include "../core/workload.hpp"
#include "../core/scheduler.hpp"
#include "hardware_profile.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

namespace llmsimbench {

// EngineComponent abstraction removed for flexibility.


// ═══════════════════════════════════════════════════════════════════════════
//  PrefillEngine
// ═══════════════════════════════════════════════════════════════════════════
//
// Models the compute-bound prefill phase.  The entire prompt is processed
// in a single parallel forward pass.  Latency is dominated by FLOPS:
//
//   time_per_token ≈ (2 * model_params * tokens) / (compute_tflops * 1e12)
//
// For simplicity we linearise: prefill_time = prompt_length * time_per_token.
// =============================================================================

class PrefillEngine {
public:
    explicit PrefillEngine(HardwareProfile hw, double model_params_B = 7.0)
        : hw_(std::move(hw))
        , model_params_B_(model_params_B)
    {
        // Approximate time per token (seconds) for the prefill pass.
        // 2 * params * 1e9 FLOP per token / (TFLOPS * 1e12 FLOP/s)
        // = 2 * params_B * 1e-3 / compute_tflops
        time_per_token_ = (2.0 * model_params_B_ * 1e-3) / hw_.compute_tflops;
    }

    [[nodiscard]] bool is_available() const {
        return !is_busy_;
    }

    void add_request(Request& req, double sim_time) {
        assert(!is_busy_);
        is_busy_ = true;
        current_request_id_ = req.get_id();
        steps_remaining_    = req.get_prompt_length();   // one "step" per prompt token
        req.start_prefill(sim_time);
    }

    /// Process entire prefill in one step (parallel pass).
    /// Returns the wall-clock time for the full prefill.
    double process_step(double /*sim_time*/) {
        assert(is_busy_);
        double elapsed = steps_remaining_ * time_per_token_;
        steps_remaining_ = 0;
        return elapsed;
    }

    [[nodiscard]] std::optional<int> current_request_id() const {
        if (is_busy_) return current_request_id_;
        return std::nullopt;
    }

    void release() {
        is_busy_ = false;
        current_request_id_ = -1;
        steps_remaining_ = 0;
    }

    [[nodiscard]] bool prefill_done() const { return is_busy_ && steps_remaining_ == 0; }
    [[nodiscard]] double time_per_token() const { return time_per_token_; }

private:
    HardwareProfile hw_;
    double model_params_B_;
    double time_per_token_;
    bool   is_busy_{false};
    int    current_request_id_{-1};
    int    steps_remaining_{0};
};


// ═══════════════════════════════════════════════════════════════════════════
//  DecodeEngine
// ═══════════════════════════════════════════════════════════════════════════
//
// Models the bandwidth-bound decode phase.  Each token is generated
// sequentially by reading the KV cache.  Latency per token:
//
//   time_per_token ≈ (2 * model_params * 1e9 bytes) /
//                    (memory_bandwidth * 1e9 bytes/s)
//                  = 2 * model_params_B / memory_bandwidth_gbps   (seconds)
// =============================================================================

class DecodeEngine {
public:
    explicit DecodeEngine(HardwareProfile hw, double model_params_B = 7.0)
        : hw_(std::move(hw))
        , model_params_B_(model_params_B)
    {
        time_to_load_weights_ = (2.0 * model_params_B_) / hw_.memory_bandwidth_gbps;
    }

    [[nodiscard]] bool is_available() const {
        return true; 
    }

    void add_request(Request& req, double sim_time) {
        active_requests_.push_back(req.get_id());
        req.transition_to_decode(sim_time);
    }

    void release(int req_id) {
        active_requests_.erase(
            std::remove(active_requests_.begin(), active_requests_.end(), req_id),
            active_requests_.end());
    }

    [[nodiscard]] double time_per_token() const { return time_to_load_weights_; }
    [[nodiscard]] const std::vector<int>& active_requests() const { return active_requests_; }

private:
    HardwareProfile hw_;
    double model_params_B_;
    double time_to_load_weights_;
    std::vector<int> active_requests_;
};


// ═══════════════════════════════════════════════════════════════════════════
//  SimulationEngine — top-level DES orchestrator
// ═══════════════════════════════════════════════════════════════════════════

/// Performance counters exposed to Python for post-simulation analysis.
struct SimulationStats {
    int    total_requests_completed{0};
    int    total_tokens_generated{0};
    double total_prefill_time{0.0};
    double total_decode_time{0.0};
    double total_e2e_latency{0.0};
    double max_e2e_latency{0.0};
    int    total_cache_hits{0};
    int    total_cache_misses{0};
    int    total_cache_evictions{0};
    int    peak_pending_queue_depth{0};

    [[nodiscard]] double avg_e2e_latency() const {
        return total_requests_completed > 0
            ? total_e2e_latency / total_requests_completed : 0.0;
    }
    [[nodiscard]] double throughput_tokens_per_sec() const {
        return total_decode_time > 0
            ? total_tokens_generated / total_decode_time : 0.0;
    }
};


class SimulationEngine {
public:
    /// Construct the engine with a scheduler, prefill and decode hardware
    /// profiles, and an optional model size (billions of parameters).
    SimulationEngine(
        std::shared_ptr<Scheduler>   scheduler,
        HardwareProfile              prefill_hw,
        HardwareProfile              decode_hw,
        double                       model_params_B = 7.0,
        size_t                       kv_cache_capacity = 4096,
        double                       kv_transfer_latency_ms = 0.5)
        : scheduler_(std::move(scheduler))
        , prefill_engine_(std::move(prefill_hw), model_params_B)
        , decode_engine_(std::move(decode_hw), model_params_B)
        , kv_cache_capacity_(kv_cache_capacity)
        , kv_transfer_latency_ms_(kv_transfer_latency_ms)
    {}

    // ── Request injection ────────────────────────────────────────────────

    void add_request(int id, int prompt_len, int gen_len, double arrival_time) {
        requests_.emplace(id, Request(id, prompt_len, gen_len, arrival_time));
        kv_caches_.emplace(id, KVCache(kv_cache_capacity_));

        // Schedule the arrival event
        event_queue_.push({arrival_time, EventType::REQUEST_ARRIVAL, id});
    }

    /// Convenience: add a Request object directly.
    void add_request_obj(const Request& req) {
        add_request(req.get_id(), req.get_prompt_length(),
                    req.get_generation_length(), req.get_arrival_time());
    }

    // ── Simulation loop ──────────────────────────────────────────────────

    /// Process the next event.  Returns `false` when the queue is empty
    /// (simulation complete).
    bool step() {
        if (event_queue_.empty()) return false;

        Event ev = event_queue_.pop();
        current_time_ = ev.time;

        switch (ev.type) {
            case EventType::REQUEST_ARRIVAL:      handle_arrival(ev);        break;
            case EventType::PREFILL_COMPLETE:     handle_prefill_complete(ev); break;
            case EventType::KV_TRANSFER_COMPLETE: handle_kv_transfer(ev);    break;
            case EventType::DECODE_STEP_COMPLETE: handle_decode_step(ev);    break;
            default: break;
        }

        return true;
    }

    /// Run the simulation to completion.
    void run() {
        while (step()) {}
    }

    // ── Queries ──────────────────────────────────────────────────────────

    [[nodiscard]] double get_current_time() const noexcept { return current_time_; }

    [[nodiscard]] std::vector<Request> get_completed_requests() const {
        std::vector<Request> out;
        for (const auto& [id, req] : requests_) {
            if (req.is_completed()) out.push_back(req);
        }
        return out;
    }

    [[nodiscard]] std::vector<Request> get_all_requests() const {
        std::vector<Request> out;
        out.reserve(requests_.size());
        for (const auto& [id, req] : requests_) {
            out.push_back(req);
        }
        return out;
    }

    [[nodiscard]] size_t pending_count() const { return pending_ids_.size(); }
    [[nodiscard]] size_t completed_count() const {
        size_t c = 0;
        for (const auto& [id, req] : requests_) {
            if (req.is_completed()) ++c;
        }
        return c;
    }
    [[nodiscard]] size_t total_request_count() const { return requests_.size(); }

    [[nodiscard]] const SimulationStats& stats() const noexcept { return stats_; }

    [[nodiscard]] std::string scheduler_name() const {
        return scheduler_ ? scheduler_->policy_name() : "None";
    }

    // ── Configuration setters ────────────────────────────────────────────

    void set_eviction_policy_for_all(const std::string& policy_name) {
        for (auto& [id, cache] : kv_caches_) {
            if (policy_name == "Random") {
                cache.set_eviction_policy(std::make_unique<RandomEvictionPolicy>());
            } else if (policy_name == "LRU") {
                cache.set_eviction_policy(std::make_unique<LRUEvictionPolicy>());
            } else if (policy_name == "AttentionGuided") {
                cache.set_eviction_policy(std::make_unique<AttentionGuidedEvictionPolicy>());
            }
            // else: no policy (unlimited cache)
        }
        default_eviction_policy_ = policy_name;
    }

private:
    // ── Event handlers ───────────────────────────────────────────────────

    void handle_arrival(const Event& ev) {
        pending_ids_.push_back(ev.request_id);
        stats_.peak_pending_queue_depth = std::max(
            stats_.peak_pending_queue_depth,
            static_cast<int>(pending_ids_.size()));

        try_schedule_prefill();
    }

    void handle_prefill_complete(const Event& ev) {
        auto& req = requests_.at(ev.request_id);
        auto& cache = kv_caches_.at(ev.request_id);

        // Generate the KV cache from the prefill phase
        cache.generate_kv_cache(req.get_prompt_length(), current_time_);
        stats_.total_cache_hits   += cache.cache_hits();
        stats_.total_cache_misses += cache.cache_misses();

        // Release the prefill engine
        prefill_engine_.release();

        // Schedule KV transfer (simulate network cost of PDD)
        double transfer_time = kv_transfer_latency_ms_ / 1000.0;  // ms → s
        event_queue_.push({
            current_time_ + transfer_time,
            EventType::KV_TRANSFER_COMPLETE,
            ev.request_id
        });

        // Try to schedule another pending request into prefill
        try_schedule_prefill();
    }

    void handle_kv_transfer(const Event& ev) {
        // KV cache has arrived at the decode node — start decoding
        decode_waiting_.push_back(ev.request_id);
        try_schedule_decode();
    }

    void handle_decode_step(const Event& ev) {
        decode_step_scheduled_ = false;

        std::vector<int> current_batch = decode_engine_.active_requests();

        for (int req_id : current_batch) {
            auto& req = requests_.at(req_id);
            bool done = req.generate_token();
            stats_.total_tokens_generated++;

            if (done) {
                req.complete(current_time_);
                decode_engine_.release(req_id);

                stats_.total_requests_completed++;
                stats_.total_prefill_time += req.get_prefill_latency();
                stats_.total_decode_time  += req.get_decode_latency();
                double e2e = req.get_end_to_end_latency();
                stats_.total_e2e_latency  += e2e;
                stats_.max_e2e_latency     = std::max(stats_.max_e2e_latency, e2e);

                auto& cache = kv_caches_.at(req_id);
                stats_.total_cache_hits   += cache.cache_hits();
                stats_.total_cache_misses += cache.cache_misses();
            }
        }

        try_schedule_decode();
    }

    // ── Scheduling helpers ───────────────────────────────────────────────

    void try_schedule_prefill() {
        if (pending_ids_.empty() || !prefill_engine_.is_available()) return;

        // Build a vector of pending requests for the scheduler
        std::vector<Request> pending_reqs;
        pending_reqs.reserve(pending_ids_.size());
        for (int id : pending_ids_) {
            pending_reqs.push_back(requests_.at(id));
        }

        auto ordered_ids = scheduler_->schedule(pending_reqs);
        if (ordered_ids.empty()) return;

        // Take the highest-priority request
        int chosen_id = ordered_ids.front();

        // Remove from pending
        pending_ids_.erase(
            std::remove(pending_ids_.begin(), pending_ids_.end(), chosen_id),
            pending_ids_.end());

        auto& req = requests_.at(chosen_id);

        // Set eviction policy on the KV cache
        if (!default_eviction_policy_.empty()) {
            auto& cache = kv_caches_.at(chosen_id);
            if (default_eviction_policy_ == "Random")
                cache.set_eviction_policy(std::make_unique<RandomEvictionPolicy>());
            else if (default_eviction_policy_ == "LRU")
                cache.set_eviction_policy(std::make_unique<LRUEvictionPolicy>());
            else if (default_eviction_policy_ == "AttentionGuided")
                cache.set_eviction_policy(std::make_unique<AttentionGuidedEvictionPolicy>());
        }

        // Assign to prefill engine
        prefill_engine_.add_request(req, current_time_);
        double dt = prefill_engine_.process_step(current_time_);

        // Schedule prefill completion
        event_queue_.push({
            current_time_ + dt,
            EventType::PREFILL_COMPLETE,
            chosen_id
        });
    }

    void try_schedule_decode() {
        if (!decode_waiting_.empty()) {
            bool is_cb = (scheduler_name().find("ContinuousBatching") != std::string::npos);
            
            if (!is_cb && !decode_engine_.active_requests().empty()) {
                // Strict sequential decode for FCFS/Priority
            } else {
                std::vector<Request> pending_reqs;
                for (int id : decode_waiting_) pending_reqs.push_back(requests_.at(id));

                auto ordered_ids = scheduler_->schedule(pending_reqs);
                if (!ordered_ids.empty()) {
                    size_t max_batch = 8; 
                    for (int id : ordered_ids) {
                        if (!is_cb && !decode_engine_.active_requests().empty()) break; 
                        if (is_cb && decode_engine_.active_requests().size() >= max_batch) break;

                        auto& req = requests_.at(id);
                        decode_engine_.add_request(req, current_time_);
                        decode_waiting_.erase(
                            std::remove(decode_waiting_.begin(), decode_waiting_.end(), id),
                            decode_waiting_.end());
                    }
                }
            }
        }

        if (!decode_engine_.active_requests().empty() && !decode_step_scheduled_) {
            const auto& active = decode_engine_.active_requests();
            double base_dt = decode_engine_.time_per_token() + active.size() * 0.0001;
            double max_miss_penalty = 0.0;
            
            for (int req_id : active) {
                auto& req = requests_.at(req_id);
                auto& cache = kv_caches_.at(req_id);

                int generated_so_far = req.get_tokens_generated();
                int total_tokens_to_attend = req.get_prompt_length() + generated_so_far;

                int misses = 0;
                for (int i = 0; i < total_tokens_to_attend; ++i) {
                    if (!cache.access_token(i, current_time_)) {
                        misses++;
                    }
                }
                cache.add_decode_token(total_tokens_to_attend, current_time_);
                
                double penalty = misses * prefill_engine_.time_per_token();
                max_miss_penalty = std::max(max_miss_penalty, penalty);
            }
            
            double dt = base_dt + max_miss_penalty;
            event_queue_.push({current_time_ + dt, EventType::DECODE_STEP_COMPLETE, -1});
            decode_step_scheduled_ = true;
        }
    }

    // ── State ────────────────────────────────────────────────────────────
    double current_time_{0.0};
    bool decode_step_scheduled_{false};

    EventQueue                          event_queue_;
    std::shared_ptr<Scheduler>          scheduler_;
    PrefillEngine                       prefill_engine_;
    DecodeEngine                        decode_engine_;

    std::map<int, Request>              requests_;        // All requests by ID
    std::map<int, KVCache>              kv_caches_;       // KV cache per request
    std::vector<int>                    pending_ids_;     // Awaiting prefill
    std::vector<int>                    decode_waiting_;  // Awaiting decode engine

    size_t                              kv_cache_capacity_;
    double                              kv_transfer_latency_ms_;
    std::string                         default_eviction_policy_;

    SimulationStats                     stats_;
};

} // namespace llmsimbench
