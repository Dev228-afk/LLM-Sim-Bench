#pragma once

#include <vector>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <algorithm>
#include <optional>
#include <limits>
#include <iostream>

#include "../core/event.hpp"
#include "../core/event_queue.hpp"
#include "../core/workload.hpp"
#include "../core/scheduler.hpp"
#include "hardware_profile.hpp"

namespace llmsimbench {

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
    int    total_preemptions{0};
    int    total_memory_pressure_evictions{0};

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
        double                       kv_transfer_latency_ms = 0.5,
        int                          prefill_chunk_size = 0,
        size_t                       global_memory_budget_bytes = 0)
        : scheduler_(std::move(scheduler))
        , prefill_engine_(std::move(prefill_hw), model_params_B, prefill_chunk_size)
        , decode_engine_(std::move(decode_hw), model_params_B)
        , kv_cache_capacity_(kv_cache_capacity)
        , kv_transfer_latency_ms_(kv_transfer_latency_ms)
        , memory_manager_(global_memory_budget_bytes)
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
            case EventType::REQUEST_ARRIVAL:        handle_arrival(ev);               break;
            case EventType::PREFILL_COMPLETE:       handle_prefill_complete(ev);      break;
            case EventType::PREFILL_CHUNK_COMPLETE: handle_prefill_chunk_complete(ev); break;
            case EventType::KV_TRANSFER_COMPLETE:   handle_kv_transfer(ev);           break;
            case EventType::DECODE_STEP_COMPLETE:   handle_decode_step(ev);           break;
            case EventType::PREEMPTION:             handle_preemption(ev);            break;
            case EventType::CACHE_EVICTION:         handle_cache_eviction(ev);        break;
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

        // Check if the newly arrived request should preempt a running prefill
        if (uses_priority_cb_scheduler() && !prefill_engine_.is_available()) {
            auto* pcb = dynamic_cast<PriorityContinuousBatchingScheduler*>(scheduler_.get());
            if (pcb) {
                auto running_id_opt = prefill_engine_.current_request_id();
                if (running_id_opt.has_value()) {
                    auto& running_req   = requests_.at(running_id_opt.value());
                    auto& candidate_req = requests_.at(ev.request_id);
                    if (pcb->should_preempt(running_req, candidate_req)) {
                        // Schedule immediate preemption of the running prefill
                        event_queue_.push({
                            current_time_,
                            EventType::PREEMPTION,
                            running_id_opt.value()
                        });
                        return; // preemption handler will re-schedule
                    }
                }
            }
        }

        try_schedule_prefill();
    }

    void handle_prefill_complete(const Event& ev) {
        if (prefill_engine_.current_request_id() != ev.request_id) return; // Stale event

        auto& req = requests_.at(ev.request_id);
        auto& cache = kv_caches_.at(ev.request_id);

        // Mark all prefill tokens as processed
        req.set_prefill_tokens_processed(req.get_prompt_length());

        // Generate the KV cache from the prefill phase
        cache.generate_kv_cache(req.get_prompt_length(), current_time_);
        stats_.total_cache_hits   += cache.cache_hits();
        stats_.total_cache_misses += cache.cache_misses();

        // Track global memory usage
        if (memory_manager_.is_enabled()) {
            memory_manager_.force_allocate(cache.memory_bytes());
        }

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

    /// Handle completion of one prefill chunk.
    /// If more chunks remain, either schedule the next chunk or preempt
    /// if a higher-priority request is waiting.
    void handle_prefill_chunk_complete(const Event& ev) {
        if (prefill_engine_.current_request_id() != ev.request_id) return; // Stale event

        auto& req = requests_.at(ev.request_id);

        // Update the request's progress
        int processed = req.get_prompt_length() - prefill_engine_.steps_remaining();
        req.set_prefill_tokens_processed(processed);

        if (prefill_engine_.prefill_done()) {
            // All chunks done — treat as full PREFILL_COMPLETE
            handle_prefill_complete(ev);
            return;
        }

        // Check if we should preempt for a higher-priority pending request
        if (uses_priority_cb_scheduler() && !pending_ids_.empty()) {
            auto* pcb = dynamic_cast<PriorityContinuousBatchingScheduler*>(scheduler_.get());
            if (pcb) {
                // Find the highest-priority pending request
                std::vector<Request> pending_reqs;
                for (int id : pending_ids_) pending_reqs.push_back(requests_.at(id));
                auto ordered = pcb->schedule(pending_reqs);
                if (!ordered.empty()) {
                    auto& candidate = requests_.at(ordered.front());
                    if (pcb->should_preempt(req, candidate)) {
                        // Preempt: save progress and re-queue
                        req.preempt(current_time_);
                        prefill_engine_.release();
                        pending_ids_.push_back(ev.request_id);
                        stats_.total_preemptions++;

                        // Schedule the higher-priority request
                        try_schedule_prefill();
                        return;
                    }
                }
            }
        }

        // No preemption — schedule the next chunk
        double dt = prefill_engine_.process_chunk(current_time_);
        EventType next_type = prefill_engine_.prefill_done()
            ? EventType::PREFILL_COMPLETE
            : EventType::PREFILL_CHUNK_COMPLETE;
        event_queue_.push({
            current_time_ + dt,
            next_type,
            ev.request_id
        });
    }

    void handle_preemption(const Event& ev) {
        auto& req = requests_.at(ev.request_id);

        if (req.get_state() == RequestState::IN_PREFILL) {
            // Check if this request is ACTUALLY the one currently in prefill engine
            if (prefill_engine_.current_request_id() != ev.request_id) {
                // Stale preemption event (e.g. prefill already completed)
                return;
            }

            // Save partial prefill progress
            int processed = req.get_prompt_length() - prefill_engine_.steps_remaining();
            req.set_prefill_tokens_processed(processed);
            req.preempt(current_time_);
            prefill_engine_.release();
        } else if (req.get_state() == RequestState::IN_DECODE) {
            // Check if it's still active in decode engine
            auto active = decode_engine_.active_requests();
            if (std::find(active.begin(), active.end(), ev.request_id) == active.end()) {
                return; // Stale event
            }
            // Preempt from decode — release KV memory
            req.preempt(current_time_);
            decode_engine_.release(ev.request_id);
            if (memory_manager_.is_enabled()) {
                auto& cache = kv_caches_.at(ev.request_id);
                memory_manager_.release(cache.memory_bytes());
                cache.clear();  // KV cache is "swapped out"
            }
        } else {
            return; // Stale event
        }

        stats_.total_preemptions++;

        // Re-queue the preempted request
        pending_ids_.push_back(ev.request_id);

        // Try to schedule the next request
        try_schedule_prefill();
        try_schedule_decode();
    }

    void handle_cache_eviction(const Event& /*ev*/) {
        // Memory pressure eviction: find lowest-priority active decode request
        check_memory_pressure();
    }

    void handle_kv_transfer(const Event& ev) {
        // KV cache has arrived at the decode node — start decoding
        decode_waiting_.push_back(ev.request_id);
        try_schedule_decode();
    }

    void handle_decode_step(const Event& ev) {
        (void)ev;
        decode_step_scheduled_ = false;

        std::vector<int> current_batch = decode_engine_.active_requests();

        for (int req_id : current_batch) {
            auto& req = requests_.at(req_id);
            bool done = req.generate_token();
            stats_.total_tokens_generated++;

            if (done) {
                req.complete(current_time_);
                decode_engine_.release(req_id);

                // Release global memory
                if (memory_manager_.is_enabled()) {
                    auto& cache = kv_caches_.at(req_id);
                    memory_manager_.release(cache.memory_bytes());
                }

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

        // Check memory pressure after decode step
        if (memory_manager_.is_enabled()) {
            check_memory_pressure();
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

        // Use chunked or full-step based on engine configuration
        if (prefill_engine_.chunk_size() > 0) {
            double dt = prefill_engine_.process_chunk(current_time_);
            EventType next_type = prefill_engine_.prefill_done()
                ? EventType::PREFILL_COMPLETE
                : EventType::PREFILL_CHUNK_COMPLETE;
            event_queue_.push({
                current_time_ + dt,
                next_type,
                chosen_id
            });
        } else {
            // Backward-compatible: process entire prefill in one step
            double dt = prefill_engine_.process_step(current_time_);
            event_queue_.push({
                current_time_ + dt,
                EventType::PREFILL_COMPLETE,
                chosen_id
            });
        }
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

    /// Check global memory pressure and preempt lowest-priority decode
    /// requests when usage exceeds 90% of budget.
    void check_memory_pressure() {
        if (!memory_manager_.is_enabled()) return;

        double pressure = memory_manager_.memory_pressure();
        if (pressure < 0.9) return;

        // Find the lowest priority request currently in DECODE
        auto active = decode_engine_.active_requests();
        if (active.empty()) return;

        // Use the scheduler to determine priority (last in sorted order = lowest priority)
        auto* pcb = dynamic_cast<PriorityContinuousBatchingScheduler*>(scheduler_.get());
        if (!pcb) return;  // Only preempt with priority-aware scheduler

        // Find worst-priority request
        int worst_id = active.front();
        double worst_score = pcb->score(requests_.at(worst_id));
        for (int id : active) {
            double s = pcb->score(requests_.at(id));
            if (s > worst_score) {
                worst_score = s;
                worst_id = id;
            }
        }

        // Preempt it
        auto& req = requests_.at(worst_id);
        req.preempt(current_time_);
        decode_engine_.release(worst_id);

        auto& cache = kv_caches_.at(worst_id);
        memory_manager_.release(cache.memory_bytes());
        cache.clear();  // Simulate KV cache swap-out

        stats_.total_preemptions++;
        stats_.total_memory_pressure_evictions++;

        // Re-queue — it will need to re-prefill (KV cache was cleared)
        req.set_prefill_tokens_processed(0);  // must re-prefill
        pending_ids_.push_back(worst_id);
    }

    [[nodiscard]] bool uses_priority_cb_scheduler() const {
        return scheduler_name().find("PriorityContinuousBatching") != std::string::npos;
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

    GlobalKVMemoryManager               memory_manager_;
    SimulationStats                     stats_;
};

} // namespace llmsimbench



