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

class PrefillEngine {
public:
    explicit PrefillEngine(HardwareProfile hw, double model_params_B = 7.0,
                           int chunk_size = 0)
        : hw_(std::move(hw))
        , model_params_B_(model_params_B)
        , chunk_size_(chunk_size)
    {
        // Approximate time per token (seconds) for the prefill pass.
        // 2 * params * 1e9 FLOP per token / (TFLOPS * 1e12 FLOP/s)
        // = 2 * params_B * 1e-3 / compute_tflops
        time_per_token_ = (2.0 * model_params_B_ * 1e-3) / hw_.compute_tflops;
    }

    [[nodiscard]] bool is_available() const {
        return !is_busy_;
    }

    /// Add a request to the prefill engine.
    /// If the request has partial prefill progress (from a previous preemption),
    /// we resume from where it left off.
    void add_request(Request& req, double sim_time) {
        assert(!is_busy_);
        is_busy_ = true;
        current_request_id_ = req.get_id();

        int already_processed = req.get_prefill_tokens_processed();
        steps_remaining_ = req.get_prompt_length() - already_processed;

        if (req.get_state() == RequestState::PREEMPTED) {
            req.resume_prefill(sim_time);
        } else if (req.get_state() == RequestState::PENDING) {
            req.start_prefill(sim_time);
        }
        // else: already IN_PREFILL (shouldn't happen, but be safe)
    }

    /// Process entire prefill in one step (parallel pass).
    /// Returns the wall-clock time for the full prefill.
    /// Backward-compatible: ignores chunk_size.
    double process_step(double /*sim_time*/) {
        assert(is_busy_);
        double elapsed = steps_remaining_ * time_per_token_;
        steps_remaining_ = 0;
        return elapsed;
    }

    /// Process a single chunk of the prefill.
    /// Returns the wall-clock time for this chunk.
    /// If chunk_size_ <= 0, processes the entire remaining prefill.
    double process_chunk(double /*sim_time*/) {
        assert(is_busy_);
        int tokens_this_chunk;
        if (chunk_size_ > 0 && steps_remaining_ > chunk_size_) {
            tokens_this_chunk = chunk_size_;
        } else {
            tokens_this_chunk = steps_remaining_;
        }
        steps_remaining_ -= tokens_this_chunk;
        return tokens_this_chunk * time_per_token_;
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
    [[nodiscard]] int  steps_remaining() const { return steps_remaining_; }
    [[nodiscard]] double time_per_token() const { return time_per_token_; }
    [[nodiscard]] int  chunk_size() const { return chunk_size_; }

    void set_chunk_size(int cs) { chunk_size_ = cs; }

private:
    HardwareProfile hw_;
    double model_params_B_;
    double time_per_token_;
    int    chunk_size_{0};   // 0 = no chunking (process all at once)
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

} // namespace llmsimbench

