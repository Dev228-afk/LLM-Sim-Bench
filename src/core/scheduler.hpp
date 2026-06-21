#pragma once
// =============================================================================
// scheduler.hpp — Request scheduling policies for LLM Sim Bench
// =============================================================================
//
// Abstract Scheduler base with three concrete strategies:
//   • FirstComeFirstServeScheduler   – FIFO by arrival time
//   • PriorityScheduler              – weighted score ranking
//   • ContinuousBatchingScheduler    – groups decode requests into batches
// =============================================================================

#include "workload.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace llmsimbench {

/// Abstract scheduler interface.
///
/// `schedule()` receives the list of pending requests and returns a vector
/// of request IDs in the order they should be dispatched to engines.
class Scheduler {
public:
    virtual ~Scheduler() = default;

    /// Return ordered request-IDs for next dispatch round.
    [[nodiscard]] virtual std::vector<int> schedule(
        std::vector<Request>& pending_requests) = 0;

    [[nodiscard]] virtual std::string policy_name() const = 0;
};


// ─────────────────────────────────────────────────────────────────────────────
//  FCFS
// ─────────────────────────────────────────────────────────────────────────────

class FirstComeFirstServeScheduler : public Scheduler {
public:
    [[nodiscard]] std::vector<int> schedule(
        std::vector<Request>& pending) override
    {
        // Already in arrival order — just return IDs
        std::sort(pending.begin(), pending.end(),
                  [](const Request& a, const Request& b) {
                      return a.get_arrival_time() < b.get_arrival_time();
                  });

        std::vector<int> order;
        order.reserve(pending.size());
        for (const auto& r : pending) order.push_back(r.get_id());
        return order;
    }

    [[nodiscard]] std::string policy_name() const override { return "FCFS"; }
};


// ─────────────────────────────────────────────────────────────────────────────
//  Priority-based
// ─────────────────────────────────────────────────────────────────────────────

/// Scores each request as:
///     score = prompt_length * prompt_weight + generation_length * gen_weight
///
/// Lower score → higher priority (processed first).  This favours short
/// requests, which is a common heuristic to minimise average latency.
class PriorityScheduler : public Scheduler {
public:
    explicit PriorityScheduler(double prompt_w = 0.1, double gen_w = 0.5)
        : prompt_weight_(prompt_w), gen_weight_(gen_w) {}

    [[nodiscard]] std::vector<int> schedule(
        std::vector<Request>& pending) override
    {
        std::sort(pending.begin(), pending.end(),
                  [this](const Request& a, const Request& b) {
                      return score(a) < score(b);
                  });

        std::vector<int> order;
        order.reserve(pending.size());
        for (const auto& r : pending) order.push_back(r.get_id());
        return order;
    }

    [[nodiscard]] std::string policy_name() const override { return "Priority"; }

private:
    double prompt_weight_;
    double gen_weight_;

    [[nodiscard]] double score(const Request& r) const {
        return r.get_prompt_length() * prompt_weight_
             + r.get_generation_length() * gen_weight_;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
//  Continuous Batching
// ─────────────────────────────────────────────────────────────────────────────

/// Groups up to `max_batch_size` requests per scheduling round.
/// Within a batch, requests are ordered FCFS.
class ContinuousBatchingScheduler : public Scheduler {
public:
    explicit ContinuousBatchingScheduler(int max_batch = 8)
        : max_batch_size_(max_batch) {}

    [[nodiscard]] std::vector<int> schedule(
        std::vector<Request>& pending) override
    {
        // FCFS within batch
        std::sort(pending.begin(), pending.end(),
                  [](const Request& a, const Request& b) {
                      return a.get_arrival_time() < b.get_arrival_time();
                  });

        std::vector<int> order;
        int count = std::min(static_cast<int>(pending.size()), max_batch_size_);
        order.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            order.push_back(pending[static_cast<size_t>(i)].get_id());
        }
        return order;
    }

    [[nodiscard]] std::string policy_name() const override {
        return "ContinuousBatching(batch=" + std::to_string(max_batch_size_) + ")";
    }

private:
    int max_batch_size_;
};

} // namespace llmsimbench
