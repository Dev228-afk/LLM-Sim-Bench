#pragma once
// =============================================================================
// event.hpp — Discrete-Event Simulation primitives for LLM Sim Bench
// =============================================================================
//
// Defines the fundamental event types and the Event struct that drives the
// simulation's time-ordered event queue.  Every state transition in the
// simulator (request arrival, phase completion, KV-cache transfer, …) is
// expressed as an Event with a scheduled timestamp.
// =============================================================================

#include <cstdint>
#include <compare>
#include <string>

namespace llmsimbench {

// ── Event taxonomy ──────────────────────────────────────────────────────────
// Each value maps to a distinct simulation state transition.
enum class EventType : uint8_t {
    REQUEST_ARRIVAL,        // A new inference request enters the system
    PREFILL_COMPLETE,       // Prefill phase finished → ready for KV transfer / decode
    KV_TRANSFER_COMPLETE,   // KV cache transferred from prefill to decode node
    DECODE_STEP_COMPLETE,   // A single decode token has been generated
    DECODE_COMPLETE,        // All tokens for a request have been decoded
    PREEMPTION,             // Scheduler preempts a running request
    CACHE_EVICTION          // KV cache eviction event triggered by memory pressure
};

/// Human-readable label for logging / debug output.
inline std::string event_type_to_string(EventType t) {
    switch (t) {
        case EventType::REQUEST_ARRIVAL:      return "REQUEST_ARRIVAL";
        case EventType::PREFILL_COMPLETE:     return "PREFILL_COMPLETE";
        case EventType::KV_TRANSFER_COMPLETE: return "KV_TRANSFER_COMPLETE";
        case EventType::DECODE_STEP_COMPLETE: return "DECODE_STEP_COMPLETE";
        case EventType::DECODE_COMPLETE:      return "DECODE_COMPLETE";
        case EventType::PREEMPTION:           return "PREEMPTION";
        case EventType::CACHE_EVICTION:       return "CACHE_EVICTION";
        default:                              return "UNKNOWN";
    }
}

// ── Event struct ────────────────────────────────────────────────────────────
// Ordered by time (ascending); ties broken by a monotonically-increasing
// sequence number so that insertion order is preserved for simultaneous events.
struct Event {
    double     time{0.0};        // Simulation timestamp (seconds)
    EventType  type{EventType::REQUEST_ARRIVAL};
    int        request_id{-1};   // Owning request (-1 = system-level event)
    uint64_t   seq{0};           // Tie-breaker (auto-assigned by EventQueue)

    // Min-heap ordering: earliest time first, then lowest seq.
    // NOTE: std::priority_queue is a *max*-heap, so operator> gives min-heap
    //       when used with std::greater<>.
    std::partial_ordering operator<=>(const Event& o) const {
        if (auto cmp = time <=> o.time; cmp != 0) return cmp;
        if (seq < o.seq) return std::partial_ordering::less;
        if (seq > o.seq) return std::partial_ordering::greater;
        return std::partial_ordering::equivalent;
    }
    bool operator==(const Event& o) const = default;
};

} // namespace llmsimbench
