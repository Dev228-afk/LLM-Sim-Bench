#pragma once
// =============================================================================
// event_queue.hpp — Time-ordered event queue for the DES engine
// =============================================================================

#include "event.hpp"

#include <queue>
#include <vector>
#include <functional>   // std::greater
#include <cstdint>

namespace llmsimbench {

/// Min-heap priority queue of Events (earliest time pops first).
///
/// Each push auto-assigns a monotonically-increasing sequence number so that
/// events scheduled at the exact same time are processed in insertion order.
class EventQueue {
public:
    /// Schedule a new event.  `e.seq` is overwritten with the next counter.
    void push(Event e) {
        e.seq = next_seq_++;
        heap_.push(std::move(e));
    }

    /// Remove and return the earliest event.
    /// Pre-condition: !empty()
    [[nodiscard]] Event pop() {
        Event top = heap_.top();
        heap_.pop();
        return top;
    }

    /// Peek at the earliest event without removing it.
    [[nodiscard]] const Event& peek() const {
        return heap_.top();
    }

    [[nodiscard]] bool   empty() const noexcept { return heap_.empty(); }
    [[nodiscard]] size_t size()  const noexcept { return heap_.size();  }

    /// Drain the queue (useful for reset).
    void clear() {
        heap_ = {};
        next_seq_ = 0;
    }

private:
    // std::greater turns the default max-heap into a min-heap.
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> heap_;
    uint64_t next_seq_{0};
};

} // namespace llmsimbench
