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



} // namespace llmsimbench

