#pragma once
// =============================================================================
// hardware_profile.hpp — GPU / accelerator performance profiles
// =============================================================================
//
// Each profile captures the key performance parameters that determine
// prefill and decode throughput in the timing model.
// =============================================================================

#include <string>

namespace llmsimbench {

/// Describes the performance envelope of a single accelerator.
struct HardwareProfile {
    std::string name;

    double compute_tflops{0.0};          // Peak FP16/BF16 TFLOPS
    double memory_bandwidth_gbps{0.0};   // HBM bandwidth (GB/s)
    double memory_capacity_gb{0.0};      // Total HBM (GB)

    // Inter-node communication (for PDD KV transfer)
    double network_bandwidth_gbps{0.0};  // NIC / NVLink bandwidth (GB/s)
    double network_latency_ms{0.0};      // One-way latency (ms)
};

// ── Predefined profiles ─────────────────────────────────────────────────────

/// NVIDIA H100 SXM (80 GB HBM3)
inline HardwareProfile H100_PROFILE() {
    return {
        .name                   = "H100_SXM",
        .compute_tflops         = 989.0,    // FP16 Tensor Core peak
        .memory_bandwidth_gbps  = 3350.0,   // HBM3
        .memory_capacity_gb     = 80.0,
        .network_bandwidth_gbps = 450.0,    // NVLink 4.0 (bi-directional)
        .network_latency_ms     = 0.005
    };
}

/// NVIDIA A100 (80 GB HBM2e)
inline HardwareProfile A100_PROFILE() {
    return {
        .name                   = "A100_80GB",
        .compute_tflops         = 312.0,
        .memory_bandwidth_gbps  = 2039.0,
        .memory_capacity_gb     = 80.0,
        .network_bandwidth_gbps = 300.0,
        .network_latency_ms     = 0.01
    };
}

/// NVIDIA L4 (24 GB GDDR6)
inline HardwareProfile L4_PROFILE() {
    return {
        .name                   = "L4",
        .compute_tflops         = 121.0,
        .memory_bandwidth_gbps  = 300.0,
        .memory_capacity_gb     = 24.0,
        .network_bandwidth_gbps = 25.0,     // PCIe Gen4 x16
        .network_latency_ms     = 0.1
    };
}

/// Construct a custom profile from Python.
inline HardwareProfile make_profile(
    const std::string& name,
    double compute, double mem_bw, double mem_cap,
    double net_bw, double net_lat)
{
    return {name, compute, mem_bw, mem_cap, net_bw, net_lat};
}

} // namespace llmsimbench
