# LLM Sim Bench

> **High-fidelity behavioral simulator for LLM inference serving systems.**

A hybrid C++20/Python simulator that models the complete lifecycle of Large Language Model inference requests — from prefill through decode — with accurate KV cache dynamics, Prefill-Decode Disaggregation (PDD), and pluggable scheduling policies.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Python Control Layer                      │
│  ┌──────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │ sweep.py │  │ clustering.py  │  │    security.py       │  │
│  └────┬─────┘  └───────┬────────┘  └──────────┬───────────┘  │
│       │                │                       │              │
│  ─────┴────────────────┴───────────────────────┴──────────── │
│                    pybind11 Bindings                           │
│              (llmsimbench_core module)                         │
│  ────────────────────────────────────────────────────────────  │
│                                                               │
│  ┌─────────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐ │
│  │ Simulation  │  │ Scheduler│  │  KVCache   │  │ Hardware │ │
│  │   Engine    │──│ (FCFS /  │  │ (Random /  │  │ Profile  │ │
│  │   (DES)     │  │ Priority │  │  LRU /     │  │ (H100 /  │ │
│  │             │  │ / Batch) │  │  Attn)     │  │  A100 /  │ │
│  └──────┬──────┘  └──────────┘  └───────────┘  │  L4)     │ │
│         │                                        └──────────┘ │
│  ┌──────┴──────┐  ┌────────────┐                              │
│  │   Prefill   │  │   Decode   │                              │
│  │   Engine    │──│   Engine   │                              │
│  │ (compute)   │  │ (bandwidth)│                              │
│  └─────────────┘  └────────────┘                              │
│                  C++20 Core                                   │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **SimulationEngine** | Discrete-event simulation loop. Processes events in time order, orchestrates request lifecycle. |
| **Request** | State machine: `PENDING → IN_PREFILL → IN_DECODE → COMPLETED`. Tracks per-phase latency. |
| **KVCache** | Behavioral cache model with hit/miss tracking and pluggable eviction policies. |
| **PrefillEngine** | Models compute-bound parallel prefill. Latency ∝ `2 × params × tokens / TFLOPS`. |
| **DecodeEngine** | Models bandwidth-bound sequential decode. Latency ∝ `2 × params / memory_BW`. |
| **Scheduler** | Polymorphic: FCFS, Priority (weighted scoring), Continuous Batching. |
| **HardwareProfile** | GPU specs (TFLOPS, HBM bandwidth, capacity, network). Predefined: H100, A100, L4. |

---

## Build

### Prerequisites

- **CMake** ≥ 3.21
- **C++20** compatible compiler (GCC 11+, Clang 14+, MSVC 2022+)
- **Python** ≥ 3.10 (recommended: latest stable)
- **pybind11** (auto-fetched by CMake if not installed)

### Compile

```bash
# Clone and navigate
cd llm_sim_bench/llmsimbench

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Verify
./build/test_des_engine        # C++ unit tests
python -c "import sys; sys.path.insert(0, 'python'); import llmsimbench_core; print(llmsimbench_core.__version__)"
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Python API

```python
import sys
sys.path.insert(0, "python")
import llmsimbench_core as lsb

# Create hardware profiles (PDD: H100 for prefill, L4 for decode)
prefill_hw = lsb.H100_PROFILE()
decode_hw  = lsb.L4_PROFILE()

# Choose a scheduler
scheduler = lsb.FirstComeFirstServeScheduler()

# Build the simulation engine
engine = lsb.SimulationEngine(
    scheduler=scheduler,
    prefill_hw=prefill_hw,
    decode_hw=decode_hw,
    model_params_B=7.0,
    kv_cache_capacity=4096,
    kv_transfer_latency_ms=0.5,
)

# Inject requests
engine.add_request(id=1, prompt_len=128, gen_len=64, arrival_time=0.0)
engine.add_request(id=2, prompt_len=256, gen_len=32, arrival_time=0.1)

# Run to completion
engine.run()

# Inspect results
stats = engine.stats()
print(f"Completed: {stats.total_requests_completed}")
print(f"Throughput: {stats.throughput_tokens_per_sec():.1f} tok/s")
print(f"Avg latency: {stats.avg_e2e_latency():.4f}s")

for req in engine.get_completed_requests():
    print(f"  Request {req.get_id()}: e2e={req.get_end_to_end_latency():.4f}s")
```

### Parameter Sweep

```bash
# Full sweep (3 schedulers × 4 eviction policies = 12 configs)
python python/sweep.py --config configs/default.json

# Quick mode (fewer requests for fast iteration)
python python/sweep.py --quick

# Results → results/cdf_comparison.png, results/roofline.png
```

### Clustering Analysis

```bash
# Run after sweep.py has generated results
python python/analysis/clustering.py --csv results/sweep_results.csv

# Results → results/clusters.png, results/traffic_clusters.png
```

---

## Configuration

All parameters are defined in `configs/default.json`:

```jsonc
{
  "simulation": {
    "model_params_B": 7.0,           // Model size (billions of params)
    "kv_cache_capacity": 4096,       // Max KV cache entries per request
    "kv_transfer_latency_ms": 0.5    // PDD inter-node transfer latency
  },
  "prefill_hardware": { ... },       // GPU profile for prefill engine
  "decode_hardware":  { ... },       // GPU profile for decode engine
  "scheduler": {
    "type": "FCFS",                  // FCFS | Priority | ContinuousBatching
    "priority_prompt_weight": 0.1,
    "priority_generation_weight": 0.5,
    "continuous_batching_max_batch": 8
  },
  "eviction_policy": "LRU",          // LRU | Random | AttentionGuided | ""
  "workload": {
    "num_requests": 50,
    "prompt_length_min": 64,
    "prompt_length_max": 2048,
    "generation_length_min": 16,
    "generation_length_max": 512,
    "arrival_rate_rps": 10.0,
    "seed": 42
  }
}
```

---

## Schedulers

| Scheduler | Strategy | Best For |
|-----------|----------|----------|
| **FCFS** | First-come, first-served | Fairness; predictable ordering |
| **Priority** | Weighted score: `prompt×w₁ + gen×w₂` (lower = higher priority) | Minimising average latency |
| **ContinuousBatching** | FCFS within batches of configurable size | Maximising GPU utilisation |

## Eviction Policies

| Policy | Strategy | Trade-off |
|--------|----------|-----------|
| **None** | Unlimited cache (no eviction) | High memory, no data loss |
| **Random** | Evict random entries | Simple, unpredictable |
| **LRU** | Evict least-recently-used | Good temporal locality |
| **AttentionGuided** | Evict lowest attention-score entries | Best quality preservation |

---

## Security Considerations

The optional `security.py` module demonstrates production-aware practices:

- **Input Validation**: Bounds checking on all request and hardware parameters with descriptive error messages.
- **Data Leakage Model**: Simulates KV cache data leakage probability. Configurable per-request leak probability for studying privacy/memory trade-offs.
- **Secure Mode**: `SecureSimulationEngine` wrapper enables all security features with a single flag.

```python
from security import SecureSimulationEngine

engine = SecureSimulationEngine(
    scheduler=scheduler,
    prefill_hw=prefill_hw,
    decode_hw=decode_hw,
    secure_mode=True,
    leak_probability=0.01,
)
engine.add_request(1, 128, 64)
engine.run()
engine.print_security_report()
```

---

## Project Structure

```
llmsimbench/
├── CMakeLists.txt              # Build system (CMake 3.21+, C++20)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── configs/
│   └── default.json            # Default simulation configuration
├── python/
│   ├── sweep.py                # Parameter sweep driver
│   ├── security.py             # Optional security module
│   └── analysis/
│       └── clustering.py       # Post-simulation clustering analysis
├── src/
│   ├── bindings/
│   │   └── pybind_module.cpp   # pybind11 bindings (GIL, smart ptrs)
│   ├── core/
│   │   ├── event.hpp           # EventType enum, Event struct
│   │   ├── event_queue.hpp     # Min-heap priority queue
│   │   ├── workload.hpp        # Request, KVCache, EvictionPolicy
│   │   └── scheduler.hpp       # FCFS, Priority, ContinuousBatching
│   └── models/
│       ├── hardware_profile.hpp # H100, A100, L4 profiles
│       └── timing_model.hpp     # PrefillEngine, DecodeEngine, SimulationEngine
├── tests/
│   └── test_des_engine.cpp     # C++ unit tests
└── results/
    └── .gitkeep                # Output directory
```

---

## License

This project is for educational and research purposes.
