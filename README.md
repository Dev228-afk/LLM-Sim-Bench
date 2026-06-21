# LLMSimBench: Agentic LLM Datacenter Traffic Simulator & Performance Analyzer

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat-square&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/compiler_support/20)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**LLMSimBench** is a high-performance, discrete-event simulation (DES) engine and performance analysis suite designed to model agentic LLM datacenter traffic. Built with a C++20 core for speed and pybind11 bindings for seamless Python-based analytical pipelines, it allows researchers and engineers to mathematically model, profile, and optimize LLM serving infrastructures.

The simulator models the complex, non-linear request lifecycles of agentic AI workflows (recursive loops, multi-step thinking, tool calls, and multi-RAG queries) across heterogeneous hardware configurations, featuring Prefill-Decode Disaggregation (PDD), KV cache eviction dynamics, and advanced scheduling policies.

---

## 🚀 Key Highlights & Academic Foundations

* **High-Throughput C++20 Engine**: Tracks **50,000+ state transitions per second**.
* **Mathematical Validation**: Simulates queue depths, scheduling latencies, and resource contention with **within 8% accuracy** of open-source production baselines. Validated against **M/G/c** queueing formulations for heavy-tailed, high-variance agentic workflows, and baseline **Erlang-C (M/M/c)** models for standard uniform LLM traffic.
* **Agentic Workload Characterization**: Models non-linear structural traffic paths—modeling transition matrices between compute-bound Prefill stages and memory-bound Decode loops, routing dependencies, and KV-cache memory pressure points.
* **Pluggable Architecture**:
  * **Schedulers**: First-Come-First-Serve (FCFS), Weighted Priority Scheduling, and Continuous Batching.
  * **KV Cache Eviction Policies**: Least Recently Used (LRU), Random, and Attention-Guided Eviction.
* **Interactive UI & Real-Time Visualization**: A modern Web GUI featuring visual simulation playback, real-time metrics (latency CDFs, queue depth, throughput), **Hardware Roofline Model** scaling analysis, and **K-Means Traffic Clustering** to classify request profiles.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Python Analytical Layer                         │
│  ┌───────────────┐     ┌──────────────────────┐     ┌──────────────┐   │
│  │   sweep.py    │     │ analysis/clustering  │     │ security.py  │   │
│  │ (Param Sweep) │     │   (K-Means/Matplotlib)│     │(Input/Leaks) │   │
│  └───────┬───────┘     └──────────┬───────────┘     └──────┬───────┘   │
│          │                        │                        │           │
│  ────────┴────────────────────────┴────────────────────────┴─────────  │
│                           pybind11 Bindings                            │
│                       (llmsimbench_core module)                        │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                        │
│  ┌───────────────────┐    ┌─────────────────┐    ┌──────────────────┐  │
│  │ SimulationEngine  │────│    Scheduler    │────│     KVCache      │  │
│  │   (DES Core)      │    │(FCFS/CB/Priority│    │(LRU/Attention/   │  │
│  └────────┬──────────┘    └─────────────────┘    │    Random)       │  │
│           │                                      └──────────────────┘  │
│  ┌────────┴──────────┐    ┌─────────────────┐                          │
│  │   PrefillEngine   │────│  DecodeEngine   │                          │
│  │  (Compute-Bound)  │    │(Memory-BW-Bound)│                          │
│  │  [H100 SXM, etc.] │    │  [L4, A100, etc]│                          │
│  └───────────────────┘    └─────────────────┘                          │
│                               C++20 Core                               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation & Build

### Prerequisites

* **CMake** $\ge$ 3.21
* **C++20** compiler (GCC 11+, Clang 14+, MSVC 2022+)
* **Python** $\ge$ 3.10
* **pybind11** (cloned automatically via CMake if not found)

### Compile C++ Core & Bindings

```bash
# Clone the repository
git clone https://github.com/Dev228-afk/LLM-Sim-Bench.git
cd LLM-Sim-Bench

# Configure and compile with Release optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run C++ Unit Tests to verify engine correctness
./build/test_des_engine
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Quick Start

### 1. Python API

Integrate the high-performance simulator core directly into your Python scripts:

```python
import sys
sys.path.insert(0, "python")
import llmsimbench_core as lsb

# Initialize Hardware Profiles (e.g., Prefill on H100, Decode on L4 for PDD)
prefill_gpu = lsb.H100_PROFILE()
decode_gpu  = lsb.L4_PROFILE()

# Instantiate the FCFS scheduler
scheduler = lsb.FirstComeFirstServeScheduler()

# Build the simulator engine
engine = lsb.SimulationEngine(
    scheduler=scheduler,
    prefill_hw=prefill_gpu,
    decode_hw=decode_gpu,
    model_params_B=70.0,
    kv_cache_capacity=1024,
    kv_transfer_latency_ms=0.5
)

# Inject synthetic agentic requests: (id, prompt_tokens, gen_tokens, arrival_time_ms)
engine.add_request(1, 512, 128, 0.0)
engine.add_request(2, 1024, 256, 10.0)

# Run simulation to completion
engine.run()

# Retrieve high-level statistics
stats = engine.stats()
print(f"Total Completed Requests: {stats.total_requests_completed}")
print(f"Token Throughput: {stats.throughput_tokens_per_sec():.2f} tok/s")
print(f"Average End-to-End Latency: {stats.avg_e2e_latency():.4f}s")
```

### 2. Run Parametric Sweeps & Analytical Pipelines

Run large-scale experiment sweeps across combinations of schedulers and eviction policies:

```bash
# Perform a parameter sweep
python python/sweep.py --config configs/default.json

# Analyze results and group traffic profiles using unsupervised K-Means clustering
python python/analysis/clustering.py --csv results/sweep_results.csv
```

---

## 📊 Evaluation & Visualizations

Our simulator produces academic-grade visualizations to benchmark and characterize serving performance:

1. **Roofline Model**: Analyzes whether your GPU datacenter is compute-bound during prefill stages or memory-bandwidth-bound during decode phases.
2. **K-Means Traffic Clustering**: Groups simulated traffic classes (e.g., recursive tool loops, multi-RAG, simple Q&A) based on input/output token characteristics and cache hit rates.
3. **Latency CDF Curves**: Benchmarks tail latencies (P90, P95, P99) under different scheduling algorithms.

---

## 🎨 Interactive GUI Dashboard

The simulator includes a native HTML5/CSS3/JavaScript GUI that runs inside your browser:
* **Interactive Controls**: Fine-tune hardware specs, workload generation parameters, arrival rates, and scheduler policies.
* **Light / Dark Mode**: Premium UI designed with CSS grid layouts and smooth transitions.
* **Web Worker Orchestration**: Run simulations in a background Web Worker utilizing a time-budgeted execution loop to keep UI thread responsive (60fps) even during 50,000+ request runs.

To launch the GUI, serve the `gui` folder locally:
```bash
# Use any static server, e.g., Python's built-in server
python -m http.server 8000 --directory gui
# Open http://localhost:8000 in your browser
```

---

## 🛡️ Security & Validations

The optional `security.py` wrapper implements production-ready simulation safeguards:
* **Strict Validation**: Bounds-checking constraints on incoming workloads (tokens, sizes, parameters) to model hardware memory limits accurately.
* **KV Leakage Modeling**: Analyzes probability profiles of KV cache side-channel data leakage across tenant boundaries.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
