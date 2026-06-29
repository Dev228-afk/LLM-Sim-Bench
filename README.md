# LLMSimBench: Agentic LLM Serving Performance Simulator

[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6%2B-yellow.svg?style=flat-square&logo=javascript)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg?style=flat-square&logo=python)](https://www.python.org/)

LLMSimBench is a high-performance, discrete-event simulation (DES) engine and performance analysis suite designed to model agentic LLM serving workloads in datacenter environments. It allows researchers and engineers to visually model, profile, and optimize LLM serving infrastructures right in their browser.

The simulator models the complex, non-linear request lifecycles of agentic AI workflows (recursive loops, multi-step thinking, tool calls, and multi-RAG queries) across heterogeneous hardware configurations. It accurately simulates hardware bottlenecks like Prefill Compute saturation vs. Decode Memory Bandwidth limitations.

---

## Key Capabilities

* **Web Worker DES Engine**: A pure JavaScript Discrete Event Simulation (DES) engine running in a background Web Worker, capable of tracking over 50,000 state transitions per second while keeping the UI responsive at 60fps.
* **Accurate Hardware Modeling**: Simulates custom hardware profiles, calculating precise FLOPs for compute-bound prefill and GB/s memory bandwidth limits for decode steps.
* **Tensor Parallelism (TP) & Quantization**: Models distributed multi-GPU clusters (TP scaling) and supports detailed memory modeling for `FP16`, `BF16`, `FP8`, `INT8`, and `INT4` data types.
* **Pluggable Schedulers**: 
  * **First-Come-First-Serve (FCFS)**: Baseline sequential scheduling.
  * **Continuous Batching**: Fills the decode batch up to the `Max Decode Batch` size.
  * **Priority Continuous Batching**: Preempts lower-priority requests for high-priority short prompts.
* **VRAM Memory Management**: Explicitly models KV Cache memory consumption, automatically triggering memory pressure evictions when the GPU VRAM budget is exceeded, complete with garbage collection to prevent memory leaks.
* **Agentic Workload Characterization**: Models structural traffic paths for complex workflows like ToolCall, Multi-Step CoT, and Multi-RAG.

---

## Interactive GUI Dashboard

The simulator includes a stunning, interactive HTML5/CSS3/JavaScript GUI that runs completely locally in your browser.

### Features
* **Real-Time Visualizations**: Watch the simulation unfold with live charts for Throughput, Tail Latency, Queue Evacuation, and Cumulative Tokens.
* **Hardware Saliency Analysis**: See exactly where your bottlenecks are with execution profiling broken down into *Prefill Compute*, *Decode Memory*, and *Overhead* (queue wait times).
* **Latency CDFs**: Analyze tail latencies (P50, P95, P99) under different scheduling algorithms.
* **Custom Configuration**: Fine-tune hardware specs (e.g., GB300 vs H100 vs L4), workload generation parameters, arrival rates, and scheduler policies on the fly.

### Running the GUI

To launch the dashboard, simply serve the `gui` folder locally:

```bash
# Use Python's built-in HTTP server
python -m http.server 8000 --directory gui

# Open http://localhost:8000 in your browser
```

---

## Simulation Architecture

The Javascript DES Engine (`gui/sim_worker.js`) is strictly event-driven. The lifecycle of a request is governed by the following core events:
1. `REQUEST_ARRIVAL`: Calculates prompt/generation length and enqueues the request.
2. `PREFILL_COMPLETE`: Simulates the Time-To-First-Token (TTFT) compute latency based on hardware TFLOPS.
3. `KV_TRANSFER_COMPLETE`: Simulates the PCIe/NVLink overhead of moving KV caches (if using Disaggregated architectures).
4. `DECODE_STEP_COMPLETE`: Iteratively generates tokens based on Memory Bandwidth (GB/s), bounded by the active Continuous Batching size.

---

## Evaluation Metrics

The simulator exposes highly accurate metrics for LLM Serving Performance:
* **Throughput (tokens/sec)**: Total tokens generated per second across the cluster.
* **Time To First Token (TTFT)**: End-to-End prefill latency, accurately reflecting both the strict GPU compute time and the time spent waiting in queues (Overhead).
* **Decode Latency**: Time spent generating tokens after prefill finishes.
* **KV Cache Evictions**: Counts how many times requests were preempted due to VRAM exhaustion and forced to re-prefill.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
