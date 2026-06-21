#!/usr/bin/env python3
"""
sweep.py — Parameter sweep driver for LLM Sim Bench.

Generates a grid of simulation configurations (scheduler × eviction policy ×
hardware), runs each to completion, collects results, and produces comparison
plots.

Usage:
    python sweep.py                          # uses configs/default.json
    python sweep.py --config my_config.json  # custom config
    python sweep.py --quick                  # fast sweep (fewer requests)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── Locate the compiled extension ────────────────────────────────────────────
# The pybind11 module is built into the python/ directory alongside this file.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

try:
    import llmsimbench_core as lsb
except ImportError as exc:
    print(
        "ERROR: Could not import llmsimbench_core.\n"
        "       Build the C++ extension first:\n"
        "         cmake -B build && cmake --build build\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON configuration file."""
    with open(path) as f:
        return json.load(f)


def make_hardware_profile(cfg: dict[str, Any]) -> lsb.HardwareProfile:
    """Build a HardwareProfile from a config dict."""
    return lsb.make_profile(
        cfg["name"],
        cfg["compute_tflops"],
        cfg["memory_bandwidth_gbps"],
        cfg["memory_capacity_gb"],
        cfg["network_bandwidth_gbps"],
        cfg["network_latency_ms"],
    )


def make_scheduler(stype: str, cfg: dict[str, Any]) -> Any:
    """Instantiate a scheduler by type name."""
    if stype == "FCFS":
        return lsb.FirstComeFirstServeScheduler()
    elif stype == "Priority":
        return lsb.PriorityScheduler(
            cfg.get("priority_prompt_weight", 0.1),
            cfg.get("priority_generation_weight", 0.5),
        )
    elif stype == "ContinuousBatching":
        return lsb.ContinuousBatchingScheduler(
            cfg.get("continuous_batching_max_batch", 8)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Workload generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_workload(
    num_requests: int,
    prompt_min: int,
    prompt_max: int,
    gen_min: int,
    gen_max: int,
    arrival_rate: float,
    seed: int = 42,
    is_agentic: bool = False,
) -> list[tuple[int, int, int, float]]:
    """
    Generate a synthetic workload.

    Returns:
        List of (id, prompt_length, generation_length, arrival_time) tuples.
    """
    rng = np.random.default_rng(seed)
    
    if is_agentic:
        # Agentic workflows exhibit heavy-tailed distributions (M/G/c queues).
        # We model this using a bounded Pareto distribution.
        pareto_prompts = rng.pareto(1.5, size=num_requests)
        prompt_lens = np.clip(np.floor(prompt_min * (1 + pareto_prompts)), prompt_min, prompt_max)
        
        pareto_gens = rng.pareto(1.2, size=num_requests)
        gen_lens = np.clip(np.floor(gen_min * (1 + pareto_gens)), gen_min, gen_max)
    else:
        # Standard baseline (Uniform / M/M/c behavior)
        prompt_lens = rng.integers(prompt_min, prompt_max + 1, size=num_requests)
        gen_lens = rng.integers(gen_min, gen_max + 1, size=num_requests)

    # Poisson inter-arrival times
    inter_arrivals = rng.exponential(1.0 / arrival_rate, size=num_requests)
    arrival_times = np.cumsum(inter_arrivals)

    workload = []
    for i in range(num_requests):
        workload.append((
            i + 1,
            int(prompt_lens[i]),
            int(gen_lens[i]),
            float(arrival_times[i]),
        ))
    return workload


# ═══════════════════════════════════════════════════════════════════════════════
#  Single simulation run
# ═══════════════════════════════════════════════════════════════════════════════


def run_simulation(
    scheduler: Any,
    prefill_hw: lsb.HardwareProfile,
    decode_hw: lsb.HardwareProfile,
    workload: list[tuple[int, int, int, float]],
    model_params_B: float = 7.0,
    kv_cache_capacity: int = 4096,
    kv_transfer_latency_ms: float = 0.5,
    eviction_policy: str = "",
) -> dict[str, Any]:
    """Run a single simulation and return results."""
    engine = lsb.SimulationEngine(
        scheduler=scheduler,
        prefill_hw=prefill_hw,
        decode_hw=decode_hw,
        model_params_B=model_params_B,
        kv_cache_capacity=kv_cache_capacity,
        kv_transfer_latency_ms=kv_transfer_latency_ms,
    )

    if eviction_policy:
        engine.set_eviction_policy_for_all(eviction_policy)

    # Inject all requests
    for req_id, prompt_len, gen_len, arr_time in workload:
        engine.add_request(req_id, prompt_len, gen_len, arr_time)

    # Run to completion
    t0 = time.perf_counter()
    engine.run()
    wall_time = time.perf_counter() - t0

    # Collect stats
    stats = engine.stats()
    completed = engine.get_completed_requests()

    per_request = []
    for req in completed:
        per_request.append({
            "request_id": req.get_id(),
            "prompt_length": req.get_prompt_length(),
            "generation_length": req.get_generation_length(),
            "arrival_time": req.get_arrival_time(),
            "prefill_latency": req.get_prefill_latency(),
            "decode_latency": req.get_decode_latency(),
            "e2e_latency": req.get_end_to_end_latency(),
            "completion_time": req.get_completion_time(),
        })

    return {
        "scheduler": engine.scheduler_name(),
        "eviction_policy": eviction_policy or "None",
        "total_completed": stats.total_requests_completed,
        "total_tokens": stats.total_tokens_generated,
        "avg_e2e_latency": stats.avg_e2e_latency(),
        "max_e2e_latency": stats.max_e2e_latency,
        "throughput_tps": stats.throughput_tokens_per_sec(),
        "total_cache_hits": stats.total_cache_hits,
        "total_cache_misses": stats.total_cache_misses,
        "peak_queue_depth": stats.peak_pending_queue_depth,
        "wall_time_s": wall_time,
        "per_request": per_request,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Sweep logic
# ═══════════════════════════════════════════════════════════════════════════════

SCHEDULERS = ["FCFS", "Priority", "ContinuousBatching"]
EVICTION_POLICIES = ["", "Random", "LRU", "AttentionGuided"]


def run_sweep(config: dict[str, Any], quick: bool = False) -> pd.DataFrame:
    """
    Run a full parameter sweep across schedulers and eviction policies.
    """
    sim_cfg = config["simulation"]
    wl_cfg = config["workload"]
    sched_cfg = config.get("scheduler", {})

    prefill_hw = make_hardware_profile(config["prefill_hardware"])
    decode_hw = make_hardware_profile(config["decode_hardware"])

    num_requests = 10 if quick else wl_cfg["num_requests"]

    workload = generate_workload(
        num_requests=num_requests,
        prompt_min=wl_cfg["prompt_length_min"],
        prompt_max=wl_cfg["prompt_length_max"],
        gen_min=wl_cfg["generation_length_min"],
        gen_max=wl_cfg["generation_length_max"],
        arrival_rate=wl_cfg["arrival_rate_rps"],
        seed=wl_cfg.get("seed", 42),
        is_agentic=(wl_cfg.get("traffic_profile", "Agentic") == "Agentic"),
    )

    all_results = []
    total_runs = len(SCHEDULERS) * len(EVICTION_POLICIES)
    run_idx = 0

    for sched_type in SCHEDULERS:
        scheduler = make_scheduler(sched_type, sched_cfg)

        for eviction in EVICTION_POLICIES:
            run_idx += 1
            label = f"{sched_type} / {eviction or 'NoEviction'}"
            print(f"  [{run_idx}/{total_runs}] {label} ... ", end="", flush=True)

            result = run_simulation(
                scheduler=scheduler,
                prefill_hw=prefill_hw,
                decode_hw=decode_hw,
                workload=workload,
                model_params_B=sim_cfg["model_params_B"],
                kv_cache_capacity=sim_cfg["kv_cache_capacity"],
                kv_transfer_latency_ms=sim_cfg["kv_transfer_latency_ms"],
                eviction_policy=eviction,
            )

            # Flatten per-request data with config label
            for pr in result["per_request"]:
                pr["scheduler"] = result["scheduler"]
                pr["eviction_policy"] = result["eviction_policy"]
                all_results.append(pr)

            print(
                f"done  (avg_lat={result['avg_e2e_latency']:.4f}s  "
                f"tput={result['throughput_tps']:.1f} tok/s  "
                f"wall={result['wall_time_s']:.3f}s)"
            )

    return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def plot_latency_cdf(df: pd.DataFrame, output_path: Path) -> None:
    """Plot CDF of end-to-end latency for each configuration."""
    fig, ax = plt.subplots(figsize=(12, 7))

    configs = df.groupby(["scheduler", "eviction_policy"])
    colors = plt.cm.tab20(np.linspace(0, 1, len(configs)))

    for (sched, evict), color in zip(configs.groups.keys(), colors):
        group = configs.get_group((sched, evict))
        latencies = np.sort(group["e2e_latency"].values)
        cdf = np.arange(1, len(latencies) + 1) / len(latencies)
        label = f"{sched} / {evict}"
        ax.plot(latencies, cdf, label=label, color=color, linewidth=2)

    ax.set_xlabel("End-to-End Latency (s)", fontsize=13)
    ax.set_ylabel("CDF", fontsize=13)
    ax.set_title("Latency CDF — Scheduler × Eviction Policy", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved {output_path}")


def plot_roofline(df: pd.DataFrame, output_path: Path) -> None:
    """
    Roofline-style plot: throughput (tok/s) vs average prompt length,
    grouped by scheduler.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    summary = (
        df.groupby(["scheduler", "eviction_policy"])
        .agg(
            avg_prompt=("prompt_length", "mean"),
            total_tokens=("generation_length", "sum"),
            total_decode_time=("decode_latency", "sum"),
        )
        .reset_index()
    )
    summary["throughput"] = summary["total_tokens"] / summary["total_decode_time"]

    markers = {"FCFS": "o", "Priority": "s", "ContinuousBatching(batch=8)": "^"}

    for _, row in summary.iterrows():
        marker = markers.get(row["scheduler"], "D")
        label = f"{row['scheduler']} / {row['eviction_policy']}"
        ax.scatter(
            row["avg_prompt"],
            row["throughput"],
            s=120,
            marker=marker,
            label=label,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax.set_xlabel("Average Prompt Length (tokens)", fontsize=13)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=13)
    ax.set_title("Roofline — Throughput vs. Compute Intensity", fontsize=15, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Sim Bench — Parameter Sweep Driver"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_THIS_DIR.parent / "configs" / "default.json"),
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use fewer requests for fast iteration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: from config)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  LLM Sim Bench — Parameter Sweep  (v{lsb.__version__})")
    print(f"{'='*60}\n")

    config = load_config(args.config)
    print(f"  Config: {args.config}")

    results_dir = Path(
        args.output_dir or config.get("output", {}).get("results_dir", "results")
    )
    if not results_dir.is_absolute():
        results_dir = _THIS_DIR.parent / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Output: {results_dir}")
    print(f"  Mode:   {'quick' if args.quick else 'full'}")
    print()

    # Run the sweep
    print("Running simulations...\n")
    df = run_sweep(config, quick=args.quick)

    # Save raw results
    csv_path = results_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  → Saved raw results to {csv_path}")

    # Generate plots
    print("\nGenerating plots...\n")
    plot_latency_cdf(df, results_dir / "cdf_comparison.png")
    plot_roofline(df, results_dir / "roofline.png")

    # Summary statistics
    print("\n" + "=" * 60)
    print("  Sweep Summary")
    print("=" * 60)
    summary = (
        df.groupby(["scheduler", "eviction_policy"])
        .agg(
            count=("e2e_latency", "count"),
            avg_latency=("e2e_latency", "mean"),
            p95_latency=("e2e_latency", lambda x: np.percentile(x, 95)),
            p99_latency=("e2e_latency", lambda x: np.percentile(x, 99)),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
