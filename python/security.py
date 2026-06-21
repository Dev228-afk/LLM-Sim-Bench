#!/usr/bin/env python3
"""
security.py — Optional security enhancements for LLM Sim Bench.

Demonstrates production-aware security practices in a research simulator:

  1. Input Validation   — Strict bounds checking on all simulation parameters.
  2. Data Leakage Model — Simulates KV cache data leakage probability to
                          illustrate privacy/memory trade-offs.
  3. Secure Mode        — A wrapper that enables all security features with
                          a single flag.

Usage:
    from security import SecureSimulationEngine, validate_request_params

    # Standalone validation
    validate_request_params(prompt_length=128, generation_length=64)

    # Secure wrapper (enables validation + leakage tracking)
    engine = SecureSimulationEngine(
        scheduler=scheduler,
        prefill_hw=prefill_hw,
        decode_hw=decode_hw,
        secure_mode=True,
        leak_probability=0.01,
    )
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Locate the compiled extension ────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

try:
    import llmsimbench_core as lsb
except ImportError:
    lsb = None  # Allow import for type checking even without the extension


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Input Validation
# ═══════════════════════════════════════════════════════════════════════════════

# Reasonable bounds for simulation parameters
PARAM_BOUNDS = {
    "prompt_length":     (1, 1_000_000),
    "generation_length": (1, 1_000_000),
    "arrival_time":      (0.0, 1e9),
    "model_params_B":    (0.01, 10_000.0),
    "kv_cache_capacity": (1, 100_000_000),
    "compute_tflops":    (0.01, 100_000.0),
    "memory_bandwidth_gbps": (0.01, 100_000.0),
    "memory_capacity_gb":    (0.01, 100_000.0),
}


class ValidationError(ValueError):
    """Raised when a simulation parameter fails bounds checking."""
    pass


def validate_request_params(
    prompt_length: int,
    generation_length: int,
    arrival_time: float = 0.0,
    request_id: int | None = None,
) -> None:
    """
    Validate request parameters against reasonable bounds.

    Raises:
        ValidationError: If any parameter is out of bounds.
    """
    _check_int_bounds("prompt_length", prompt_length, *PARAM_BOUNDS["prompt_length"])
    _check_int_bounds("generation_length", generation_length, *PARAM_BOUNDS["generation_length"])
    _check_float_bounds("arrival_time", arrival_time, *PARAM_BOUNDS["arrival_time"])

    if request_id is not None and request_id < 0:
        raise ValidationError(f"request_id must be non-negative, got {request_id}")


def validate_hardware_profile(
    name: str,
    compute_tflops: float,
    memory_bandwidth_gbps: float,
    memory_capacity_gb: float,
) -> None:
    """Validate hardware profile parameters."""
    if not name or not name.strip():
        raise ValidationError("Hardware profile name must not be empty")

    _check_float_bounds("compute_tflops", compute_tflops, *PARAM_BOUNDS["compute_tflops"])
    _check_float_bounds("memory_bandwidth_gbps", memory_bandwidth_gbps, *PARAM_BOUNDS["memory_bandwidth_gbps"])
    _check_float_bounds("memory_capacity_gb", memory_capacity_gb, *PARAM_BOUNDS["memory_capacity_gb"])


def validate_config(config: dict[str, Any]) -> list[str]:
    """
    Validate a full simulation config dict.

    Returns a list of warning messages (empty if config is clean).
    """
    warnings_list: list[str] = []

    sim = config.get("simulation", {})
    if sim.get("model_params_B", 0) > 1000:
        warnings_list.append(
            f"model_params_B={sim['model_params_B']} is unusually large — "
            "simulation may be slow"
        )

    wl = config.get("workload", {})
    if wl.get("num_requests", 0) > 10000:
        warnings_list.append(
            f"num_requests={wl['num_requests']} — large workload, "
            "consider using --quick mode"
        )

    return warnings_list


# ── Internal helpers ─────────────────────────────────────────────────────────

def _check_int_bounds(name: str, value: int, lo: int, hi: int) -> None:
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    if value < lo or value > hi:
        raise ValidationError(f"{name} must be in [{lo}, {hi}], got {value}")


def _check_float_bounds(name: str, value: float, lo: float, hi: float) -> None:
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value).__name__}")
    if value < lo or value > hi:
        raise ValidationError(f"{name} must be in [{lo}, {hi}], got {value}")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Data Leakage Model
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LeakageTracker:
    """
    Tracks simulated KV cache data leakage events.

    In real systems, the KV cache contains representations of user prompts.
    If the cache is not properly isolated or cleared, data from one user's
    session could theoretically leak to another.  This model simulates that
    risk to study the privacy/memory trade-off.

    Each completed request has a configurable probability of "leaking" its
    cache contents.  This is purely for analysis — no real data is exposed.
    """

    leak_probability: float = 0.01
    total_completions: int = 0
    total_leaks: int = 0
    leaked_request_ids: list[int] = field(default_factory=list)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )

    def on_request_complete(self, request_id: int) -> bool:
        """
        Called when a request completes.  Returns True if a leak occurred.
        """
        self.total_completions += 1
        if self._rng.random() < self.leak_probability:
            self.total_leaks += 1
            self.leaked_request_ids.append(request_id)
            return True
        return False

    @property
    def leak_rate(self) -> float:
        """Observed leak rate (0.0 – 1.0)."""
        if self.total_completions == 0:
            return 0.0
        return self.total_leaks / self.total_completions

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for reporting."""
        return {
            "leak_probability_configured": self.leak_probability,
            "total_completions": self.total_completions,
            "total_leaks": self.total_leaks,
            "observed_leak_rate": self.leak_rate,
            "leaked_request_ids": self.leaked_request_ids.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"<LeakageTracker leaks={self.total_leaks}/{self.total_completions} "
            f"rate={self.leak_rate:.4f}>"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Secure Simulation Engine Wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class SecureSimulationEngine:
    """
    A security-enhanced wrapper around the C++ SimulationEngine.

    Features (when secure_mode=True):
      - Validates all request parameters before injection.
      - Tracks simulated data leakage events.
      - Provides a security report after simulation.
    """

    def __init__(
        self,
        scheduler: Any,
        prefill_hw: Any,
        decode_hw: Any,
        model_params_B: float = 7.0,
        kv_cache_capacity: int = 4096,
        kv_transfer_latency_ms: float = 0.5,
        secure_mode: bool = True,
        leak_probability: float = 0.01,
    ):
        if lsb is None:
            raise RuntimeError(
                "llmsimbench_core not available — build the C++ extension first"
            )

        self.secure_mode = secure_mode
        self.leakage_tracker = LeakageTracker(
            leak_probability=leak_probability
        ) if secure_mode else None

        self._engine = lsb.SimulationEngine(
            scheduler=scheduler,
            prefill_hw=prefill_hw,
            decode_hw=decode_hw,
            model_params_B=model_params_B,
            kv_cache_capacity=kv_cache_capacity,
            kv_transfer_latency_ms=kv_transfer_latency_ms,
        )
        self._request_ids: list[int] = []

    def add_request(
        self,
        request_id: int,
        prompt_length: int,
        generation_length: int,
        arrival_time: float = 0.0,
    ) -> None:
        """Add a request with optional validation."""
        if self.secure_mode:
            validate_request_params(
                prompt_length=prompt_length,
                generation_length=generation_length,
                arrival_time=arrival_time,
                request_id=request_id,
            )

        self._engine.add_request(request_id, prompt_length, generation_length, arrival_time)
        self._request_ids.append(request_id)

    def run(self) -> None:
        """Run the simulation to completion, tracking leakage."""
        self._engine.run()

        # Check for leakage on each completed request
        if self.leakage_tracker is not None:
            completed = self._engine.get_completed_requests()
            for req in completed:
                leaked = self.leakage_tracker.on_request_complete(req.get_id())
                if leaked:
                    warnings.warn(
                        f"[SECURITY] Simulated KV cache leak for request {req.get_id()}",
                        stacklevel=2,
                    )

    def step(self) -> bool:
        """Process a single event."""
        return self._engine.step()

    @property
    def engine(self) -> Any:
        """Access the underlying C++ engine."""
        return self._engine

    def security_report(self) -> dict[str, Any]:
        """
        Generate a security summary report.

        Includes leakage statistics and validation status.
        """
        report: dict[str, Any] = {
            "secure_mode": self.secure_mode,
            "total_requests_injected": len(self._request_ids),
        }

        if self.leakage_tracker is not None:
            report["leakage"] = self.leakage_tracker.summary()
        else:
            report["leakage"] = "disabled"

        stats = self._engine.stats()
        report["simulation_stats"] = {
            "completed": stats.total_requests_completed,
            "tokens_generated": stats.total_tokens_generated,
            "avg_latency": stats.avg_e2e_latency(),
        }

        return report

    def print_security_report(self) -> None:
        """Print a human-readable security report."""
        report = self.security_report()

        print("\n" + "=" * 50)
        print("  Security Report")
        print("=" * 50)
        print(f"  Secure mode:      {report['secure_mode']}")
        print(f"  Requests injected: {report['total_requests_injected']}")

        if isinstance(report["leakage"], dict):
            leak = report["leakage"]
            print(f"\n  [Data Leakage Model]")
            print(f"    Configured probability: {leak['leak_probability_configured']:.4f}")
            print(f"    Total completions:      {leak['total_completions']}")
            print(f"    Total leaks:            {leak['total_leaks']}")
            print(f"    Observed leak rate:      {leak['observed_leak_rate']:.4f}")
            if leak["leaked_request_ids"]:
                ids = leak["leaked_request_ids"][:10]
                suffix = f" ... (+{len(leak['leaked_request_ids']) - 10} more)" if len(leak["leaked_request_ids"]) > 10 else ""
                print(f"    Leaked request IDs:     {ids}{suffix}")
        else:
            print(f"\n  Leakage tracking: {report['leakage']}")

        sim = report["simulation_stats"]
        print(f"\n  [Simulation Summary]")
        print(f"    Completed:        {sim['completed']}")
        print(f"    Tokens generated: {sim['tokens_generated']}")
        print(f"    Avg latency:      {sim['avg_latency']:.4f}s")
        print("=" * 50 + "\n")

    def __repr__(self) -> str:
        mode = "SECURE" if self.secure_mode else "STANDARD"
        return f"<SecureSimulationEngine mode={mode} requests={len(self._request_ids)}>"
