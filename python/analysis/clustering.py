#!/usr/bin/env python3
"""
clustering.py — Post-simulation performance regime analysis.

Loads sweep results CSV, extracts feature vectors, runs KMeans clustering
to identify distinct performance regimes, and generates visualisation plots.

Usage:
    python -m analysis.clustering                            # default path
    python -m analysis.clustering --csv ../results/sweep_results.csv
    python -m analysis.clustering --clusters 4               # override K
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading & feature engineering
# ═══════════════════════════════════════════════════════════════════════════════


def load_sweep_results(csv_path: Path) -> pd.DataFrame:
    """Load the per-request sweep results CSV."""
    if not csv_path.exists():
        print(f"ERROR: Results file not found: {csv_path}", file=sys.stderr)
        print("       Run sweep.py first to generate results.", file=sys.stderr)
        raise SystemExit(1)
    return pd.read_csv(csv_path)


def build_config_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-request data into per-configuration feature vectors.

    Features:
        - avg_e2e_latency     : Mean end-to-end latency (s)
        - p95_e2e_latency     : 95th percentile latency (s)
        - throughput_tps      : Tokens generated per second of decode time
        - avg_prefill_latency : Mean prefill phase latency (s)
        - cache_hit_rate      : (Approximated from latency variance)
        - avg_queue_wait      : Mean time from arrival to prefill start
    """
    features = (
        df.groupby(["scheduler", "eviction_policy"])
        .agg(
            avg_e2e_latency=("e2e_latency", "mean"),
            p95_e2e_latency=("e2e_latency", lambda x: np.percentile(x, 95)),
            std_e2e_latency=("e2e_latency", "std"),
            avg_prefill_latency=("prefill_latency", "mean"),
            avg_decode_latency=("decode_latency", "mean"),
            total_tokens=("generation_length", "sum"),
            total_decode_time=("decode_latency", "sum"),
            avg_prompt_length=("prompt_length", "mean"),
            num_requests=("e2e_latency", "count"),
        )
        .reset_index()
    )

    features["throughput_tps"] = features["total_tokens"] / features["total_decode_time"]
    features["latency_cv"] = features["std_e2e_latency"] / features["avg_e2e_latency"]

    return features


# ═══════════════════════════════════════════════════════════════════════════════
#  Clustering
# ═══════════════════════════════════════════════════════════════════════════════


def cluster_configs(
    features: pd.DataFrame,
    n_clusters: int = 3,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run KMeans clustering on configuration feature vectors.

    Returns the features DataFrame with an added 'cluster' column.
    """
    if feature_cols is None:
        feature_cols = [
            "avg_e2e_latency",
            "p95_e2e_latency",
            "throughput_tps",
            "avg_prefill_latency",
            "latency_cv",
        ]

    X = features[feature_cols].values

    # Standardise features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clamp n_clusters to number of samples
    k = min(n_clusters, len(X_scaled))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    features = features.copy()
    features["cluster"] = kmeans.fit_predict(X_scaled)

    print(f"\n  KMeans clustering (K={k}) on {len(features)} configurations")
    print(f"  Features: {feature_cols}")
    print(f"  Cluster distribution: {dict(features['cluster'].value_counts().sort_index())}")

    return features


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P"]


def plot_latency_vs_throughput(features: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot: avg latency vs throughput, colored by cluster."""
    fig, ax = plt.subplots(figsize=(11, 7))

    for cluster_id in sorted(features["cluster"].unique()):
        subset = features[features["cluster"] == cluster_id]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        marker = CLUSTER_MARKERS[cluster_id % len(CLUSTER_MARKERS)]

        ax.scatter(
            subset["avg_e2e_latency"],
            subset["throughput_tps"],
            c=color,
            marker=marker,
            s=160,
            edgecolors="black",
            linewidths=0.7,
            label=f"Cluster {cluster_id}",
            zorder=5,
        )

        # Annotate each point
        for _, row in subset.iterrows():
            label = f"{row['scheduler']}\n{row['eviction_policy']}"
            ax.annotate(
                label,
                (row["avg_e2e_latency"], row["throughput_tps"]),
                fontsize=7,
                ha="left",
                va="bottom",
                xytext=(6, 4),
                textcoords="offset points",
            )

    ax.set_xlabel("Average End-to-End Latency (s)", fontsize=13)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=13)
    ax.set_title(
        "Performance Regime Clusters — Latency vs. Throughput",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved {output_path}")


def plot_traffic_clusters(features: pd.DataFrame, output_path: Path) -> None:
    """
    Scatter plot: average prompt length vs latency coefficient of variation,
    colored by cluster.  Shows how traffic characteristics map to regimes.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    for cluster_id in sorted(features["cluster"].unique()):
        subset = features[features["cluster"] == cluster_id]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        marker = CLUSTER_MARKERS[cluster_id % len(CLUSTER_MARKERS)]

        ax.scatter(
            subset["avg_prompt_length"],
            subset["latency_cv"],
            c=color,
            marker=marker,
            s=160,
            edgecolors="black",
            linewidths=0.7,
            label=f"Cluster {cluster_id}",
            zorder=5,
        )

        for _, row in subset.iterrows():
            label = f"{row['scheduler']}\n{row['eviction_policy']}"
            ax.annotate(
                label,
                (row["avg_prompt_length"], row["latency_cv"]),
                fontsize=7,
                ha="left",
                va="bottom",
                xytext=(6, 4),
                textcoords="offset points",
            )

    ax.set_xlabel("Average Prompt Length (tokens)", fontsize=13)
    ax.set_ylabel("Latency Coefficient of Variation", fontsize=13)
    ax.set_title(
        "Traffic Clusters — Prompt Complexity vs. Latency Variability",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Sim Bench — Post-Simulation Clustering Analysis"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "results" / "sweep_results.csv"),
        help="Path to sweep_results.csv",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Number of KMeans clusters (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for plots",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  LLM Sim Bench — Clustering Analysis")
    print(f"{'='*60}")

    csv_path = Path(args.csv)
    df = load_sweep_results(csv_path)
    print(f"\n  Loaded {len(df)} per-request records from {csv_path}")

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build features and cluster
    features = build_config_features(df)
    features = cluster_configs(features, n_clusters=args.clusters)

    # Print cluster summary
    print("\n  Cluster Assignments:")
    print("  " + "-" * 56)
    for _, row in features.iterrows():
        print(
            f"    Cluster {row['cluster']}:  {row['scheduler']:30s}  "
            f"evict={row['eviction_policy']:18s}  "
            f"lat={row['avg_e2e_latency']:.4f}s  "
            f"tput={row['throughput_tps']:.1f} tok/s"
        )

    # Generate plots
    print("\n  Generating plots...")
    plot_latency_vs_throughput(features, output_dir / "clusters.png")
    plot_traffic_clusters(features, output_dir / "traffic_clusters.png")

    # Save clustered features
    features_path = output_dir / "cluster_features.csv"
    features.to_csv(features_path, index=False)
    print(f"  → Saved {features_path}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
