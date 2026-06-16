#!/usr/bin/env python3
"""Create report-friendly summary plots for the GPT-2 pilot experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUBTYPE_ORDER = ["local", "early_recall", "middle_recall", "comparison"]
MASS_COLORS = {
    "sink_mass_mean": "#4C78A8",
    "recent_mass_mean": "#59A14F",
    "cold_mass_mean": "#E15759",
}
SUBTYPE_COLORS = {
    "local": "#4C78A8",
    "early_recall": "#F28E2B",
    "middle_recall": "#59A14F",
    "comparison": "#B07AA1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create subtype-level summary CSVs and PNGs for pilot outputs."
    )
    parser.add_argument(
        "--samples",
        default="data/samples_pilot_40.jsonl",
        help="Pilot JSONL containing sample id/type/subtype metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/kv_importance_gpt2_pilot_40",
        help="Directory containing pilot CSV outputs and receiving summary files.",
    )
    return parser.parse_args()


def read_sample_metadata(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample_id = str(sample["id"])
            request_type = str(sample.get("type", "unknown"))
            subtype = str(sample.get("subtype") or request_type)
            rows.append(
                {
                    "sample_id": sample_id,
                    "type": request_type,
                    "subtype": subtype,
                }
            )
    if not rows:
        raise ValueError(f"No sample metadata found in {path}")
    return pd.DataFrame(rows)


def ordered_subtypes(values: pd.Series) -> List[str]:
    seen = set(str(value) for value in values.dropna().unique())
    ordered = [subtype for subtype in SUBTYPE_ORDER if subtype in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def merge_metadata(df: pd.DataFrame, metadata: pd.DataFrame, name: str) -> pd.DataFrame:
    merged = df.merge(metadata, on="sample_id", how="left")
    if merged["subtype"].isna().any():
        missing = sorted(merged.loc[merged["subtype"].isna(), "sample_id"].unique())
        raise ValueError(f"{name} contains sample ids missing from metadata: {missing}")
    return merged


def save_hot_set_overlap_mean(stability: pd.DataFrame, output_path: Path) -> None:
    mean_df = (
        stability.dropna(subset=["hot_set_overlap"])
        .groupby("step", as_index=False)["hot_set_overlap"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        mean_df["step"],
        mean_df["hot_set_overlap"],
        marker="o",
        linewidth=2.2,
        color="#222222",
        label="Overall mean",
    )
    ax.set_title("Mean Hot Set Overlap Across Pilot Samples")
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Hot set overlap")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_hot_set_overlap_by_subtype(stability: pd.DataFrame, output_path: Path) -> None:
    subtypes = ordered_subtypes(stability["subtype"])
    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    for subtype in subtypes:
        group = (
            stability[
                (stability["subtype"] == subtype)
                & stability["hot_set_overlap"].notna()
            ]
            .groupby("step", as_index=False)["hot_set_overlap"]
            .mean()
        )
        if group.empty:
            continue
        ax.plot(
            group["step"],
            group["hot_set_overlap"],
            marker="o",
            linewidth=2.0,
            label=subtype,
            color=SUBTYPE_COLORS.get(subtype),
        )

    ax.set_title("Mean Hot Set Overlap by Request Subtype")
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Hot set overlap")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_attention_mass_by_subtype(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.set_index("subtype").loc[
        [subtype for subtype in SUBTYPE_ORDER if subtype in set(summary["subtype"])]
    ]
    remaining = [
        subtype
        for subtype in summary["subtype"]
        if subtype not in set(plot_df.index)
    ]
    if remaining:
        plot_df = pd.concat([plot_df, summary.set_index("subtype").loc[remaining]])

    x = np.arange(len(plot_df.index))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = [
        ("sink_mass_mean", "Sink KV", -width),
        ("recent_mass_mean", "Recent KV", 0.0),
        ("cold_mass_mean", "Cold KV", width),
    ]
    for column, label, offset in bars:
        ax.bar(
            x + offset,
            plot_df[column],
            width=width,
            label=label,
            color=MASS_COLORS[column],
        )

    ax.set_title("Mean Attention Mass by Request Subtype")
    ax.set_xlabel("Request subtype")
    ax.set_ylabel("Mean attention mass")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples)
    output_dir = Path(args.output_dir)

    metadata = read_sample_metadata(samples_path)
    attention = merge_metadata(
        pd.read_csv(output_dir / "attention_mass.csv"), metadata, "attention_mass.csv"
    )
    stability = merge_metadata(
        pd.read_csv(output_dir / "stability_metrics.csv"),
        metadata,
        "stability_metrics.csv",
    )
    chunk_importance = merge_metadata(
        pd.read_csv(output_dir / "chunk_importance.csv"),
        metadata,
        "chunk_importance.csv",
    )
    topk = merge_metadata(
        pd.read_csv(output_dir / "topk_chunks.csv"), metadata, "topk_chunks.csv"
    )

    subtype_order = ordered_subtypes(metadata["subtype"])

    attention_summary = (
        attention.groupby("subtype", as_index=False)
        .agg(
            sink_mass_mean=("sink_mass", "mean"),
            recent_mass_mean=("recent_mass", "mean"),
            cold_mass_mean=("cold_mass", "mean"),
            cold_mass_min=("cold_mass", "min"),
            cold_mass_max=("cold_mass", "max"),
        )
        .set_index("subtype")
        .loc[subtype_order]
        .reset_index()
    )
    attention_summary.to_csv(
        output_dir / "attention_mass_summary_by_subtype.csv", index=False
    )

    stability_summary = (
        stability.groupby("subtype", as_index=False)
        .agg(
            hot_set_overlap_mean=("hot_set_overlap", "mean"),
            hot_set_overlap_min=("hot_set_overlap", "min"),
            hot_set_overlap_max=("hot_set_overlap", "max"),
            churn_rate_mean=("churn_rate", "mean"),
            churn_rate_max=("churn_rate", "max"),
        )
        .set_index("subtype")
        .loc[subtype_order]
        .reset_index()
    )
    stability_summary.to_csv(
        output_dir / "stability_summary_by_subtype.csv", index=False
    )

    topk_frequency = (
        topk.groupby(["subtype", "chunk_id"], as_index=False)
        .agg(
            topk_count=("chunk_id", "size"),
            mean_importance=("importance", "mean"),
        )
        .sort_values(["subtype", "topk_count", "chunk_id"], ascending=[True, False, True])
    )
    subtype_totals = topk.groupby("subtype").size().rename("subtype_topk_rows")
    topk_frequency = topk_frequency.merge(subtype_totals, on="subtype", how="left")
    topk_frequency["frequency"] = (
        topk_frequency["topk_count"] / topk_frequency["subtype_topk_rows"]
    )
    topk_frequency = (
        topk_frequency.set_index("subtype")
        .loc[subtype_order]
        .reset_index()
        [
            [
                "subtype",
                "chunk_id",
                "topk_count",
                "subtype_topk_rows",
                "frequency",
                "mean_importance",
            ]
        ]
    )
    topk_frequency.to_csv(
        output_dir / "topk_chunk_frequency_by_subtype.csv", index=False
    )

    # Read and summarize chunk_importance as an input sanity check without writing
    # another CSV that could be confused with the original analysis outputs.
    chunk_counts = chunk_importance.groupby(["sample_id", "step"])["chunk_id"].nunique()

    save_hot_set_overlap_mean(
        stability, output_dir / "hot_set_overlap_mean.png"
    )
    save_hot_set_overlap_by_subtype(
        stability, output_dir / "hot_set_overlap_by_subtype.png"
    )
    save_attention_mass_by_subtype(
        attention_summary, output_dir / "attention_mass_by_subtype.png"
    )

    print(f"Wrote summary outputs to {output_dir}")
    print(f"Subtypes: {', '.join(subtype_order)}")
    print(
        "Average chunks per sample/step: "
        f"{chunk_counts.mean():.3f} "
        f"(min={chunk_counts.min()}, max={chunk_counts.max()})"
    )


if __name__ == "__main__":
    main()
