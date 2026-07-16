#!/usr/bin/env python3
"""Compare two package-generated temporal ligand-receptor runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _cluster_column(table: pd.DataFrame) -> str:
    for column in ("cluster", "cluster_id"):
        if column in table:
            return column
    raise KeyError("Pattern summary must contain 'cluster' or 'cluster_id'.")


def _load_run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tables = path / "tables"
    timecourse = pd.read_csv(tables / "lr_pair_timecourse.csv")
    patterns = pd.read_csv(tables / "lr_pattern_summary.csv")
    manifest_path = path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"pair", "time", "score"}
    if not required.issubset(timecourse.columns):
        raise KeyError(
            f"{tables / 'lr_pair_timecourse.csv'} is missing {sorted(required - set(timecourse.columns))}."
        )
    return timecourse, patterns, manifest


def compare_runs(reference_run: Path, candidate_run: Path, output_dir: Path) -> dict:
    reference, reference_patterns, reference_manifest = _load_run(reference_run)
    candidate, candidate_patterns, candidate_manifest = _load_run(candidate_run)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = reference[["pair", "time", "score"]].merge(
        candidate[["pair", "time", "score"]],
        on=["pair", "time"],
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="1:1",
    )
    if merged.empty:
        raise ValueError("The two runs have no shared pair/time rows.")
    merged["score_ratio_candidate_reference"] = np.divide(
        merged["score_candidate"],
        np.maximum(merged["score_reference"], 1e-12),
    )
    merged.to_csv(output_dir / "lr_pair_timecourse_comparison.csv", index=False)

    shape_rows = []
    for pair, subset in merged.groupby("pair", sort=True):
        subset = subset.sort_values("time")
        reference_values = subset["score_reference"].to_numpy(dtype=float)
        candidate_values = subset["score_candidate"].to_numpy(dtype=float)
        reference_range = max(float(np.ptp(reference_values)), 1e-12)
        candidate_range = max(float(np.ptp(candidate_values)), 1e-12)
        reference_normalized = (
            reference_values - reference_values.min()
        ) / reference_range
        candidate_normalized = (
            candidate_values - candidate_values.min()
        ) / candidate_range
        shape_rows.append(
            {
                "pair": str(pair),
                "pearson_shape": float(
                    np.corrcoef(reference_values, candidate_values)[0, 1]
                ),
                "minmax_rmse": float(
                    np.sqrt(
                        np.mean(
                            (reference_normalized - candidate_normalized) ** 2
                        )
                    )
                ),
            }
        )
    pair_shapes = pd.DataFrame(shape_rows)
    pair_shapes.to_csv(output_dir / "lr_pair_shape_comparison.csv", index=False)

    per_time_rows = []
    for time_value, subset in merged.groupby("time", sort=True):
        per_time_rows.append(
            {
                "time": float(time_value),
                "n_pairs": int(len(subset)),
                "pearson": float(
                    subset["score_reference"].corr(subset["score_candidate"])
                ),
                "spearman": float(
                    subset["score_reference"].corr(
                        subset["score_candidate"], method="spearman"
                    )
                ),
                "median_score_ratio": float(
                    np.median(subset["score_ratio_candidate_reference"])
                ),
            }
        )
    per_time = pd.DataFrame(per_time_rows)
    per_time.to_csv(output_dir / "lr_per_time_comparison.csv", index=False)

    reference_cluster = _cluster_column(reference_patterns)
    candidate_cluster = _cluster_column(candidate_patterns)
    assignments = reference_patterns[["pair", reference_cluster]].merge(
        candidate_patterns[["pair", candidate_cluster]],
        on="pair",
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="1:1",
    )
    assignments.columns = ["pair", "cluster_reference", "cluster_candidate"]
    assignments.to_csv(output_dir / "lr_cluster_assignments_comparison.csv", index=False)
    confusion = pd.crosstab(
        assignments["cluster_reference"], assignments["cluster_candidate"]
    )
    confusion.to_csv(output_dir / "lr_cluster_confusion.csv")

    same_reference_pairs = set(reference["pair"]) == set(candidate["pair"])
    score_pearson = float(
        merged["score_reference"].corr(merged["score_candidate"])
    )
    score_spearman = float(
        merged["score_reference"].corr(
            merged["score_candidate"], method="spearman"
        )
    )
    cluster_agreement = float(
        (assignments["cluster_reference"] == assignments["cluster_candidate"]).mean()
    )
    row_indices, column_indices = linear_sum_assignment(
        -confusion.to_numpy(dtype=float)
    )
    best_matched_cluster_agreement = float(
        confusion.to_numpy(dtype=float)[row_indices, column_indices].sum()
        / max(len(assignments), 1)
    )
    adjusted_rand = float(
        adjusted_rand_score(
            assignments["cluster_reference"], assignments["cluster_candidate"]
        )
    )
    normalized_mutual_information = float(
        normalized_mutual_info_score(
            assignments["cluster_reference"], assignments["cluster_candidate"]
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    scatter = axes[0].scatter(
        np.log1p(merged["score_reference"]),
        np.log1p(merged["score_candidate"]),
        c=merged["time"],
        cmap="viridis",
        s=14,
        alpha=0.72,
        linewidths=0,
    )
    lo = float(
        min(
            np.log1p(merged["score_reference"]).min(),
            np.log1p(merged["score_candidate"]).min(),
        )
    )
    hi = float(
        max(
            np.log1p(merged["score_reference"]).max(),
            np.log1p(merged["score_candidate"]).max(),
        )
    )
    axes[0].plot([lo, hi], [lo, hi], color="#4d4d4d", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Reference log1p LR score")
    axes[0].set_ylabel("Candidate log1p LR score")
    axes[0].set_title(f"Shared pair-time scores\nPearson={score_pearson:.3f}")
    fig.colorbar(scatter, ax=axes[0], label="Time")

    axes[1].plot(per_time["time"], per_time["pearson"], marker="o", label="Pearson")
    axes[1].plot(
        per_time["time"], per_time["spearman"], marker="s", label="Spearman"
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Pair-score agreement by time")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    image = axes[2].imshow(confusion.to_numpy(dtype=float), cmap="Blues")
    axes[2].set_xticks(np.arange(confusion.shape[1]), confusion.columns)
    axes[2].set_yticks(np.arange(confusion.shape[0]), confusion.index)
    axes[2].set_xlabel("Candidate cluster")
    axes[2].set_ylabel("Reference cluster")
    axes[2].set_title(
        "Pattern assignments\n"
        f"ARI={adjusted_rand:.3f}, best-match={best_matched_cluster_agreement:.1%}"
    )
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axes[2].text(
                column,
                row,
                str(int(confusion.iloc[row, column])),
                ha="center",
                va="center",
                color="#111111",
            )
    fig.colorbar(image, ax=axes[2], label="Pairs")
    fig.tight_layout()
    fig.savefig(output_dir / "lr_run_comparison.svg", bbox_inches="tight")
    fig.savefig(output_dir / "lr_run_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "reference_run": str(reference_run.resolve()),
        "candidate_run": str(candidate_run.resolve()),
        "reference_manifest_summary": reference_manifest.get("summary", {}),
        "candidate_manifest_summary": candidate_manifest.get("summary", {}),
        "same_pair_set": bool(same_reference_pairs),
        "n_shared_pairs": int(assignments.shape[0]),
        "n_shared_pair_time_rows": int(merged.shape[0]),
        "score_pearson": score_pearson,
        "score_spearman": score_spearman,
        "median_pair_shape_pearson": float(
            pair_shapes["pearson_shape"].median()
        ),
        "median_pair_shape_minmax_rmse": float(
            pair_shapes["minmax_rmse"].median()
        ),
        "median_score_ratio_candidate_reference": float(
            np.median(merged["score_ratio_candidate_reference"])
        ),
        "cluster_agreement": cluster_agreement,
        "best_matched_cluster_agreement": best_matched_cluster_agreement,
        "adjusted_rand_index": adjusted_rand,
        "normalized_mutual_information": normalized_mutual_information,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_runs(args.reference_run, args.candidate_run, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
