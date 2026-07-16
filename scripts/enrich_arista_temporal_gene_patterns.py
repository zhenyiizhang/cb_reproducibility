#!/usr/bin/env python3
"""Recluster all ARISTA temporal genes and run package-backed offline ORA."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.pl import (  # noqa: E402
    plot_enrichment_bar,
    plot_enrichment_dot,
    plot_temporal_pattern_prototypes,
)
from CytoBridge.tl import (  # noqa: E402
    cluster_temporal_profiles,
    load_gmt_gene_sets,
    overrepresentation_analysis,
)

from downstream_helpers.arista_api import assert_package_only_runtime  # noqa: E402


def _file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_comparison(
    assignments: pd.DataFrame,
    pattern_one_path: Path | None,
    pattern_two_path: Path | None,
) -> dict[str, object] | None:
    if pattern_one_path is None and pattern_two_path is None:
        return None
    if pattern_one_path is None or pattern_two_path is None:
        raise ValueError(
            "reference-pattern-one and reference-pattern-two must be supplied together."
        )
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    one_path = _file(pattern_one_path, "reference Pattern 1 genes")
    two_path = _file(pattern_two_path, "reference Pattern 2 genes")
    one = set(pd.read_csv(one_path, sep="\t").iloc[:, 0].astype(str))
    two = set(pd.read_csv(two_path, sep="\t").iloc[:, 0].astype(str))
    reference = one | two
    shared = assignments[assignments["profile"].astype(str).isin(reference)].copy()
    if shared.empty:
        raise ValueError("No recomputed gene profiles overlap the reference patterns.")
    truth = np.asarray(
        [1 if profile in one else 2 for profile in shared["profile"].astype(str)],
        dtype=int,
    )
    candidate = shared["cluster"].to_numpy(dtype=int)
    raw_agreement = float(np.mean(candidate == truth))
    best_agreement = max(raw_agreement, 1.0 - raw_agreement)
    return {
        "reference_pattern_one": str(one_path),
        "reference_pattern_one_sha256": _sha256(one_path),
        "reference_pattern_two": str(two_path),
        "reference_pattern_two_sha256": _sha256(two_path),
        "n_reference_genes": int(len(reference)),
        "n_shared_genes": int(len(shared)),
        "best_matched_agreement": best_agreement,
        "adjusted_rand_index": float(adjusted_rand_score(truth, candidate)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-run", type=Path, required=True)
    parser.add_argument("--gene-set-gmt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument(
        "--normalization", choices=["zscore", "minmax", "none"], default="zscore"
    )
    parser.add_argument(
        "--linkage-method",
        choices=["single", "complete", "average", "weighted", "centroid", "median", "ward"],
        default="average",
    )
    parser.add_argument(
        "--cluster-order", choices=["peak_time", "dendrogram", "raw"], default="peak_time"
    )
    parser.add_argument("--min-set-size", type=int, default=5)
    parser.add_argument("--max-set-size", type=int, default=5000)
    parser.add_argument("--min-overlap", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-terms", type=int, default=20)
    parser.add_argument(
        "--background", choices=["expression", "library"], default="expression"
    )
    parser.add_argument("--reference-pattern-one", type=Path)
    parser.add_argument("--reference-pattern-two", type=Path)
    args = parser.parse_args()

    run_dir = args.temporal_run.expanduser().resolve()
    expression_path = _file(
        run_dir / "tables/gene_expression_by_time.csv",
        "temporal gene-expression table",
    )
    name_map_path = _file(
        run_dir / "tables/gene_name_map.csv",
        "temporal gene-name map",
    )
    gmt_path = _file(args.gene_set_gmt, "GMT gene-set library")
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    expression = pd.read_csv(expression_path, index_col=0)
    expression.index = expression.index.astype(str)
    expression.columns = [float(value) for value in expression.columns]
    name_map = pd.read_csv(name_map_path)
    clustering = cluster_temporal_profiles(
        expression,
        n_clusters=int(args.n_clusters),
        normalization=args.normalization,
        method=args.linkage_method,
        cluster_order=args.cluster_order,
    )
    assignments = clustering.assignments.copy()
    symbol_map = name_map.set_index("gene")["gene_symbol"]
    assignments["gene_symbol"] = assignments["profile"].map(symbol_map)

    assignments_path = tables_dir / "gene_pattern_assignments.csv"
    normalized_path = tables_dir / "gene_normalized_profiles.csv"
    prototypes_path = tables_dir / "gene_pattern_prototypes.csv"
    diagnostics_path = tables_dir / "gene_pattern_diagnostics.csv"
    assignments.to_csv(assignments_path, index=False)
    clustering.normalized_profiles.to_csv(normalized_path)
    clustering.prototypes.to_csv(prototypes_path, index=False)
    clustering.diagnostics.to_csv(diagnostics_path, index=False)

    library = load_gmt_gene_sets(gmt_path)
    background = name_map["gene_symbol"].dropna().astype(str).tolist()
    enrichment_tables = []
    significant_counts: dict[int, int] = {}
    figure_paths = []
    for cluster, subset in assignments.groupby("cluster", sort=True):
        enrichment = overrepresentation_analysis(
            subset["gene_symbol"].dropna().astype(str).tolist(),
            library,
            background_genes=background if args.background == "expression" else None,
            min_set_size=int(args.min_set_size),
            max_set_size=int(args.max_set_size),
            min_overlap=int(args.min_overlap),
            alpha=float(args.alpha),
        )
        enrichment.insert(0, "cluster", int(cluster))
        enrichment_tables.append(enrichment)
        significant_counts[int(cluster)] = int(enrichment["significant"].sum())
        if not enrichment.empty:
            significant_enrichment = enrichment.loc[enrichment["significant"]].copy()
            plot_enrichment = (
                significant_enrichment
                if not significant_enrichment.empty
                else enrichment
            )
            figure_paths.extend(
                [
                    plot_enrichment_bar(
                        plot_enrichment,
                        out_path=figures_dir
                        / f"gene_pattern_{int(cluster)}_enrichment_bar.svg",
                        top_n=int(args.top_terms),
                        title=f"ARISTA Pattern {int(cluster)} gene-set enrichment",
                    ),
                    plot_enrichment_dot(
                        plot_enrichment,
                        out_path=figures_dir
                        / f"gene_pattern_{int(cluster)}_enrichment_dot.svg",
                        top_n=int(args.top_terms),
                        title=f"ARISTA Pattern {int(cluster)} gene-set enrichment",
                    ),
                ]
            )
    enrichment_table = pd.concat(enrichment_tables, ignore_index=True)
    enrichment_path = tables_dir / "gene_pattern_enrichment.csv"
    enrichment_table.to_csv(enrichment_path, index=False)
    prototype_figure = plot_temporal_pattern_prototypes(
        clustering.prototypes,
        out_path=figures_dir / "gene_pattern_prototypes.svg",
        title="ARISTA temporal gene-pattern prototypes",
        y_label=(
            "Mean gene-wise z-score"
            if args.normalization == "zscore"
            else "Mean normalized expression"
        ),
    )

    assert_package_only_runtime()
    import CytoBridge

    manifest = {
        "workflow": "arista_temporal_gene_enrichment_postprocess",
        "runtime_contract": "installed CytoBridge public APIs",
        "cytobridge_module": str(Path(CytoBridge.__file__).resolve()),
        "inputs": {
            "temporal_run": str(run_dir),
            "gene_expression": str(expression_path),
            "gene_expression_sha256": _sha256(expression_path),
            "gene_name_map": str(name_map_path),
            "gene_name_map_sha256": _sha256(name_map_path),
            "gene_set_gmt": str(gmt_path),
            "gene_set_gmt_sha256": _sha256(gmt_path),
        },
        "settings": {
            "n_clusters": int(args.n_clusters),
            "normalization": args.normalization,
            "linkage_method": args.linkage_method,
            "cluster_order": args.cluster_order,
            "min_set_size": int(args.min_set_size),
            "max_set_size": int(args.max_set_size),
            "min_overlap": int(args.min_overlap),
            "alpha": float(args.alpha),
            "top_terms": int(args.top_terms),
            "plots_use_significant_terms_only_when_available": True,
            "background_contract": (
                "all reconstructed gene symbols intersected with GMT"
                if args.background == "expression"
                else "all genes in the supplied GMT library"
            ),
        },
        "summary": {
            "n_expression_genes": int(expression.shape[0]),
            "n_timepoints": int(expression.shape[1]),
            "cluster_counts": assignments["cluster"].value_counts().sort_index().to_dict(),
            "significant_term_counts": significant_counts,
            "tested_background_sizes": sorted(
                enrichment_table["background_size"].dropna().astype(int).unique().tolist()
            ),
            "tested_query_sizes": {
                str(cluster): int(subset["query_size"].iloc[0])
                for cluster, subset in enrichment_table.groupby("cluster", sort=True)
                if not subset.empty
            },
            "reference_comparison": _reference_comparison(
                assignments,
                args.reference_pattern_one,
                args.reference_pattern_two,
            ),
        },
        "outputs": {
            "assignments": str(assignments_path),
            "normalized_profiles": str(normalized_path),
            "prototypes": str(prototypes_path),
            "diagnostics": str(diagnostics_path),
            "enrichment": str(enrichment_path),
            "prototype_figure": str(prototype_figure),
            "enrichment_figures": [str(path) for path in figure_paths],
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
