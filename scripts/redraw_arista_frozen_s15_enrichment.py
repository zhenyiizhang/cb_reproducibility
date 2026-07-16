#!/usr/bin/env python3
"""Redraw the frozen ARISTA S15 GO panels through public CytoBridge plots."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.pl import plot_enrichment_bar, plot_enrichment_dot  # noqa: E402

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


def _ratio_parts(series: pd.Series, name: str) -> tuple[pd.Series, pd.Series]:
    parts = series.astype(str).str.split("/", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError(f"{name} must contain numerator/denominator ratios.")
    return (
        pd.to_numeric(parts.iloc[:, 0], errors="raise"),
        pd.to_numeric(parts.iloc[:, 1], errors="raise"),
    )


def _standardize_clusterprofiler_go(
    path: Path,
    *,
    alpha: float,
    ontology: str,
) -> pd.DataFrame:
    source = pd.read_csv(path, sep="\t")
    required = {
        "ID",
        "Description",
        "GeneRatio",
        "BgRatio",
        "pvalue",
        "p.adjust",
        "geneID",
        "Count",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise KeyError(f"{path} is missing clusterProfiler columns: {missing}")
    if ontology != "ALL":
        if "ONTOLOGY" not in source.columns:
            raise KeyError(f"{path} is missing the ONTOLOGY column.")
        source = source.loc[source["ONTOLOGY"].astype(str) == ontology].copy()
        if source.empty:
            raise ValueError(f"{path} contains no {ontology} ontology rows.")
    query_overlap, query_size = _ratio_parts(source["GeneRatio"], "GeneRatio")
    set_size, background_size = _ratio_parts(source["BgRatio"], "BgRatio")
    overlap_count = pd.to_numeric(source["Count"], errors="raise")
    if not np.array_equal(query_overlap.to_numpy(), overlap_count.to_numpy()):
        raise ValueError(f"GeneRatio numerators and Count disagree in {path}.")
    expected = query_size * set_size / background_size
    a = overlap_count.astype(float)
    b = query_size.astype(float) - a
    c = set_size.astype(float) - a
    d = background_size.astype(float) - query_size.astype(float) - c
    denominator = b * c
    odds_ratio = np.divide(
        a * d,
        denominator,
        out=np.full(len(source), np.inf, dtype=float),
        where=denominator.to_numpy(dtype=float) > 0.0,
    )
    adjusted = pd.to_numeric(source["p.adjust"], errors="raise")
    standardized = pd.DataFrame(
        {
            "term": source["Description"].astype(str)
            + " ("
            + source["ID"].astype(str)
            + ")",
            "term_name": source["Description"].astype(str),
            "term_id": source["ID"].astype(str),
            "description": source["Description"].astype(str),
            "query_size": query_size.astype(int),
            "query_input_size": query_size.astype(int),
            "background_size": background_size.astype(int),
            "set_size": set_size.astype(int),
            "overlap_count": overlap_count.astype(int),
            "expected_overlap": expected.astype(float),
            "gene_ratio": (overlap_count / query_size).astype(float),
            "background_ratio": (set_size / background_size).astype(float),
            "fold_enrichment": (overlap_count / expected).astype(float),
            "odds_ratio": odds_ratio,
            "p_value": pd.to_numeric(source["pvalue"], errors="raise"),
            "adjusted_p_value": adjusted,
            "significant": adjusted <= float(alpha),
            "overlap_genes": source["geneID"].astype(str).str.replace(
                "/", ";", regex=False
            ),
        }
    )
    return standardized.sort_values(
        ["adjusted_p_value", "p_value", "fold_enrichment"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern-one-go", type=Path, required=True)
    parser.add_argument("--pattern-two-go", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--ontology", choices=["BP", "CC", "MF", "ALL"], default="BP"
    )
    parser.add_argument("--pattern-one-top-terms", type=int, default=25)
    parser.add_argument("--pattern-two-top-terms", type=int, default=20)
    args = parser.parse_args()

    one_path = _file(args.pattern_one_go, "frozen Pattern 1 GO table")
    two_path = _file(args.pattern_two_go, "frozen Pattern 2 GO table")
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        1: _standardize_clusterprofiler_go(
            one_path, alpha=float(args.alpha), ontology=args.ontology
        ),
        2: _standardize_clusterprofiler_go(
            two_path, alpha=float(args.alpha), ontology=args.ontology
        ),
    }
    standardized_paths: dict[int, Path] = {}
    for pattern, table in tables.items():
        standardized_path = tables_dir / f"pattern_{pattern}_go_standardized.csv"
        table.to_csv(standardized_path, index=False)
        standardized_paths[pattern] = standardized_path

    pattern_one_plot = tables[1].loc[tables[1]["significant"]].copy()
    pattern_two_plot = tables[2].loc[tables[2]["significant"]].copy()
    if pattern_one_plot.empty or pattern_two_plot.empty:
        raise ValueError("The frozen GO tables contain no significant terms to plot.")
    figure_paths = {
        "pattern_one_bar": plot_enrichment_bar(
            pattern_one_plot,
            out_path=figures_dir / "pattern_1_go_bar.svg",
            top_n=int(args.pattern_one_top_terms),
            title="ARISTA Pattern 1 GO biological processes",
        ),
        "pattern_two_dot": plot_enrichment_dot(
            pattern_two_plot,
            out_path=figures_dir / "pattern_2_go_dot.svg",
            top_n=int(args.pattern_two_top_terms),
            title="ARISTA Pattern 2 GO biological processes",
        ),
    }

    assert_package_only_runtime()
    import CytoBridge

    manifest = {
        "workflow": "arista_frozen_s15_enrichment_redraw",
        "runtime_contract": "installed CytoBridge public plotting APIs",
        "cytobridge_module": str(Path(CytoBridge.__file__).resolve()),
        "inputs": {
            "pattern_one_go": str(one_path),
            "pattern_one_go_sha256": _sha256(one_path),
            "pattern_two_go": str(two_path),
            "pattern_two_go_sha256": _sha256(two_path),
        },
        "settings": {
            "alpha": float(args.alpha),
            "ontology": args.ontology,
            "pattern_one_top_terms": int(args.pattern_one_top_terms),
            "pattern_two_top_terms": int(args.pattern_two_top_terms),
            "plots_use_significant_terms_only": True,
        },
        "summary": {
            "pattern_one_terms": int(len(tables[1])),
            "pattern_one_significant_terms": int(tables[1]["significant"].sum()),
            "pattern_two_terms": int(len(tables[2])),
            "pattern_two_significant_terms": int(tables[2]["significant"].sum()),
            "background_sizes": sorted(
                set(tables[1]["background_size"].astype(int))
                | set(tables[2]["background_size"].astype(int))
            ),
        },
        "outputs": {
            "standardized_tables": {
                str(pattern): str(path)
                for pattern, path in standardized_paths.items()
            },
            "figures": {name: str(path) for name, path in figure_paths.items()},
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
