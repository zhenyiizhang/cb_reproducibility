#!/usr/bin/env python3
"""Recompute clean-counts ARISTA S15--S17 through public CytoBridge APIs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_helpers.arista_api import (  # noqa: E402
    AristaSpatiotemporalConfig,
    assert_package_only_runtime,
    run_arista_temporal_programs_api,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute prospective ARISTA temporal gene and ligand-receptor "
            "programs (S15-S17) from a current package model. The aligned "
            "H5AD is also the inverse-PCA reference and must retain "
            "varm['PCs']; legacy PCA loading/center sidecars are intentionally "
            "not accepted by this command."
        )
    )
    parser.add_argument("--aligned-h5ad", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lr-database", required=True)
    parser.add_argument("--gene-set-gmt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier-cache",
        default=None,
        help=(
            "Optional compatible classifier checkpoint file, or a classifier "
            "cache directory. Omit it to train and cache below OUTPUT_DIR."
        ),
    )
    return parser


def _classifier_cache_config(path: str | Path | None) -> dict[str, Path | None]:
    """Map the compact CLI option to the API's file/directory cache fields."""
    if path is None:
        return {"classifier_cache_path": None, "classifier_cache_dir": None}
    cache = Path(path).expanduser().resolve()
    if cache.exists() and cache.is_dir():
        return {"classifier_cache_path": None, "classifier_cache_dir": cache}
    return {"classifier_cache_path": cache, "classifier_cache_dir": None}


def run(args: argparse.Namespace) -> Any:
    aligned_h5ad = Path(args.aligned_h5ad).expanduser().resolve()
    config = AristaSpatiotemporalConfig(
        output_name="arista_clean_temporal_programs_api",
        output_dir=Path(args.output_dir).expanduser().resolve(),
        aligned_h5ad=aligned_h5ad,
        model_dir=Path(args.model_dir).expanduser().resolve(),
        model_format="current",
        time_points=(0.0, 1.0, 2.0, 3.0, 4.0),
        interp_time_points=(0.5, 1.5, 2.5, 3.5),
        n_samples=3072,
        skip_nonsplit_sde=True,
        split_sde_dt=0.01,
        split_sigma=0.03,
        use_real_for_observed=True,
        spatial_warp_to_observed_piecewise=False,
        classifier_knn_neighbors=10,
        random_seed=int(args.seed),
        device=args.device,
        run_communication=True,
        run_3d=False,
        **_classifier_cache_config(args.classifier_cache),
    )
    result = run_arista_temporal_programs_api(
        config,
        lr_database=Path(args.lr_database).expanduser().resolve(),
        reference_h5ad=aligned_h5ad,
        n_top_genes=250,
        n_gene_cluster_genes=2000,
        n_gene_clusters=2,
        n_lr_clusters=2,
        # This clean prospective run must use reference_h5ad.varm['PCs'].
        pca_components_csv=None,
        pca_center_csv=None,
        # Preserve the manuscript paper-parity symbol mapping rather than
        # silently preferring a human-tagged duplicate in the LR database.
        preferred_species_tag=None,
        gene_profile_normalization="zscore",
        gene_profile_linkage_method="average",
        gene_profile_cluster_order="raw",
        lr_profile_linkage_method="ward",
        lr_profile_cluster_order="dendrogram",
        communication_max_cells_per_timepoint=3072,
        communication_random_seed=int(args.seed),
        communication_rng_warmup_max_cells_per_timepoint=2500,
        gene_set_gmt=Path(args.gene_set_gmt).expanduser().resolve(),
        gene_set_background="expression",
    )
    assert_package_only_runtime()
    return result


def main() -> None:
    result = run(_parser().parse_args())
    print(result.manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
