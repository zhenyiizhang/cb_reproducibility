#!/usr/bin/env python3
"""Evaluate/reuse the published ARISTA checkpoint and compare retrained runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-metrics", required=True)
    parser.add_argument(
        "--current-name",
        default="retrained_auto_thresholds",
        help="Stable label used for --current-metrics in tables and figures.",
    )
    parser.add_argument("--current-aligned-h5ad", required=True)
    parser.add_argument("--current-threshold-meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--max-ot-points", type=int, default=1024)
    parser.add_argument("--structure-max-points", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--legacy-data-csv", default=None)
    parser.add_argument("--legacy-model-dir", default=None)
    parser.add_argument("--legacy-edge-predictor-root", default=None)
    parser.add_argument(
        "--legacy-metrics",
        default=None,
        help="Reuse an existing published-model metrics CSV instead of simulating it again.",
    )
    parser.add_argument(
        "--additional-metrics",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Add another identically configured metric table to the comparison.",
    )
    return parser


def _require_file(value: str, description: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_named_metrics(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --additional-metrics '{value}'; expected NAME=CSV."
            )
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in paths:
            raise ValueError(f"Metric table name must be non-empty and unique: '{name}'.")
        paths[name] = _require_file(raw_path, f"additional metrics '{name}'")
    return paths


def _nearest_neighbor_summary(
    coordinates: np.ndarray,
    *,
    random_seed: int,
    max_points: int = 12000,
) -> dict[str, float | int]:
    from sklearn.neighbors import NearestNeighbors

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape[0] > max_points:
        rng = np.random.default_rng(int(random_seed))
        indices = rng.choice(coordinates.shape[0], size=max_points, replace=False)
        coordinates = coordinates[indices]
    distances, _ = NearestNeighbors(n_neighbors=2).fit(coordinates).kneighbors()
    nearest = distances[:, 1]
    return {
        "n_sampled": int(coordinates.shape[0]),
        "median": float(np.median(nearest)),
        "q25": float(np.quantile(nearest, 0.25)),
        "q75": float(np.quantile(nearest, 0.75)),
        "mean": float(nearest.mean()),
    }


def _coordinate_scale_summary(
    old_coordinates: np.ndarray,
    current_coordinates: np.ndarray,
    *,
    old_cutoff: float,
    current_cutoff: float,
    random_seed: int,
) -> dict[str, object]:
    old_coordinates = np.asarray(old_coordinates, dtype=np.float64)
    current_coordinates = np.asarray(current_coordinates, dtype=np.float64)
    old_nn = _nearest_neighbor_summary(old_coordinates, random_seed=random_seed)
    current_nn = _nearest_neighbor_summary(
        current_coordinates, random_seed=random_seed
    )
    summary: dict[str, object] = {
        "old": {
            "min": old_coordinates.min(axis=0).tolist(),
            "max": old_coordinates.max(axis=0).tolist(),
            "std": old_coordinates.std(axis=0).tolist(),
            "nearest_neighbor": old_nn,
            "spatial_cutoff": float(old_cutoff),
            "cutoff_over_median_nn": float(old_cutoff / old_nn["median"]),
        },
        "current": {
            "min": current_coordinates.min(axis=0).tolist(),
            "max": current_coordinates.max(axis=0).tolist(),
            "std": current_coordinates.std(axis=0).tolist(),
            "nearest_neighbor": current_nn,
            "spatial_cutoff": float(current_cutoff),
            "cutoff_over_median_nn": float(
                current_cutoff / current_nn["median"]
            ),
        },
        "same_shape": bool(old_coordinates.shape == current_coordinates.shape),
    }
    if old_coordinates.shape == current_coordinates.shape:
        delta = current_coordinates - old_coordinates
        summary["rowwise"] = {
            "mean_absolute_difference": float(np.abs(delta).mean()),
            "max_absolute_difference": float(np.abs(delta).max()),
            "pearson_by_axis": [
                float(np.corrcoef(old_coordinates[:, axis], current_coordinates[:, axis])[0, 1])
                for axis in range(old_coordinates.shape[1])
            ],
        }
    return summary


def main() -> int:
    args = _parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_metrics_path = _require_file(args.current_metrics, "current metrics")
    current_h5ad_path = _require_file(
        args.current_aligned_h5ad, "current aligned H5AD"
    )
    threshold_meta_path = _require_file(
        args.current_threshold_meta, "current threshold metadata"
    )
    additional_metric_paths = _parse_named_metrics(args.additional_metrics)

    import anndata as ad
    import CytoBridge as cb

    from downstream_helpers.arista_api import (
        AristaSpatiotemporalConfig,
        assert_package_only_runtime,
        load_arista_spatiotemporal_context,
    )

    legacy_config = AristaSpatiotemporalConfig(
        data_csv=args.legacy_data_csv,
        model_dir=args.legacy_model_dir,
        edge_predictor_root=args.legacy_edge_predictor_root,
        model_format="legacy",
        device=args.device,
        run_communication=False,
        run_3d=False,
    )
    legacy = load_arista_spatiotemporal_context(legacy_config)
    if args.legacy_metrics is None:
        legacy_evaluation = cb.tl.evaluate_model_distributions(
            legacy["adata"],
            legacy["loaded"].model,
            n_samples=int(args.n_samples),
            dt=float(args.dt),
            sigma=float(args.sigma),
            include_score=True,
            interaction_m=int(args.interaction_m),
            max_ot_points=int(args.max_ot_points),
            structure_max_points=int(args.structure_max_points),
            device=str(args.device),
            time_key=str(legacy["resolved_time_key"]),
            obsm_key=legacy_config.obsm_key,
            spatial_key=legacy_config.spatial_key,
            concat_spatial=True,
            random_seed=int(args.random_seed),
        )
        legacy_metrics = legacy_evaluation.metrics
        legacy_paths = cb.tl.save_distribution_evaluation(
            legacy_evaluation,
            output_dir / "published_saved_model",
        )
        legacy_metrics_path = Path(legacy_paths["metrics"])
    else:
        legacy_metrics_path = _require_file(
            args.legacy_metrics, "published saved-model metrics"
        )
        legacy_metrics = pd.read_csv(legacy_metrics_path)
        legacy_paths = {"metrics": str(legacy_metrics_path)}

    current_metrics = pd.read_csv(current_metrics_path)
    current_name = str(args.current_name).strip()
    if not current_name or current_name == "published_saved_model":
        raise ValueError(
            "--current-name must be non-empty and different from "
            "'published_saved_model'."
        )
    metrics_by_model = {
        "published_saved_model": legacy_metrics,
        current_name: current_metrics,
    }
    for name, path in additional_metric_paths.items():
        if name in metrics_by_model:
            raise ValueError(f"Duplicate model metric name: '{name}'.")
        metrics_by_model[name] = pd.read_csv(path)
    comparison = cb.tl.compare_distribution_metric_tables(
        metrics_by_model,
        baseline="published_saved_model",
    )
    comparison_paths = cb.tl.save_distribution_metric_comparison(
        comparison,
        output_dir / "comparison",
    )

    current_adata = ad.read_h5ad(current_h5ad_path)
    threshold_meta = json.loads(threshold_meta_path.read_text(encoding="utf-8"))
    current_cutoff_raw = threshold_meta.get(
        "distance_threshold",
        threshold_meta.get("interaction_cutoff"),
    )
    if current_cutoff_raw is None:
        raise KeyError(
            "Current threshold metadata must contain 'distance_threshold' "
            "or the legacy-compatible 'interaction_cutoff'."
        )
    current_cutoff = float(current_cutoff_raw)
    coordinate_summary = _coordinate_scale_summary(
        legacy["adata"].obsm[legacy_config.spatial_key],
        current_adata.obsm[legacy_config.spatial_key],
        old_cutoff=0.05,
        current_cutoff=current_cutoff,
        random_seed=int(args.random_seed),
    )
    coordinate_summary["edge_predictor_thresholds"] = {
        "published_saved_model": 0.45,
        current_name: float(
            threshold_meta["edge_predictor_threshold_selected"]
        ),
        f"{current_name}_selection_source": str(threshold_meta["selection_source"]),
    }
    coordinate_path = output_dir / "coordinate_threshold_comparison.json"
    coordinate_path.write_text(
        json.dumps(coordinate_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    assert_package_only_runtime()
    manifest = {
        "workflow": "arista_published_vs_retrained_distribution_comparison",
        "settings": {
            "n_samples": int(args.n_samples),
            "dt": float(args.dt),
            "sigma": float(args.sigma),
            "interaction_m": int(args.interaction_m),
            "max_ot_points": int(args.max_ot_points),
            "structure_max_points": int(args.structure_max_points),
            "random_seed": int(args.random_seed),
            "device": str(args.device),
            "include_score": True,
        },
        "inputs": {
            "current_metrics": str(current_metrics_path),
            "current_name": current_name,
            "current_metrics_sha256": _sha256(current_metrics_path),
            "current_aligned_h5ad": str(current_h5ad_path),
            "current_aligned_h5ad_sha256": _sha256(current_h5ad_path),
            "current_threshold_meta": str(threshold_meta_path),
            "legacy_metrics": str(legacy_metrics_path),
            "legacy_metrics_sha256": _sha256(legacy_metrics_path),
            "legacy_data_csv": str(legacy["data_csv"]),
            "legacy_model_dir": str(legacy["model_dir"]),
            "legacy_weight_checkpoint": str(legacy["loaded"].weight_path),
            "legacy_weight_checkpoint_sha256": _sha256(
                Path(legacy["loaded"].weight_path)
            ),
            "legacy_score_checkpoint": str(legacy["loaded"].score_path),
            "legacy_score_checkpoint_sha256": _sha256(
                Path(legacy["loaded"].score_path)
            ),
            "legacy_edge_predictor_root": str(legacy["edge_predictor_root"]),
            "additional_metrics": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in additional_metric_paths.items()
            },
        },
        "outputs": {
            "legacy_evaluation": legacy_paths,
            "comparison": comparison_paths,
            "coordinate_threshold_comparison": str(coordinate_path),
        },
        "cytobridge_module": str(Path(cb.__file__).resolve()),
        "runtime_contract": "installed CytoBridge public API; no vendored model/solver imports",
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
