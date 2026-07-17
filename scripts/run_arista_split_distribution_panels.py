#!/usr/bin/env python3
"""Generate ARISTA split-SDE generated-versus-observed PCA/spatial panels."""

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
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--model-format", choices=("legacy", "current"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-csv", default=None)
    parser.add_argument("--aligned-h5ad", default=None)
    parser.add_argument("--edge-predictor-root", default=None)
    parser.add_argument("--label-color-json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-points", default="0,1,2,3,4")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--growth-alpha", type=float, default=1.0)
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_file_manifest(path: object) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return None
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _time_points(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("--time-points must contain at least one number.")
    if tuple(sorted(set(values))) != values:
        raise ValueError("--time-points must be sorted and unique.")
    return values


def main() -> int:
    args = _parser().parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive.")
    if args.dt <= 0:
        raise ValueError("--dt must be positive.")

    import CytoBridge as cb

    from downstream_helpers.arista_api import (
        AristaSpatiotemporalConfig,
        assert_package_only_runtime,
        load_arista_spatiotemporal_context,
    )

    times = _time_points(args.time_points)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cb.tl.set_global_random_seed(int(args.random_seed))

    config = AristaSpatiotemporalConfig(
        aligned_h5ad=args.aligned_h5ad,
        data_csv=args.data_csv,
        model_dir=args.model_dir,
        model_format=args.model_format,
        edge_predictor_root=args.edge_predictor_root,
        label_color_json=args.label_color_json,
        time_points=times,
        interp_time_points=(),
        n_samples=int(args.n_samples),
        split_sde_dt=float(args.dt),
        split_sigma=float(args.sigma),
        split_growth_alpha=float(args.growth_alpha),
        random_seed=int(args.random_seed),
        device=str(args.device),
        run_communication=False,
        run_3d=False,
    )
    context = load_arista_spatiotemporal_context(config)
    runtime = context["runtime"]
    points = cb.tl.simulate_sde_points_split(
        df=context["df"],
        dim=int(context["dim"]),
        f_net=runtime.f_net,
        score_net=runtime.score_net,
        time_index=0,
        n_samples=int(args.n_samples),
        ts_points=times,
        dt=float(args.dt),
        sigma=float(args.sigma),
        growth_alpha=float(args.growth_alpha),
        interaction_m=int(args.interaction_m),
        device=str(args.device),
        verbose=True,
    )

    observed_points: dict[float, np.ndarray] = {}
    predicted_points: dict[float, np.ndarray] = {}
    predicted_weights: dict[float, np.ndarray] = {}
    feature_cols = list(context["feature_cols"])
    sample_values = pd.to_numeric(context["df"]["samples"], errors="raise").to_numpy(
        dtype=np.float64
    )
    payload: dict[str, np.ndarray] = {
        "time_points": np.asarray(times, dtype=np.float64),
        "spatial_dim": np.asarray([2], dtype=np.int64),
    }
    for index, time_value in enumerate(times):
        predicted = np.asarray(points[index], dtype=np.float32)
        observed = context["df"].loc[
            np.isclose(sample_values, time_value, rtol=0.0, atol=1e-9), feature_cols
        ].to_numpy(dtype=np.float32)
        if observed.shape[0] == 0:
            raise ValueError(f"No observed points found at t={time_value:g}.")
        predicted_points[time_value] = predicted
        observed_points[time_value] = observed
        predicted_weights[time_value] = np.full(
            predicted.shape[0], 1.0 / predicted.shape[0], dtype=np.float32
        )
        payload[f"predicted_{index}"] = predicted
        payload[f"observed_{index}"] = observed

    samples_path = output_dir / "split_distribution_samples.npz"
    np.savez_compressed(samples_path, **payload)
    plot_result = cb.tl.DistributionEvaluationResult(
        time_points=times,
        spatial_dim=2,
        predicted_points=predicted_points,
        predicted_weights=predicted_weights,
        observed_points=observed_points,
        metrics=pd.DataFrame(),
        settings={
            "simulation_mode": "split",
            "dt": float(args.dt),
            "sigma": float(args.sigma),
            "growth_alpha": float(args.growth_alpha),
            "interaction_m": int(args.interaction_m),
            "n_samples": int(args.n_samples),
            "random_seed": int(args.random_seed),
        },
    )
    figures: dict[str, str] = {}
    for space in ("spatial", "pca"):
        for suffix in ("svg", "png"):
            path = output_dir / f"generated_vs_observed_{space}_split.{suffix}"
            cb.tl.plot_generated_vs_observed(
                plot_result,
                space=space,
                out_path=path,
                max_points=int(args.max_plot_points),
                random_seed=int(args.random_seed),
            )
            figures[f"{space}_{suffix}"] = str(path)

    loaded = context["loaded"]
    weight_path = getattr(loaded, "weight_path", None)
    score_path = getattr(loaded, "score_path", None)
    manifest = {
        "workflow": "arista_split_distribution_panels",
        "condition": str(args.condition_name),
        "simulation": {
            "mode": "split",
            "piecewise": False,
            "spatial_warp": False,
            "time_points": list(times),
            "n_samples": int(args.n_samples),
            "dt": float(args.dt),
            "sigma": float(args.sigma),
            "growth_alpha": float(args.growth_alpha),
            "interaction_m": int(args.interaction_m),
            "random_seed": int(args.random_seed),
            "device": str(args.device),
        },
        "inputs": {
            "model_format": str(args.model_format),
            "model_dir": str(context["model_dir"]),
            "aligned_h5ad": _optional_file_manifest(context["aligned_h5ad"]),
            "data_csv": (
                _optional_file_manifest(context["data_csv"])
                if context["aligned_h5ad"] is None
                else None
            ),
            "weight_checkpoint": _optional_file_manifest(weight_path),
            "score_checkpoint": _optional_file_manifest(score_path),
        },
        "outputs": {"samples": str(samples_path), "figures": figures},
        "cytobridge_module": str(Path(cb.__file__).resolve()),
        "runtime_contract": "CytoBridge public split-SDE and plotting APIs",
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    assert_package_only_runtime()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
