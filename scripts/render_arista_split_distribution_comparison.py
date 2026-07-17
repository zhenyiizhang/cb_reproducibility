#!/usr/bin/env python3
"""Render several saved ARISTA split-SDE runs with matched plot semantics."""

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
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        metavar="NAME=NPZ",
        help="Repeat for every split_distribution_samples.npz to compare.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--axis-scope",
        choices=("per_time", "global"),
        default="per_time",
        help="Use common axes across runs at each time or across all runs/times.",
    )
    parser.add_argument("--formats", default="svg,png")
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--observed-color", default="#4C4C4C")
    parser.add_argument("--generated-color", default="#A33A3A")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _named_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected NAME=NPZ, received {text!r}.")
    name, raw_path = text.split("=", 1)
    name = name.strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid condition name {name!r}.")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def _load_result(cb, path: Path):
    with np.load(path) as payload:
        time_points = tuple(float(value) for value in payload["time_points"])
        spatial_dim = int(np.asarray(payload["spatial_dim"]).reshape(-1)[0])
        predicted_points = {}
        predicted_weights = {}
        observed_points = {}
        stored_weight_keys = []
        for index, time_value in enumerate(time_points):
            predicted = np.asarray(payload[f"predicted_{index}"], dtype=np.float32)
            observed = np.asarray(payload[f"observed_{index}"], dtype=np.float32)
            weight_key = f"predicted_weights_{index}"
            if weight_key in payload:
                weights = np.asarray(payload[weight_key], dtype=np.float32)
                stored_weight_keys.append(weight_key)
            else:
                weights = np.full(
                    predicted.shape[0], 1.0 / predicted.shape[0], dtype=np.float32
                )
            predicted_points[time_value] = predicted
            predicted_weights[time_value] = weights
            observed_points[time_value] = observed
    return cb.tl.DistributionEvaluationResult(
        time_points=time_points,
        spatial_dim=spatial_dim,
        predicted_points=predicted_points,
        predicted_weights=predicted_weights,
        observed_points=observed_points,
        metrics=pd.DataFrame(),
        settings={
            "source": str(path),
            "stored_weight_keys": stored_weight_keys,
        },
    )


def _formats(text: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in text.split(",") if value.strip())
    if not values or any(value not in {"svg", "png", "pdf"} for value in values):
        raise ValueError("--formats must be a comma-separated subset of svg,png,pdf.")
    return tuple(dict.fromkeys(values))


def main() -> int:
    args = _parser().parse_args()
    if args.max_plot_points <= 0:
        raise ValueError("--max-plot-points must be positive.")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive.")
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must be in (0, 1].")

    import CytoBridge as cb

    named_paths = [_named_path(value) for value in args.sample]
    names = [name for name, _ in named_paths]
    if len(set(names)) != len(names):
        raise ValueError("Every --sample NAME must be unique.")
    formats = _formats(args.formats)
    results = {name: _load_result(cb, path) for name, path in named_paths}
    time_grids = {result.time_points for result in results.values()}
    if len(time_grids) != 1:
        raise ValueError("All saved runs must use the same time-point grid.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = list(results.values())
    limits_by_space = {
        space: cb.tl.compute_generated_vs_observed_plot_limits(
            all_results,
            space=space,
            scope=str(args.axis_scope),
        )
        for space in ("spatial", "pca")
    }
    output_paths: dict[str, dict[str, str]] = {}
    for name, result in results.items():
        condition_dir = output_dir / name
        condition_dir.mkdir(parents=True, exist_ok=True)
        output_paths[name] = {}
        for space in ("spatial", "pca"):
            for suffix in formats:
                path = condition_dir / f"generated_vs_observed_{space}_split.{suffix}"
                cb.tl.plot_generated_vs_observed(
                    result,
                    space=space,
                    out_path=path,
                    max_points=int(args.max_plot_points),
                    random_seed=int(args.random_seed),
                    axis_limits=limits_by_space[space],
                    point_size=float(args.point_size),
                    generated_point_size_mode="fixed",
                    observed_color=str(args.observed_color),
                    generated_color=str(args.generated_color),
                    alpha=float(args.alpha),
                )
                output_paths[name][f"{space}_{suffix}"] = str(path)

    manifest = {
        "workflow": "render_arista_split_distribution_comparison",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in named_paths
        },
        "time_points": list(next(iter(time_grids))),
        "axis_limit_scope": str(args.axis_scope),
        "axis_limits": {
            space: {
                f"{time_value:g}": [list(xlim), list(ylim)]
                for time_value, (xlim, ylim) in limits.items()
            }
            for space, limits in limits_by_space.items()
        },
        "plot_semantics": {
            "colors": {
                "observed": str(args.observed_color),
                "generated": str(args.generated_color),
            },
            "alpha": float(args.alpha),
            "point_size": {
                "mode": "fixed",
                "marker_area_points_squared": float(args.point_size),
                "predicted_weights_encoded": False,
            },
            "split_growth": (
                "Growth is represented by particle splitting/extinction and therefore "
                "by the number/density of points, not by marker area."
            ),
        },
        "outputs": output_paths,
        "cytobridge_module": str(Path(cb.__file__).resolve()),
        "runtime_contract": "CytoBridge public shared-limit and distribution plotting APIs",
    }
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
