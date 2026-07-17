#!/usr/bin/env python3
"""Generate the formal no-warp ARISTA Figure 5a/5b and S13/S14 panels.

This is a dataset-level command adapter.  Simulation, classifier fitting,
communication inference, lineage construction, growth evaluation, and plotting
are delegated to :func:`downstream_helpers.arista_api.run_arista_spatiotemporal_api`.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _time_points(value: str) -> tuple[float, ...]:
    try:
        points = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"time points must be comma-separated numbers, got {value!r}"
        ) from exc
    if not points:
        raise argparse.ArgumentTypeError("at least one time point is required")
    if tuple(sorted(set(points))) != points:
        raise argparse.ArgumentTypeError("time points must be sorted and unique")
    return points


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-h5ad", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-format", choices=("current", "legacy"), default="current"
    )
    parser.add_argument("--edge-predictor-root", default=None)
    parser.add_argument("--label-color-json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--time-key", default=None)
    parser.add_argument("--obsm-key", default="X_latent")
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument(
        "--observed-time-points",
        type=_time_points,
        default=_time_points("0,1,2,3,4"),
        metavar="T0,T1,...",
    )
    parser.add_argument(
        "--interpolated-time-points",
        type=_time_points,
        default=_time_points("0.5,1.5,2.5,3.5"),
        metavar="T0,T1,...",
    )
    parser.add_argument(
        "--plot-3d-time-points",
        type=_time_points,
        default=_time_points("0,0.5,1,1.5,2"),
        metavar="T0,T1,...",
    )
    parser.add_argument("--n-samples", type=int, default=7668)
    parser.add_argument("--nonsplit-dt", type=float, default=0.05)
    parser.add_argument("--split-dt", type=float, default=0.01)
    parser.add_argument("--split-sigma", type=float, default=0.03)
    parser.add_argument("--split-growth-alpha", type=float, default=1.0)
    parser.add_argument("--random-seed", "--seed", dest="random_seed", type=int, default=42)
    parser.add_argument("--classifier-knn-neighbors", type=int, default=1)
    parser.add_argument("--classifier-epochs", type=int, default=1000)
    parser.add_argument("--classifier-hidden-size", type=int, default=128)
    parser.add_argument("--classifier-best-metric", default="bacc")
    parser.add_argument("--classifier-train-on-full-data", action="store_true")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--classifier-cache-path",
        default=None,
        help="Explicit classifier checkpoint to reuse instead of fitting a fresh one.",
    )
    cache.add_argument(
        "--classifier-cache-dir",
        default=None,
        help="Directory in which a newly fitted classifier is cached.",
    )
    parser.add_argument(
        "--reuse-classifier-cache-dir",
        action="store_true",
        help=(
            "Allow a non-empty classifier cache directory. By default the command "
            "refuses it so that the formal run fits a fresh classifier."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.nonsplit_dt <= 0 or args.split_dt <= 0:
        raise ValueError("--nonsplit-dt and --split-dt must be positive")
    if args.split_sigma < 0:
        raise ValueError("--split-sigma must be non-negative")
    if args.classifier_knn_neighbors <= 0:
        raise ValueError("--classifier-knn-neighbors must be positive")
    if args.classifier_epochs <= 0 or args.classifier_hidden_size <= 0:
        raise ValueError("classifier epochs and hidden size must be positive")
    state_times = set(args.observed_time_points) | set(args.interpolated_time_points)
    unknown_3d_times = set(args.plot_3d_time_points) - state_times
    if unknown_3d_times:
        raise ValueError(
            "--plot-3d-time-points must be present in the observed/interpolated "
            f"grid; unknown={sorted(unknown_3d_times)}"
        )
    overlap = set(args.observed_time_points) & set(args.interpolated_time_points)
    if overlap:
        raise ValueError(
            "observed and interpolated time points must be disjoint; "
            f"overlap={sorted(overlap)}"
        )
    if args.classifier_cache_path and args.reuse_classifier_cache_dir:
        raise ValueError(
            "--reuse-classifier-cache-dir applies only to a cache directory, not "
            "--classifier-cache-path"
        )


def _resolve_classifier_cache(
    args: argparse.Namespace, output_dir: Path
) -> tuple[Path | None, Path]:
    cache_path = (
        Path(args.classifier_cache_path).expanduser().resolve()
        if args.classifier_cache_path is not None
        else None
    )
    cache_dir = (
        Path(args.classifier_cache_dir).expanduser().resolve()
        if args.classifier_cache_dir is not None
        else output_dir / "classifier_cache"
    )
    if (
        cache_path is None
        and cache_dir.is_dir()
        and any(cache_dir.iterdir())
        and not args.reuse_classifier_cache_dir
    ):
        raise FileExistsError(
            f"Classifier cache directory is not empty: {cache_dir}. "
            "Use a new output/cache directory for the formal fresh fit, or pass "
            "--reuse-classifier-cache-dir explicitly."
        )
    return cache_path, cache_dir


def _existing_files(paths: Sequence[Path]) -> list[str]:
    return [str(path) for path in paths if path.is_file()]


def _validate_result_artifacts(result, *, config, cache_path: Path | None) -> None:
    snapshot_tag = str(float(config.interp_time_points[0]))
    required = {
        "API run manifest": result.manifest_path,
        "Figure 5a HTML": result.spatiotemporal_html,
        "Figure 5b SVG": result.snapshots_dir / f"time_{snapshot_tag}.svg",
        "S13 table": result.growth_csv,
        "S13 figure": result.growth_figure,
        "S14a HTML": result.lineage_html,
        "S14b table": result.composition_csv,
        "S14b figure": result.composition_figure,
        "communication pickle": result.communications_pickle,
    }
    missing = [
        f"{description}: {path}"
        for description, path in required.items()
        if path is None or not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Formal ARISTA API run did not emit every required artifact:\n- "
            + "\n- ".join(missing)
        )
    if cache_path is None:
        classifier_cache_dir = Path(result.classifier_cache_dir)
        if not classifier_cache_dir.is_dir() or not any(
            path.is_file() for path in classifier_cache_dir.iterdir()
        ):
            raise FileNotFoundError(
                "The formal fresh classifier fit did not emit a cache file under "
                f"{classifier_cache_dir}."
            )


def _write_panel_index(
    result, *, output_dir: Path, config, classifier_cache_mode: str
) -> Path:
    snapshot_t = float(config.interp_time_points[0])
    snapshot_tag = str(snapshot_t)
    panel_index = {
        "workflow": "arista_formal_panels_api",
        "source_manifest": str(result.manifest_path),
        "contract": {
            "model_format": config.model_format,
            "spatial_warp_to_observed_piecewise": False,
            "lineage_state": "non_split_fixed_particles",
            "generated_cloud_and_communication_state": "split_sde_no_warp",
            "classifier_knn_neighbors": int(config.classifier_knn_neighbors),
            "classifier_cache_mode": classifier_cache_mode,
        },
        "panels": {
            "Figure_5a": {
                "description": "No-warp spatiotemporal 3D lineage/communication panel",
                "files": _existing_files(
                    [
                        output_dir / "spatiotemporal_3d.html",
                        output_dir / "spatiotemporal_3d.svg",
                        output_dir / "spatiotemporal_3d.pdf",
                        output_dir / "spatiotemporal_3d.png",
                    ]
                ),
            },
            "Figure_5b": {
                "description": (
                    f"Generated no-warp spatial slice at model time {snapshot_tag}"
                ),
                "files": _existing_files(
                    [
                        result.snapshots_dir / f"time_{snapshot_tag}.svg",
                        result.snapshots_dir / f"time_{snapshot_tag}.pdf",
                    ]
                ),
            },
            "Figure_S13": {
                "description": "Growth-rate maps on the dense observed/interpolated grid",
                "files": _existing_files([result.growth_csv, result.growth_figure]),
            },
            "Figure_S14a": {
                "description": "Persistent-particle lineage Sankey",
                "files": _existing_files(
                    [
                        result.lineage_html,
                        output_dir / "lineage_sankey.svg",
                        output_dir / "lineage_sankey.pdf",
                        output_dir / "lineage_sankey.png",
                    ]
                ),
            },
            "Figure_S14b": {
                "description": "Cell-type composition across the dense time grid",
                "files": _existing_files(
                    [result.composition_csv, result.composition_figure]
                ),
            },
        },
    }
    panel_index_path = output_dir / "formal_panel_index.json"
    panel_index_path.write_text(
        json.dumps(panel_index, indent=2, sort_keys=True), encoding="utf-8"
    )
    return panel_index_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_path, cache_dir = _resolve_classifier_cache(args, output_dir)

    arista_api = importlib.import_module("downstream_helpers.arista_api")
    config = arista_api.AristaSpatiotemporalConfig(
        output_dir=output_dir,
        aligned_h5ad=Path(args.aligned_h5ad).expanduser().resolve(),
        model_dir=Path(args.model_dir).expanduser().resolve(),
        model_format=str(args.model_format),
        edge_predictor_root=(
            Path(args.edge_predictor_root).expanduser().resolve()
            if args.edge_predictor_root is not None
            else None
        ),
        label_color_json=(
            Path(args.label_color_json).expanduser().resolve()
            if args.label_color_json is not None
            else None
        ),
        annotation_key=str(args.annotation_key),
        time_key=args.time_key,
        obsm_key=str(args.obsm_key),
        spatial_key=str(args.spatial_key),
        concat_spatial=True,
        time_points=tuple(args.observed_time_points),
        interp_time_points=tuple(args.interpolated_time_points),
        plot_3d_time_points=tuple(args.plot_3d_time_points),
        n_samples=int(args.n_samples),
        skip_nonsplit_sde=False,
        sde_dt=float(args.nonsplit_dt),
        split_sde_dt=float(args.split_dt),
        split_sigma=float(args.split_sigma),
        split_growth_alpha=float(args.split_growth_alpha),
        use_real_for_observed=True,
        spatial_warp_to_observed_piecewise=False,
        spatial_warp_visualization_only=False,
        classifier_cache_path=cache_path,
        classifier_cache_dir=cache_dir,
        classifier_epochs=int(args.classifier_epochs),
        classifier_hidden_size=int(args.classifier_hidden_size),
        classifier_best_metric=str(args.classifier_best_metric),
        classifier_train_on_full_data=bool(args.classifier_train_on_full_data),
        classifier_knn_neighbors=int(args.classifier_knn_neighbors),
        random_seed=int(args.random_seed),
        device=str(args.device),
        run_communication=True,
        run_3d=True,
        publication_3d_layout=True,
        highlight_3d_endpoints=True,
    )
    result = arista_api.run_arista_spatiotemporal_api(config)
    arista_api.assert_package_only_runtime()
    _validate_result_artifacts(result, config=config, cache_path=cache_path)
    classifier_cache_mode = (
        "explicit_checkpoint_reuse"
        if cache_path is not None
        else "existing_directory_reuse"
        if args.reuse_classifier_cache_dir
        else "fresh_directory_fit"
    )
    panel_index_path = _write_panel_index(
        result,
        output_dir=output_dir,
        config=config,
        classifier_cache_mode=classifier_cache_mode,
    )
    print(panel_index_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
