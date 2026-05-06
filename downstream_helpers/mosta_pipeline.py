#!/usr/bin/env python3
"""
Importable MOSTA downstream pipeline used by both notebooks and the CLI wrapper.

This module keeps a single implementation path for:
- piecewise/non-piecewise interpolation
- lineage Sankey generation
- 3D spatiotemporal communication plots
- attention export and communication aggregation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYTOBRIDGE_REPO = Path(
    os.environ.get("CYTOBRIDGE_REPO", str(PROJECT_ROOT.parent / "cytobridge-spatial"))
).resolve()
VENDOR_ROOT = PROJECT_ROOT / "vendor"


def _bootstrap_module_paths() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(CYTOBRIDGE_REPO) not in sys.path:
        sys.path.insert(0, str(CYTOBRIDGE_REPO))
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))


_bootstrap_module_paths()

from CytoBridge.tl import (  # noqa: E402
    build_dynamical_runtime,
    compute_timepoint_communications,
    load_label_to_color,
    load_legacy_dynamical_model_from_dir,
    predict_labels_for_points,
    plot_lineage_sankey,
    plot_spatiotemporal_3d,
    run_interpolation_workflow,
    save_timepoint_snapshots,
)
from CytoBridge.tl.downstream.pipeline_utils import (  # noqa: E402
    downsample_xy,
    parse_boolish,
    parse_csv_floats,
    parse_csv_floats_or_all,
    require_columns,
    resolve_split_sigma,
)
from CytoBridge.utils.config import load_config  # noqa: E402
from evaluation.arista_code.arista_helpers import plot_sankey  # noqa: E402
from evaluation.arista_code.arista_helpers_focus_anchor import (  # noqa: E402
    plot_3d_spatial_sankey_style_focus_anchor,
)


# =========================
# User-editable parameters
# =========================

# You can either edit these defaults, or override a subset via CLI flags.

MOSTA_ASSET_ROOT = PROJECT_ROOT / "assets" / "mosta"
MOSTA_DATA_ROOT = PROJECT_ROOT / "data" / "mosta"
MOSTA_RESULTS_ROOT = PROJECT_ROOT / "results"

# Data / model
config_path = str(MOSTA_ASSET_ROOT / "mosta_config.yaml")
data_csv = str(MOSTA_DATA_ROOT / "mosta_four_time_with_celltype_refined.csv")
annotation_key = "Annotation"
model_dir = str(MOSTA_ASSET_ROOT / "model")
edge_predictor_root = str(MOSTA_ASSET_ROOT / "edge_classifier")

# Output
output_dir = str(MOSTA_RESULTS_ROOT / "mosta_interp_0_3_0208_n_pc_12_aniso")

# Optional: reuse annotation colors from h5ad (loaded backed='r' to avoid 38GB RAM usage)
color_h5ad = None
label_color_json = str(MOSTA_ASSET_ROOT / "label_to_color.json")

# Random seed (set None to disable)
random_seed = 42

# Observed/interpolated timepoints
# - If `interp_time_points` is empty, no SDE simulation is needed (only real data used).
# - If non-empty, we simulate trajectories and classify simulated cells for lineage ribbons.
interp_time_points = [0.5, 1.5, 2.5]
target_total_slices = None  # set e.g. 5 to downselect observed slices evenly
use_real_for_observed = True

# 3D plot: which timepoints to render as slices (communication 3D).
# Set to None to render all `ts_points`. Example: [0.0, 0.5, 1.0]
plot_3d_time_points = [0.0, 0.5, 1.0]

# SDE settings (only used when interpolation requested)
sde_dt = 0.05
split_sde_dt = 0.05
split_sigma = 0.03
split_sigma_spatial = None
split_sigma_gene = None
split_sigma_by_dim = None
split_growth_alpha = 1
piecewise_observed_sample_mode = "t0_fixed"  # "t0_fixed" | "per_timepoint"
spatial_warp_to_observed = False
spatial_warp_to_observed_piecewise = False
spatial_warp_k = 8
spatial_warp_eps = 1e-6

# Classifier settings (only used when interpolation requested)
classifier_epochs = 500
classifier_hidden = 128
classifier_n_pcs = 12  # None uses x1..x52; or set int for speed (e.g. 10)
classifier_knn_neighbors = 10  # set 1 to disable KNN refinement
classifier_best_metric = "accuracy"  # "accuracy" | "bacc"
classifier_train_on_full_data = False
classifier_save_test_models = False
classifier_last_k_epochs = 5
classifier_cache_path = str(
    MOSTA_ASSET_ROOT
    / "classifier_cache"
    / "classifier_resmlp_52fb7dc647bfe334.pt"
)

# Communication aggregation settings
remove_self_loop = False
winsor_quantile = 0.995

# 3D plot controls
reverse_time_order = True
comm_focus_label = "Brain"  # focus communication edges on one label; set None to disable
comm_edge_threshold = 0.0  # draw if weight > threshold
comm_edge_top_k = 5  # per-timepoint top-K edges (set None to disable)
comm_edge_top_k_focus_label = comm_focus_label

fate_focus_label = "Brain"  # set None to disable ribbon filtering
fate_focus_mode = "source"  # 'either' | 'source' | 'target'
fate_min_flow = None  # minimum count threshold for ribbon rendering (set 0 to disable)
fate_keep_source_cumfrac = 0.8  # e.g. 0.8 keeps top outgoing links per source until >=80% coverage (None disables)

# Focus-anchor: for edges/ribbons involving focus_anchor_label, use local centroids near focus.
focus_anchor_label = fate_focus_label
focus_anchor_frac = 0.2
focus_anchor_k = None
focus_anchor_radius = None
focus_anchor_min_count = None

# Styling
background_color = None
font_color = "#1a1a1a"
comm_edge_color = "rgba(25,25,25,0.75)"
z_spacing = 1.0

# Slice styling for observed vs generated (subtle cool vs warm)
slice_border_color_observed = "#5f6a72"  # cool gray
slice_border_color_generated = "#8c6d5a"  # warm taupe
slice_fill_color_observed = "#e6f0f6"  # light blue-gray
slice_fill_color_generated = "#f6eee5"  # light sand
slice_fill_opacity = 0.5
slice_border_width = 5

# Export settings (Plotly requires kaleido for static export)
export_svg = True
export_pdf = True
export_png = True
png_scale = 2
vector_scale = 3

# Attention export: dense N x N matrix is memory-heavy for large slices.
save_dense_attention_matrix = False

# Lineage Sankey (cell fate flow) from non-split SDE labels
plot_lineage_sankey = True
sankey_min_flow = None  # None means no filtering
sankey_normalize_mode = None  # None | 'source' | 'global'
sankey_keep_source_cumfrac = 0.8  # e.g. 0.8 keeps top outgoing links per source until >=80% coverage (None disables)
sankey_style = "nature-methods"  # 'default' | 'nature-methods'


# =========================
# Helpers
# =========================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MOSTA multilayer communication + 3D focus-anchor plot")
    parser.add_argument("--config", default=config_path)
    parser.add_argument("--data-csv", default=data_csv)
    parser.add_argument("--annotation-key", default=annotation_key)
    parser.add_argument("--model-dir", default=model_dir)
    parser.add_argument("--edge-predictor-root", default=edge_predictor_root)
    parser.add_argument("--output-dir", default=output_dir)
    parser.add_argument("--color-h5ad", default=color_h5ad)
    parser.add_argument("--label-color-json", default=label_color_json)
    parser.add_argument("--interp-time-points", default=",".join(str(x) for x in interp_time_points))
    parser.add_argument("--no-interp", action="store_true", help="Disable interpolation (no SDE/classifier)")
    parser.add_argument(
        "--plot-3d-time-points",
        default=",".join(str(x) for x in plot_3d_time_points) if plot_3d_time_points is not None else "all",
        help="Comma-separated timepoints to render in the 3D communication plot (use 'all' to render all).",
    )
    parser.add_argument("--target-total-slices", type=int, default=target_total_slices)
    parser.add_argument(
        "--max-observed-timepoints",
        type=int,
        default=None,
        help=(
            "Optional hard cap on number of observed timepoints kept from the input CSV. "
            "If --plot-3d-time-points specifies observed times, those are always kept."
        ),
    )
    parser.add_argument(
        "--use-real-for-observed",
        action=argparse.BooleanOptionalAction,
        default=use_real_for_observed,
        help="Use real cells for observed timepoints (disable to use split-SDE generated points instead).",
    )
    parser.add_argument("--classifier-epochs", type=int, default=classifier_epochs)
    parser.add_argument("--classifier-hidden", type=int, default=classifier_hidden)
    parser.add_argument("--classifier-n-pcs", type=int, default=classifier_n_pcs)
    parser.add_argument(
        "--classifier-knn-neighbors",
        type=int,
        default=classifier_knn_neighbors,
        help=(
            "KNN neighbors used to spatially refine classifier labels after MLP prediction. "
            "Set to 1 to disable refinement."
        ),
    )
    parser.add_argument(
        "--classifier-best-metric",
        choices=["accuracy", "bacc"],
        default=classifier_best_metric,
        help="Metric used to keep the best classifier epoch: accuracy or balanced accuracy (bacc).",
    )
    parser.add_argument(
        "--classifier-train-on-full-data",
        action=argparse.BooleanOptionalAction,
        default=classifier_train_on_full_data,
        help="Train classifier using all rows without train/val split.",
    )
    parser.add_argument(
        "--classifier-save-test-models",
        action=argparse.BooleanOptionalAction,
        default=classifier_save_test_models,
        help="Save classifier checkpoints for offline testing: best_acc.pt, best_bacc.pt, and last-K epochs.",
    )
    parser.add_argument(
        "--classifier-last-k-epochs",
        type=int,
        default=classifier_last_k_epochs,
        help="How many final epoch checkpoints to save when --classifier-save-test-models is enabled.",
    )
    parser.add_argument(
        "--classifier-checkpoint-dir",
        default=None,
        help="Directory for classifier checkpoints (default: <output_dir>/classifier_checkpoints).",
    )
    parser.add_argument(
        "--classifier-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse/save classifier weights in a cache dir to avoid retraining every run.",
    )
    parser.add_argument(
        "--classifier-cache-dir",
        default=None,
        help="Directory for classifier cache files (default: <output_dir>/classifier_cache).",
    )
    parser.add_argument(
        "--classifier-cache-path",
        default=classifier_cache_path,
        help="Path to a precomputed classifier cache .pt file. If unset, auto-detect under classifier_cache/.",
    )
    parser.add_argument(
        "--classifier-cache-tag",
        default=None,
        help="Optional tag mixed into the cache key (useful if you want separate caches).",
    )
    parser.add_argument(
        "--split-sde-piecewise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run split-SDE for interpolation in piecewise segments, restarting from each observed timepoint's "
            "real cell distribution (e.g. 0->1, then 1->2, etc.)."
        ),
    )
    parser.add_argument(
        "--split-sde-piecewise-include-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In piecewise split-SDE mode, also integrate to the next observed endpoint even if not needed for "
            "requested interpolation times (slower, but matches 'run to 1, then restart at 1' literally)."
        ),
    )
    parser.add_argument(
        "--piecewise-observed-sample-mode",
        choices=["t0_fixed", "per_timepoint"],
        default=piecewise_observed_sample_mode,
        help=(
            "Observed-start sampling policy for --split-sde-piecewise: "
            "t0_fixed uses one global cap from earliest observed timepoint; "
            "per_timepoint uses min(--sde-n-samples, cells_at_that_timepoint) independently per observed time."
        ),
    )
    parser.add_argument(
        "--split-sigma",
        type=float,
        default=split_sigma,
        help="Scalar split-SDE noise used when no per-dimension override is set.",
    )
    parser.add_argument(
        "--split-sigma-spatial",
        type=float,
        default=split_sigma_spatial,
        help="Override split-SDE sigma for spatial dimensions x1,x2.",
    )
    parser.add_argument(
        "--split-sigma-gene",
        type=float,
        default=split_sigma_gene,
        help="Override split-SDE sigma for non-spatial dimensions x3..xN.",
    )
    parser.add_argument(
        "--split-sigma-by-dim",
        default=split_sigma_by_dim,
        help="Optional comma-separated split-SDE sigma for every dimension x1..xN.",
    )
    parser.add_argument(
        "--spatial-warp-to-observed",
        action=argparse.BooleanOptionalAction,
        default=spatial_warp_to_observed,
        help=(
            "After split-SDE, warp only spatial dims x1,x2 toward each observed endpoint using a "
            "segment-wise nearest-neighbor displacement field. Gene dims are unchanged."
        ),
    )
    parser.add_argument(
        "--spatial-warp-to-observed-piecewise",
        nargs="?",
        const="true",
        default="true" if spatial_warp_to_observed_piecewise else "false",
        help=(
            "Run split-SDE in observed-time segments; after each segment, warp only spatial dims x1,x2 "
            "toward the observed endpoint shape and use the warped endpoint as the next segment start."
        ),
    )
    parser.add_argument(
        "--spatial-warp-k",
        type=int,
        default=spatial_warp_k,
        help="Number of simulated endpoint anchors used to interpolate spatial warp displacements.",
    )
    parser.add_argument(
        "--spatial-warp-eps",
        type=float,
        default=spatial_warp_eps,
        help="Small positive constant for inverse-distance spatial warp weights.",
    )
    parser.add_argument(
        "--skip-nonsplit-sde",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip non-split SDE simulation and only run split/piecewise interpolation.",
    )
    parser.add_argument(
        "--plot-lineage-sankey",
        action=argparse.BooleanOptionalAction,
        default=plot_lineage_sankey,
        help="Plot a lineage Sankey (cell fate flow) over all timepoints using non-split SDE labels.",
    )
    parser.add_argument("--sankey-min-flow", type=float, default=sankey_min_flow)
    parser.add_argument(
        "--sankey-keep-source-cumfrac",
        type=float,
        default=sankey_keep_source_cumfrac,
        help=(
            "Optional proportion filter for the lineage Sankey: for each source label within each time slice, keep the "
            "strongest outgoing transitions until cumulative flow coverage reaches this fraction (e.g. 0.8). Range: (0, 1]."
        ),
    )
    parser.add_argument(
        "--sankey-style",
        choices=["default", "nature-methods"],
        default=sankey_style,
        help="Styling preset for the lineage Sankey (recommended: nature-methods).",
    )
    parser.add_argument(
        "--sankey-normalize-mode",
        choices=["none", "source", "global"],
        default="none" if sankey_normalize_mode is None else sankey_normalize_mode,
        help="Normalize Sankey link values: none|source|global.",
    )
    parser.add_argument(
        "--sde-n-samples",
        type=int,
        default=None,
        help="Number of unique initial cells sampled at the earliest observed timepoint for SDE simulation (no replacement).",
    )
    parser.add_argument(
        "--slice-max-cells-per-timepoint",
        "--max-cells-per-timepoint",
        dest="slice_max_cells_per_timepoint",
        type=int,
        default=None,
        help="Optional downsample per slice for attention/3D (leave unset to use all cells).",
    )
    parser.add_argument("--skip-snapshots", action="store_true", help="Skip 2D per-timepoint SVG/PDF snapshots")
    parser.add_argument("--snapshot-point-size", type=float, default=2.5)
    parser.add_argument("--snapshot-alpha", type=float, default=0.9)
    parser.add_argument("--mosaic-cols", type=int, default=4)
    parser.add_argument("--mosaic-cell-size", type=float, default=2.2)
    parser.add_argument("--mosaic-no-title", action="store_true", help="Hide titles in the timepoint mosaic panel")
    parser.add_argument(
        "--fate-min-flow",
        type=float,
        default=fate_min_flow,
        help="Minimum count threshold for rendering fate-flow ribbons (set 0 to disable).",
    )
    parser.add_argument(
        "--fate-keep-source-cumfrac",
        type=float,
        default=fate_keep_source_cumfrac,
        help=(
            "Optional proportion filter for fate-flow ribbons: for each source label, keep the strongest outgoing "
            "transitions until cumulative count coverage reaches this fraction (e.g. 0.8). Range: (0, 1]."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=random_seed)
    parser.add_argument("--skip-export", action="store_true", help="Skip static svg/pdf/png export")
    parser.add_argument(
        "--save-dense-attention-matrix",
        action=argparse.BooleanOptionalAction,
        default=save_dense_attention_matrix,
        help=(
            "Save dense attention matrix per slice (N x N). "
            "Disable for large real timepoints to avoid O(N^2) memory usage."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.spatial_warp_to_observed_piecewise = parse_boolish(args.spatial_warp_to_observed_piecewise)
    return args


def run_with_args(args: argparse.Namespace) -> None:
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = load_config(args.config)
    if int(args.classifier_knn_neighbors) <= 0:
        raise ValueError("--classifier-knn-neighbors must be > 0")
    dim = int(config["data"]["dim"])
    split_sigma_scalar, split_sigma_vector = resolve_split_sigma(
        dim=dim,
        sigma=args.split_sigma,
        sigma_spatial=args.split_sigma_spatial,
        sigma_gene=args.split_sigma_gene,
        sigma_by_dim_text=args.split_sigma_by_dim,
    )

    df = pd.read_csv(args.data_csv, low_memory=False)
    require_columns(df, ["samples"] + [f"x{i}" for i in range(1, dim + 1)], ctx=args.data_csv)
    if args.annotation_key not in df.columns:
        raise ValueError(f"Expected '{args.annotation_key}' column in {args.data_csv}.")
    df = df.copy()
    df["samples"] = df["samples"].astype(float)
    df[args.annotation_key] = df[args.annotation_key].astype(str)
    # Ensure df['samples'].unique() (used inside simulate_sde_points*) is in time order.
    df = df.sort_values("samples").reset_index(drop=True)
    feature_cols_full = [f"x{i}" for i in range(1, dim + 1)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model = load_legacy_dynamical_model_from_dir(
        args.model_dir,
        device=device,
        edge_predictor_root=args.edge_predictor_root,
    )
    runtime = build_dynamical_runtime(loaded_model)
    model_dyn = runtime.model
    f_net = runtime.f_net
    score_net = runtime.score_net
    exp_dir = str(Path(args.model_dir).resolve())

    print("Project root:", PROJECT_ROOT)
    print("CytoBridge repo:", CYTOBRIDGE_REPO)
    print("Device:", device)
    print("Experiment dir:", exp_dir)
    print("Data:", args.data_csv, "| rows:", len(df))
    if split_sigma_vector is None:
        print("Split-SDE sigma:", split_sigma_scalar)
    else:
        print(
            "Split-SDE sigma vector | "
            f"spatial={split_sigma_vector[: min(2, len(split_sigma_vector))]} "
            f"gene_head={split_sigma_vector[2:5]} "
            f"dim={len(split_sigma_vector)}"
        )
    print(
        "Sankey settings | keep_source_cumfrac=",
        args.sankey_keep_source_cumfrac,
        "| min_flow=",
        args.sankey_min_flow,
        "| normalize_mode=",
        args.sankey_normalize_mode,
    )
    if args.spatial_warp_to_observed and args.spatial_warp_to_observed_piecewise:
        raise ValueError(
            "Use only one of --spatial-warp-to-observed or --spatial-warp-to-observed-piecewise."
        )
    if args.spatial_warp_to_observed_piecewise and args.split_sde_piecewise:
        raise ValueError(
            "--spatial-warp-to-observed-piecewise conflicts with --split-sde-piecewise; "
            "both are segment-wise restart modes."
        )
    if args.spatial_warp_to_observed and args.split_sde_piecewise and (not args.split_sde_piecewise_include_end):
        args.split_sde_piecewise_include_end = True
        print(
            "[info] enabling --split-sde-piecewise-include-end because "
            "--spatial-warp-to-observed with --split-sde-piecewise needs simulated segment endpoints as warp anchors."
        )
    if args.spatial_warp_to_observed:
        print(
            "Spatial warp: enabled | "
            f"k={args.spatial_warp_k} eps={args.spatial_warp_eps} "
            f"piecewise={args.split_sde_piecewise}"
        )
    if args.spatial_warp_to_observed_piecewise:
        print(
            "Spatial warp piecewise: enabled | "
            f"k={args.spatial_warp_k} eps={args.spatial_warp_eps}"
        )

    requested_plot_points = parse_csv_floats_or_all(args.plot_3d_time_points)
    interpolation = run_interpolation_workflow(
        df=df,
        dim=dim,
        annotation_key=args.annotation_key,
        runtime=runtime,
        device=device,
        output_dir=args.output_dir,
        requested_plot_points=requested_plot_points,
        interp_time_points=parse_csv_floats(args.interp_time_points),
        no_interp=bool(args.no_interp),
        target_total_slices=args.target_total_slices,
        max_observed_timepoints=args.max_observed_timepoints,
        use_real_for_observed=bool(args.use_real_for_observed),
        classifier_cache_path=args.classifier_cache_path,
        classifier_cache_dir=args.classifier_cache_dir,
        classifier_best_metric=args.classifier_best_metric,
        classifier_n_pcs=args.classifier_n_pcs,
        classifier_knn_neighbors=int(args.classifier_knn_neighbors),
        sde_n_samples=args.sde_n_samples,
        skip_nonsplit_sde=bool(args.skip_nonsplit_sde),
        sde_dt=sde_dt,
        split_sde_dt=split_sde_dt,
        split_sigma_scalar=split_sigma_scalar,
        split_sigma_vector=split_sigma_vector,
        split_growth_alpha=split_growth_alpha,
        split_sde_piecewise=bool(args.split_sde_piecewise),
        split_sde_piecewise_include_end=bool(args.split_sde_piecewise_include_end),
        piecewise_observed_sample_mode=args.piecewise_observed_sample_mode,
        spatial_warp_to_observed=bool(args.spatial_warp_to_observed),
        spatial_warp_to_observed_piecewise=bool(args.spatial_warp_to_observed_piecewise),
        spatial_warp_k=int(args.spatial_warp_k),
        spatial_warp_eps=float(args.spatial_warp_eps),
        slice_max_cells_per_timepoint=args.slice_max_cells_per_timepoint,
        random_seed=args.random_seed,
    )
    adata_dict = interpolation.adata_dict
    ts_points = interpolation.ts_points
    time_keys = interpolation.time_keys
    observed_time_points = interpolation.observed_time_points
    interp_points = interpolation.interp_points
    plot_3d_ts_points = interpolation.plot_3d_ts_points
    plot_3d_time_keys = interpolation.plot_3d_time_keys
    predicted_labels_list = interpolation.predicted_labels_list
    predicted_labels_split = interpolation.predicted_labels_split
    predicted_labels_split_prewarp = interpolation.predicted_labels_split_prewarp
    sde_points_split = interpolation.sde_points_split
    sde_points_split_prewarp = interpolation.sde_points_split_prewarp
    piecewise_x0_by_observed = interpolation.piecewise_x0_by_observed
    piecewise_labels_by_observed = interpolation.piecewise_labels_by_observed
    piecewise_endpoint_by_observed = interpolation.piecewise_endpoint_by_observed
    model = interpolation.classifier_model
    label_encoder = interpolation.label_encoder
    classifier_feature_dim = interpolation.classifier_feature_dim

    # Colors
    label_to_color = load_label_to_color(
        df[args.annotation_key].astype(str).values,
        label_color_json=args.label_color_json,
        color_h5ad=args.color_h5ad,
        annotation_key=args.annotation_key,
    )
    with open(os.path.join(args.output_dir, "label_to_color.json"), "w", encoding="utf-8") as f:
        json.dump(label_to_color, f, indent=2)

    # Lineage Sankey (cell fate flow) over all timepoints (0, 0.5, 1, 1.5, 2, 2.5, 3 by default)
    fig_sankey = None
    if args.plot_lineage_sankey:
        if predicted_labels_list is None:
            print("[warn] --plot-lineage-sankey requested but interpolation/classifier was not run; skipping.")
        else:
            sankey_path = os.path.join(args.output_dir, "lineage_sankey.html")
            normalize_mode = None if args.sankey_normalize_mode == "none" else args.sankey_normalize_mode
            fig_sankey = plot_lineage_sankey(
                plot_fn=plot_sankey,
                predicted_labels_list=predicted_labels_list,
                time_keys=time_keys,
                label_to_color=label_to_color,
                out_html=sankey_path,
                min_flow=args.sankey_min_flow,
                keep_source_cumfrac=args.sankey_keep_source_cumfrac,
                normalize_mode=normalize_mode,
                style=args.sankey_style,
                title="Cell Fate Transitions",
                show_time_axis=True,
            )

    if not args.skip_snapshots:
        snapshot_dir = os.path.join(args.output_dir, "timepoint_svg")
        observed_variants = None
        if len(observed_time_points) > 0:
            feature_cols_snapshot = [f"x{i}" for i in range(1, dim + 1)]
            rng_snapshot = np.random.default_rng(
                100 if args.random_seed is None else int(args.random_seed) + 100
            )
            ts_index = {float(t): i for i, t in enumerate(ts_points)}
            compare_variants = {}
            has_generated_compare = False

            for t_obs in observed_time_points:
                t_obs_f = float(t_obs)
                subset_obs = df[df["samples"] == t_obs_f]
                if subset_obs.empty:
                    continue

                X_obs = subset_obs[feature_cols_snapshot].values.astype(np.float32)
                labels_obs = subset_obs[args.annotation_key].astype(str).values
                X_obs, labels_obs = downsample_xy(
                    X_obs,
                    labels_obs,
                    args.slice_max_cells_per_timepoint,
                    rng_snapshot,
                )
                compare_variants.setdefault(t_obs_f, {})["observed"] = (
                    np.asarray(X_obs, dtype=np.float32)[:, :2],
                    np.asarray(labels_obs).astype(str),
                )

                X_gen = None
                labels_gen = None
                idx = ts_index.get(t_obs_f)
                use_final_generated = (
                    args.spatial_warp_to_observed_piecewise
                    or (not args.split_sde_piecewise)
                    or (
                        args.split_sde_piecewise
                        and args.spatial_warp_to_observed
                        and (not args.use_real_for_observed)
                    )
                )
                if (
                    use_final_generated
                    and sde_points_split is not None
                    and predicted_labels_split is not None
                    and idx is not None
                ):
                    X_gen = np.asarray(sde_points_split[idx], dtype=np.float32)
                    labels_gen = np.asarray(predicted_labels_split[idx]).astype(str)
                elif (
                    piecewise_endpoint_by_observed is not None
                    and t_obs_f in piecewise_endpoint_by_observed
                ):
                    X_gen = np.asarray(piecewise_endpoint_by_observed[t_obs_f], dtype=np.float32)
                    labels_gen = predict_labels_for_points(
                        points=X_gen,
                        time_value=t_obs_f,
                        model=model,
                        label_encoder=label_encoder,
                        feature_dim=classifier_feature_dim,
                        device=device,
                        knn_neighbors=int(args.classifier_knn_neighbors),
                    )

                if X_gen is not None and labels_gen is not None:
                    X_gen, labels_gen = downsample_xy(
                        X_gen,
                        labels_gen,
                        args.slice_max_cells_per_timepoint,
                        rng_snapshot,
                    )
                    compare_variants.setdefault(t_obs_f, {})["generated"] = (
                        np.asarray(X_gen, dtype=np.float32)[:, :2],
                        np.asarray(labels_gen).astype(str),
                    )
                    has_generated_compare = True

            if has_generated_compare:
                observed_variants = compare_variants

            # Preserve pre/post-warp inspection panels for interior interpolation slices.
            if (
                args.split_sde_piecewise
                and args.split_sde_piecewise_include_end
                and args.spatial_warp_to_observed
                and sde_points_split_prewarp is not None
                and predicted_labels_split_prewarp is not None
                and sde_points_split is not None
                and predicted_labels_split is not None
            ):
                if observed_variants is None:
                    observed_variants = {}
                observed_set = {float(t) for t in observed_time_points}
                for t_val in ts_points:
                    t_float = float(t_val)
                    if t_float in observed_set:
                        continue
                    idx = ts_index.get(t_float)
                    if idx is None:
                        continue
                    observed_variants.setdefault(t_float, {})["prewarp"] = (
                        np.asarray(sde_points_split_prewarp[idx], dtype=np.float32)[:, :2],
                        np.asarray(predicted_labels_split_prewarp[idx]).astype(str),
                    )
                    observed_variants.setdefault(t_float, {})["postwarp"] = (
                        np.asarray(sde_points_split[idx], dtype=np.float32)[:, :2],
                        np.asarray(predicted_labels_split[idx]).astype(str),
                    )
        save_timepoint_snapshots(
            adata_dict=adata_dict,
            time_keys=time_keys,
            annotation_key=args.annotation_key,
            label_to_color=label_to_color,
            observed_variants=observed_variants,
            snapshot_dir=snapshot_dir,
            background_color=background_color,
            font_color=font_color,
            snapshot_point_size=args.snapshot_point_size,
            snapshot_alpha=args.snapshot_alpha,
            mosaic_cols=args.mosaic_cols,
            mosaic_cell_size=args.mosaic_cell_size,
            mosaic_show_title=(not args.mosaic_no_title),
            save_pdf=bool(export_pdf),
        )
        print("Saved snapshots to:", snapshot_dir)

    attn_dir = os.path.join(args.output_dir, "attention")
    comm_path = os.path.join(args.output_dir, "mosta_all_time_communications.pkl")
    all_time_communications = compute_timepoint_communications(
        adata_dict=adata_dict,
        time_points=plot_3d_ts_points,
        annotation_key=args.annotation_key,
        f_net=f_net,
        device=device,
        out_dir=attn_dir,
        save_dense_attention_matrix=bool(args.save_dense_attention_matrix),
        remove_self_loop=remove_self_loop,
        winsor_quantile=winsor_quantile,
        save_pickle_path=comm_path,
    )

    # 3D plot
    focus_source_only = fate_focus_mode == "source"
    focus_target_only = fate_focus_mode == "target"

    spatiotemporal_path = os.path.join(args.output_dir, "spatiotemporal_3d.html")
    fig_3d = plot_spatiotemporal_3d(
        plot_fn=plot_3d_spatial_sankey_style_focus_anchor,
        adata_dict=adata_dict,
        all_time_communications=all_time_communications,
        time_keys=plot_3d_time_keys,
        plot_time_points=plot_3d_ts_points,
        ts_points=ts_points,
        observed_time_points=observed_time_points,
        interp_points=interp_points,
        annotation_key=args.annotation_key,
        label_to_color=label_to_color,
        out_html=spatiotemporal_path,
        predicted_labels_list=predicted_labels_list,
        spatial_key="spatial",
        z_spacing=z_spacing,
        reverse_time_order=reverse_time_order,
        intra_threshold=comm_edge_threshold,
        edge_focus_celltype=comm_focus_label,
        edge_top_k=comm_edge_top_k,
        edge_top_k_focus_label=comm_edge_top_k_focus_label,
        ribbon_min_count=float(args.fate_min_flow) if args.fate_min_flow is not None else None,
        ribbon_keep_source_cumfrac=args.fate_keep_source_cumfrac,
        ribbon_focus_celltype=fate_focus_label,
        ribbon_focus_source_only=focus_source_only if fate_focus_label else False,
        ribbon_focus_target_only=focus_target_only if fate_focus_label else False,
        background_color=background_color,
        font_color=font_color,
        anchor_mode="centroid",
        anchor_subsample=1000,
        highlight_endpoints=True,
        endpoint_size=6,
        endpoint_opacity=0.9,
        edge_color=comm_edge_color,
        edge_line_width_base=5,
        edge_line_width_scale=0.7,
        bidirectional_offset=0.2,
        bidirectional_curve=True,
        bidirectional_curve_points=18,
        ribbon_line_width_base=6,
        ribbon_line_width_scale=1.0,
        ribbon_line_alpha=0.55,
        ribbon_line_curve=0.12,
        ribbon_line_points=18,
        point_size=1.0,
        observed_point_subsample=None,
        generated_point_subsample=None,
        observed_point_alpha=0.7,
        generated_point_alpha=0.7,
        slices_only=False,
        show_time_axis=False,
        show_legend=False,
        show_title=False,
        show_slice_border=True,
        slice_border_width=slice_border_width,
        slice_border_color_observed=slice_border_color_observed,
        slice_border_color_generated=slice_border_color_generated,
        slice_fill_color_observed=slice_fill_color_observed,
        slice_fill_color_generated=slice_fill_color_generated,
        slice_fill_opacity=slice_fill_opacity,
        width=1400,
        height=1000,
        focus_anchor_label=focus_anchor_label,
        focus_anchor_k=focus_anchor_k,
        focus_anchor_frac=focus_anchor_frac,
        focus_anchor_radius=focus_anchor_radius,
        focus_anchor_min_count=focus_anchor_min_count,
    )
    try:
        fig_3d.update_layout(
            scene_camera=dict(
                eye=dict(x=1.7, y=1.0, z=0.9),
                projection=dict(type="orthographic"),
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            scene=dict(
                domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
                aspectratio=dict(x=1.2, y=1.0, z=1.6),
            ),
            font=dict(family="Helvetica", size=16, color=font_color),
        )
    except Exception:
        pass

    try:
        if args.skip_export:
            return
        import plotly.io as pio

        if fig_sankey is not None:
            if export_svg:
                pio.write_image(fig_sankey, os.path.join(args.output_dir, "lineage_sankey.svg"), scale=vector_scale)
            if export_pdf:
                pio.write_image(fig_sankey, os.path.join(args.output_dir, "lineage_sankey.pdf"), scale=vector_scale)
        if export_svg:
            pio.write_image(fig_3d, os.path.join(args.output_dir, "spatiotemporal_3d.svg"), scale=vector_scale)
        if export_pdf:
            pio.write_image(fig_3d, os.path.join(args.output_dir, "spatiotemporal_3d.pdf"), scale=vector_scale)
        if export_png:
            pio.write_image(fig_3d, os.path.join(args.output_dir, "spatiotemporal_3d.png"), scale=png_scale)
        print("Exported vector/bitmap files.")
        print("Note: Plotly 3D exports are rasterized inside SVG/PDF.")
    except Exception as exc:
        print("Export failed (likely missing kaleido):", exc)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_with_args(parse_args(argv))


if __name__ == "__main__":
    main()
