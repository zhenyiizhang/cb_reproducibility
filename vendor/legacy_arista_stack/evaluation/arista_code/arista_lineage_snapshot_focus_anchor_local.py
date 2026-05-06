#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from evaluation.arista_code.arista_focus_anchor_pipeline_shared import (  # noqa: E402
    prepare_arista_focus_anchor_context,
    save_timepoint_mosaic_panels,
    save_timepoint_snapshots,
)
from evaluation.arista_code.arista_helpers import plot_sankey  # noqa: E402


def _parse_csv_floats(value: str) -> list[float]:
    if value is None:
        return []
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ARISTA timepoint SVG + lineage Sankey export with optional piecewise spatial warp."
    )
    parser.add_argument("--config-path", default="config/arista_config.yaml")
    parser.add_argument("--annotation-csv", default="data/arista_1108_with_annotation.csv")
    parser.add_argument(
        "--output-dir",
        default="results/arista_lineage_snapshot_focus_anchor_local",
    )
    parser.add_argument(
        "--time-points",
        default="0.0,1.0,2.0,3.0,4.0",
        help="Observed time points to keep, comma-separated.",
    )
    parser.add_argument(
        "--interp-time-points",
        default="0.5,1.5,2.5,3.5",
        help="Interpolated time points, comma-separated.",
    )
    parser.add_argument("--target-total-slices", type=int, default=None)
    parser.add_argument("--use-real-for-observed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spatial-warp-to-observed-piecewise", action="store_true")
    parser.add_argument("--spatial-warp-k", type=int, default=8)
    parser.add_argument("--spatial-warp-eps", type=float, default=1e-6)
    parser.add_argument(
        "--classifier-knn-neighbors",
        type=int,
        default=10,
        help="KNN refinement neighbors after MLP classification. Set to 1 to disable refinement.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    color_h5ad_path = os.path.join(PROJECT_ROOT, "spatial_data", "Regeneration.h5ad")
    color_h5ad = color_h5ad_path if os.path.exists(color_h5ad_path) else None

    ctx = prepare_arista_focus_anchor_context(
        config_path=args.config_path,
        annotation_csv=args.annotation_csv,
        label_color_json=None,
        color_h5ad=color_h5ad,
        output_dir=output_dir,
        random_seed=args.random_seed,
        n_samples=7668,
        sde_dt=0.05,
        classifier_epochs=1000,
        classifier_hidden=128,
        classifier_n_pcs=None,
        classifier_knn_neighbors=int(args.classifier_knn_neighbors),
        classifier_best_metric="bacc",
        classifier_train_on_full_data=False,
        classifier_save_test_models=False,
        classifier_last_k_epochs=5,
        classifier_checkpoint_dir=None,
        classifier_cache=True,
        classifier_cache_dir=None,
        classifier_cache_tag=None,
        split_sde_dt=0.05,
        split_sigma=0.03,
        split_growth_alpha=1.0,
        interp_time_points=_parse_csv_floats(args.interp_time_points),
        target_total_slices=args.target_total_slices,
        use_real_for_observed=bool(args.use_real_for_observed),
        time_points_override=_parse_csv_floats(args.time_points),
        spatial_warp_to_observed_piecewise=bool(args.spatial_warp_to_observed_piecewise),
        spatial_warp_k=int(args.spatial_warp_k),
        spatial_warp_eps=float(args.spatial_warp_eps),
    )

    label_to_color = ctx["label_to_color"]
    with open(os.path.join(output_dir, "label_to_color.json"), "w", encoding="utf-8") as f:
        json.dump(label_to_color, f, indent=2)
    print("Saved:", os.path.join(output_dir, "label_to_color.json"))

    snapshot_dir = os.path.join(output_dir, "timepoint_svg")
    save_timepoint_snapshots(
        adata_dict=ctx["adata_dict"],
        time_keys=ctx["time_keys"],
        label_to_color=label_to_color,
        annotation_key="Annotation",
        snapshot_dir=snapshot_dir,
        background_color=None,
        font_color="#1a1a1a",
        snapshot_point_size=2.5,
        snapshot_alpha=0.9,
        mosaic_cols=4,
        mosaic_cell_size=2.2,
        mosaic_show_title=True,
        save_pdf=True,
    )

    compare_time_keys = [str(t) for t in ctx["ts_points"]]
    if compare_time_keys and ctx["generated_adata_dict"]:
        compare_panels = []
        for tk in compare_time_keys:
            if tk in ctx["observed_adata_dict"]:
                ad_obs = ctx["observed_adata_dict"][tk]
                compare_panels.append(
                    {
                        "title": f"t = {tk} | Observed",
                        "coords": np.asarray(ad_obs.obsm["spatial"]),
                        "labels": ad_obs.obs["Annotation"].astype(str).values,
                    }
                )
            if tk in ctx["generated_adata_dict"]:
                ad_gen = ctx["generated_adata_dict"][tk]
                compare_panels.append(
                    {
                        "title": f"t = {tk} | Generated",
                        "coords": np.asarray(ad_gen.obsm["spatial"]),
                        "labels": ad_gen.obs["Annotation"].astype(str).values,
                    }
                )
        save_timepoint_mosaic_panels(
            panels=compare_panels,
            label_to_color=label_to_color,
            snapshot_dir=snapshot_dir,
            background_color=None,
            font_color="#1a1a1a",
            snapshot_point_size=2.5,
            snapshot_alpha=0.9,
            mosaic_cols=4,
            mosaic_cell_size=2.2,
            mosaic_show_title=True,
            output_name="timepoint_mosaic.svg",
        )
        print("Saved observed/generated comparison mosaic to:", os.path.join(snapshot_dir, "timepoint_mosaic.svg"))
    print("Saved snapshots to:", snapshot_dir)

    sankey_path = os.path.join(output_dir, "lineage_sankey.html")
    fig_sankey = plot_sankey(
        predicted_labels_list=ctx["predicted_labels_list"],
        out_html=sankey_path,
        time_keys=ctx["time_keys"],
        show_time_axis=True,
        min_flow=None,
        keep_source_cumfrac=0.8,
        normalize_mode=None,
        label_to_color=label_to_color,
        style="nature-methods",
        title="Cell Fate Transitions",
    )
    print("Saved:", sankey_path)

    try:
        import plotly.io as pio

        pio.write_image(fig_sankey, os.path.join(output_dir, "lineage_sankey.svg"), scale=3)
        pio.write_image(fig_sankey, os.path.join(output_dir, "lineage_sankey.pdf"), scale=3)
        print("Exported lineage Sankey static files.")
    except Exception as exc:
        print("Static Sankey export failed:", exc)


if __name__ == "__main__":
    main()
