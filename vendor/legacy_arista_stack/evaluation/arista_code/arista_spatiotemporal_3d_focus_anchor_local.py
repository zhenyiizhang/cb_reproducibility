#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from evaluation.arista_code.arista_focus_anchor_pipeline_shared import (  # noqa: E402
    prepare_arista_focus_anchor_context,
)
from evaluation.arista_code.arista_helpers import (  # noqa: E402
    analyze_attention_by_celltype,
    save_interpolated_attention,
)
from evaluation.arista_code.arista_helpers_focus_anchor import (  # noqa: E402
    plot_3d_spatial_sankey_style_focus_anchor,
)


def _parse_csv_floats(value: str) -> list[float]:
    if value is None:
        return []
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARISTA spatiotemporal 3D communication plot only.")
    parser.add_argument("--config-path", default="config/arista_config.yaml")
    parser.add_argument("--annotation-csv", default="data/arista_1108_with_annotation.csv")
    parser.add_argument(
        "--output-dir",
        default="results/arista_spatiotemporal_3d_focus_anchor_local",
    )
    parser.add_argument(
        "--time-points",
        default="0.0,1.0,2.0",
        help="Observed time points to keep, comma-separated.",
    )
    parser.add_argument(
        "--interp-time-points",
        default="0.5,1.5",
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
    parser.add_argument(
        "--classifier-cache-dir",
        default="results/arista_lineage_snapshot_focus_anchor_local/classifier_cache",
        help="Reuse classifier cache from another run. Defaults to the lineage snapshot cache directory.",
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
        classifier_cache_dir=os.path.abspath(args.classifier_cache_dir) if args.classifier_cache_dir else None,
        classifier_cache_tag=None,
        split_sde_dt=0.01,
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

    attn_dir = os.path.join(output_dir, "attention")
    os.makedirs(attn_dir, exist_ok=True)
    all_time_communications = {}
    for t in ctx["ts_points"]:
        key = str(t)
        adata_t = ctx["adata_dict"][key]
        print("Time", key, "cells", adata_t.n_obs)

        attn_out = save_interpolated_attention(
            adata_t,
            time_value=float(t),
            f_net=ctx["f_net"],
            device=ctx["device"],
            out_dir=attn_dir,
        )
        comm = analyze_attention_by_celltype(
            edge_index=attn_out["edge_index"],
            attn=attn_out["attn_mean"],
            labels=adata_t.obs["Annotation"].values,
            spatial_coord=adata_t.obsm["spatial"],
            time_title=key,
            remove_self_loop=False,
            winsor_quantile=0.995,
            distance_bins=None,
            n_permutations=0,
            plot=False,
        )
        all_time_communications[key] = comm

    comm_path = os.path.join(output_dir, "arista_all_time_communications.pkl")
    with open(comm_path, "wb") as f:
        pickle.dump(all_time_communications, f)
    print("Saved:", comm_path)

    def _apply_publication_layout(fig):
        fig.update_layout(
            scene_camera=dict(
                eye=dict(x=1.7, y=1.0, z=0.9),
                projection=dict(type="orthographic"),
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            scene=dict(
                domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
                aspectratio=dict(x=1.2, y=1.0, z=1.6),
            ),
            font=dict(family="Helvetica", size=16, color="#1a1a1a"),
        )

    common_plot_kwargs = dict(
        adata_dict=ctx["adata_dict"],
        all_time_communications=all_time_communications,
        time_keys=ctx["time_keys"],
        label_to_color=label_to_color,
        predicted_labels_list=ctx["predicted_labels_list"],
        spatial_key="spatial",
        z_spacing=3.8,
        reverse_time_order=False,
        intra_threshold=0.0,
        edge_focus_celltype="reaEGC",
        edge_top_k=5,
        edge_top_k_focus_label="reaEGC",
        ribbon_min_count=10,
        ribbon_keep_source_cumfrac=0.85,
        ribbon_focus_celltype=["reaEGC"],
        ribbon_focus_source_only=True,
        ribbon_focus_target_only=False,
        background_color=None,
        font_color="#1a1a1a",
        anchor_mode="centroid",
        anchor_subsample=1000,
        highlight_endpoints=True,
        endpoint_size=6,
        endpoint_opacity=0.9,
        edge_color="rgba(25,25,25,0.75)",
        edge_show_arrows=True,
        edge_arrow_position=0.7,
        edge_arrow_in_slice_plane=True,
        edge_arrow_length_scale=0.14,
        edge_arrow_width_scale=0.65,
        edge_line_width_base=5,
        edge_line_width_scale=0.7,
        edge_center_highlight=False,
        edge_center_highlight_width_scale=0.45,
        edge_center_highlight_alpha=0.9,
        bidirectional_offset=0.2,
        bidirectional_curve=True,
        bidirectional_curve_points=18,
        ribbon_line_width_base=6,
        ribbon_line_width_scale=1.0,
        ribbon_line_alpha=0.55,
        ribbon_line_curve=0.12,
        ribbon_line_points=18,
        ribbon_center_highlight=False,
        ribbon_center_highlight_width_scale=0.5,
        ribbon_center_highlight_alpha=0.9,
        point_size=1.0,
        observed_point_subsample=None,
        generated_point_subsample=None,
        observed_point_alpha=0.7,
        generated_point_alpha=0.7,
        observed_point_line_width=0.0,
        generated_point_line_width=0.0,
        generated_point_line_color=None,
        slices_only=False,
        show_time_axis=False,
        show_legend=False,
        show_title=False,
        show_slice_border=True,
        slice_border_width=5,
        slice_border_color_observed="#5f6a72",
        slice_border_color_generated="#8c6d5a",
        slice_fill_color_observed="#e6f0f6",
        slice_fill_color_generated="#f6eee5",
        slice_fill_opacity=0.5,
        observed_time_points=ctx["time_points"],
        generated_time_points=ctx["interp_time_points"],
        focus_anchor_label="reaEGC",
        focus_anchor_k=None,
        focus_anchor_frac=0.2,
        focus_anchor_radius=None,
        focus_anchor_min_count=None,
    )
    fig_path = os.path.join(output_dir, "spatiotemporal_3d.html")
    fig_3d = plot_3d_spatial_sankey_style_focus_anchor(
        **common_plot_kwargs,
        width=int(11.69 * 300),
        height=int(8.27 * 300),
        out_html=fig_path,
    )
    _apply_publication_layout(fig_3d)
    fig_3d.write_html(fig_path)
    print("Saved:", fig_path)

    fig_path_2 = os.path.join(output_dir, "spatiotemporal_3d_2.html")
    fig_3d_2 = plot_3d_spatial_sankey_style_focus_anchor(
        **common_plot_kwargs,
        width=1200,
        height=900,
        out_html=None,
    )
    _apply_publication_layout(fig_3d_2)
    fig_3d_2.write_html(fig_path_2)
    print("Saved:", fig_path_2)

    try:
        import plotly.io as pio

        pio.write_image(
            fig_3d,
            os.path.join(output_dir, "spatiotemporal_3d.svg"),
            width=int(11.69 * 300),
            height=int(8.27 * 300),
            scale=3,
        )
        pio.write_image(
            fig_3d,
            os.path.join(output_dir, "spatiotemporal_3d.pdf"),
            width=int(11.69 * 300),
            height=int(8.27 * 300),
            scale=3,
        )
        pio.write_image(
            fig_3d,
            os.path.join(output_dir, "spatiotemporal_3d.png"),
            width=int(11.69 * 300),
            height=int(8.27 * 300),
            scale=2,
        )
        print("Exported vector/bitmap files.")
    except Exception as exc:
        print("Export failed:", exc)


if __name__ == "__main__":
    main()
