"""Legacy DeepRUOT-backed ARISTA helpers for parity-focused reviewer notebooks."""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .runner import export_plotly_figure, list_output_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT / "vendor" / "legacy_arista_stack"
RESULTS_ROOT = PROJECT_ROOT / "results"


def _bootstrap_legacy_stack() -> None:
    for path in (LEGACY_ROOT, PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    stale = [
        name
        for name in list(sys.modules)
        if name == "evaluation"
        or name.startswith("evaluation.")
        or name == "DeepRUOT"
        or name.startswith("DeepRUOT.")
    ]
    for name in stale:
        sys.modules.pop(name, None)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _parse_float_keys(values: Sequence[float]) -> list[str]:
    return [str(float(v)) for v in values]


def resolve_arista_legacy_output_dir(name: str) -> Path:
    return (RESULTS_ROOT / name).resolve()


@dataclass(frozen=True)
class AristaLegacyLineageConfig:
    output_dir: str
    config_path: str = "vendor/legacy_arista_stack/config/arista_config.yaml"
    annotation_csv: str = "data/arista/arista_1108_with_annotation.csv"
    label_color_json: str = "assets/arista/label_to_color.json"
    classifier_cache_path: str = "assets/arista/classifier_cache/classifier_resmlp_a2ffc52825a81efd.pt"
    time_points: Sequence[float] = field(default_factory=lambda: (0.0, 1.0, 2.0, 3.0, 4.0))
    interp_time_points: Sequence[float] = field(default_factory=lambda: (0.5, 1.5, 2.5, 3.5))
    target_total_slices: Optional[int] = None
    use_real_for_observed: bool = True
    spatial_warp_to_observed_piecewise: bool = False
    spatial_warp_k: int = 8
    spatial_warp_eps: float = 1e-6
    classifier_knn_neighbors: int = 10
    random_seed: int = 42
    n_samples: int = 7668
    sde_dt: float = 0.05
    split_sde_dt: float = 0.05
    split_sigma: float = 0.03
    split_growth_alpha: float = 1.0
    classifier_epochs: int = 1000
    classifier_hidden: int = 128
    classifier_n_pcs: Optional[int] = None
    classifier_best_metric: str = "bacc"
    classifier_train_on_full_data: bool = False
    classifier_save_test_models: bool = False
    classifier_last_k_epochs: int = 5
    classifier_checkpoint_dir: Optional[str] = None
    snapshot_point_size: float = 2.5
    snapshot_alpha: float = 0.9
    mosaic_cols: int = 4
    mosaic_cell_size: float = 2.2


@dataclass(frozen=True)
class AristaLegacy3DConfig:
    output_dir: str
    config_path: str = "vendor/legacy_arista_stack/config/arista_config.yaml"
    annotation_csv: str = "data/arista/arista_1108_with_annotation.csv"
    label_color_json: str = "assets/arista/label_to_color.json"
    classifier_cache_path: str = "assets/arista/classifier_cache/classifier_resmlp_a2ffc52825a81efd.pt"
    time_points: Sequence[float] = field(default_factory=lambda: (0.0, 1.0, 2.0))
    interp_time_points: Sequence[float] = field(default_factory=lambda: (0.5, 1.5))
    target_total_slices: Optional[int] = None
    use_real_for_observed: bool = True
    spatial_warp_to_observed_piecewise: bool = False
    spatial_warp_k: int = 8
    spatial_warp_eps: float = 1e-6
    classifier_knn_neighbors: int = 10
    random_seed: int = 42
    n_samples: int = 7668
    sde_dt: float = 0.05
    split_sde_dt: float = 0.01
    split_sigma: float = 0.03
    split_growth_alpha: float = 1.0
    classifier_epochs: int = 1000
    classifier_hidden: int = 128
    classifier_n_pcs: Optional[int] = None
    classifier_best_metric: str = "bacc"
    classifier_train_on_full_data: bool = False
    classifier_save_test_models: bool = False
    classifier_last_k_epochs: int = 5
    classifier_checkpoint_dir: Optional[str] = None
    z_spacing: float = 3.8
    focus_label: str = "reaEGC"


@dataclass(frozen=True)
class AristaLegacyLineageResult:
    output_dir: Path
    snapshots_dir: Path
    label_color_json: Path
    lineage_sankey_html: Path
    lineage_sankey_svg: Path | None
    lineage_sankey_pdf: Path | None
    output_files: list[str]


@dataclass(frozen=True)
class AristaLegacy3DResult:
    output_dir: Path
    label_color_json: Path
    communications_pkl: Path
    html_path: Path
    html_path_large: Path
    svg_path: Path | None
    pdf_path: Path | None
    png_path: Path | None
    output_files: list[str]


@dataclass(frozen=True)
class AristaLegacyGrowthInteractionBubbleResult:
    output_dir: Path
    bubble_svg: Path
    bubble_png: Path
    raw_csv: Path
    grouped_csv: Path
    output_files: list[str]


@dataclass(frozen=True)
class AristaLegacyVelocityCorrelationConfig:
    output_dir: str
    config_path: str = "vendor/legacy_arista_stack/config/arista_config.yaml"
    annotation_csv: str = "data/arista/arista_1108_with_annotation.csv"
    target_timepoint: float = 1.0
    label_keyword: str = "reaEGC"
    pad_ratio: float = 0.15
    n_neighbors: int = 30
    interaction_m: int = 1024
    interaction_threshold: float = 1000.0


@dataclass(frozen=True)
class AristaLegacyVelocityCorrelationResult:
    output_dir: Path
    svg_path: Path
    png_path: Path
    roi_csv: Path
    output_files: list[str]


def _build_legacy_context(config: AristaLegacyLineageConfig | AristaLegacy3DConfig):
    _bootstrap_legacy_stack()
    from evaluation.arista_code.arista_focus_anchor_pipeline_shared import (
        prepare_arista_focus_anchor_context,
    )


def _load_legacy_velocity_inputs(
    *,
    config_path: str,
    annotation_csv: str,
):
    _bootstrap_legacy_stack()
    import pandas as pd
    from evaluation.arista_code import arista_helpers as helpers

    config = helpers.load_config(str(_resolve_path(config_path)))
    df, _ = helpers.load_arista_df(config)

    df_anno = pd.read_csv(_resolve_path(annotation_csv))
    if "Annotation" not in df_anno.columns:
        raise ValueError(f"Annotation column missing in {annotation_csv}")
    if len(df_anno) != len(df):
        raise ValueError("Annotation CSV length does not match data CSV")

    df = df.copy()
    df["Annotation"] = df_anno["Annotation"].astype(str).values
    dim = int(config["data"]["dim"])
    f_net, score_net, exp_dir, device = helpers.load_models(
        config,
        exp_name=config["exp"]["name"],
        model_tag="model_final",
        score_tag="score_model",
    )
    return {
        "config": config,
        "df": df,
        "dim": dim,
        "f_net": f_net,
        "score_net": score_net,
        "exp_dir": exp_dir,
        "device": device,
        "helpers": helpers,
    }

    output_dir = _resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return prepare_arista_focus_anchor_context(
        config_path=str(_resolve_path(config.config_path)),
        annotation_csv=str(_resolve_path(config.annotation_csv)),
        label_color_json=str(_resolve_path(config.label_color_json)),
        color_h5ad=None,
        output_dir=str(output_dir),
        random_seed=int(config.random_seed),
        n_samples=int(config.n_samples),
        sde_dt=float(config.sde_dt),
        classifier_epochs=int(config.classifier_epochs),
        classifier_hidden=int(config.classifier_hidden),
        classifier_n_pcs=config.classifier_n_pcs,
        classifier_knn_neighbors=int(config.classifier_knn_neighbors),
        classifier_best_metric=str(config.classifier_best_metric),
        classifier_train_on_full_data=bool(config.classifier_train_on_full_data),
        classifier_save_test_models=bool(config.classifier_save_test_models),
        classifier_last_k_epochs=int(config.classifier_last_k_epochs),
        classifier_checkpoint_dir=(
            str(_resolve_path(config.classifier_checkpoint_dir))
            if config.classifier_checkpoint_dir
            else None
        ),
        classifier_cache=True,
        classifier_cache_dir=None,
        classifier_cache_tag=None,
        classifier_cache_path=str(_resolve_path(config.classifier_cache_path)),
        classifier_force_load=True,
        split_sde_dt=float(config.split_sde_dt),
        split_sigma=float(config.split_sigma),
        split_growth_alpha=float(config.split_growth_alpha),
        interp_time_points=[float(x) for x in config.interp_time_points],
        target_total_slices=config.target_total_slices,
        use_real_for_observed=bool(config.use_real_for_observed),
        time_points_override=[float(x) for x in config.time_points],
        spatial_warp_to_observed_piecewise=bool(config.spatial_warp_to_observed_piecewise),
        spatial_warp_k=int(config.spatial_warp_k),
        spatial_warp_eps=float(config.spatial_warp_eps),
    )


def run_arista_legacy_lineage_snapshot(config: AristaLegacyLineageConfig) -> AristaLegacyLineageResult:
    _bootstrap_legacy_stack()
    from evaluation.arista_code.arista_focus_anchor_pipeline_shared import (
        save_timepoint_mosaic_panels,
        save_timepoint_snapshots,
    )
    from evaluation.arista_code.arista_helpers import plot_sankey

    output_dir = _resolve_path(config.output_dir)
    ctx = _build_legacy_context(config)

    label_color_path = output_dir / "label_to_color.json"
    with open(label_color_path, "w", encoding="utf-8") as f:
        json.dump(ctx["label_to_color"], f, indent=2)

    snapshots_dir = output_dir / "timepoint_svg"
    save_timepoint_snapshots(
        adata_dict=ctx["adata_dict"],
        time_keys=ctx["time_keys"],
        label_to_color=ctx["label_to_color"],
        annotation_key="Annotation",
        snapshot_dir=str(snapshots_dir),
        background_color=None,
        font_color="#1a1a1a",
        snapshot_point_size=float(config.snapshot_point_size),
        snapshot_alpha=float(config.snapshot_alpha),
        mosaic_cols=int(config.mosaic_cols),
        mosaic_cell_size=float(config.mosaic_cell_size),
        mosaic_show_title=True,
        save_pdf=True,
    )

    compare_time_keys = _parse_float_keys(ctx["ts_points"])
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
            label_to_color=ctx["label_to_color"],
            snapshot_dir=str(snapshots_dir),
            background_color=None,
            font_color="#1a1a1a",
            snapshot_point_size=float(config.snapshot_point_size),
            snapshot_alpha=float(config.snapshot_alpha),
            mosaic_cols=int(config.mosaic_cols),
            mosaic_cell_size=float(config.mosaic_cell_size),
            mosaic_show_title=True,
            output_name="timepoint_mosaic.svg",
        )

    html_path = output_dir / "lineage_sankey.html"
    fig_sankey = plot_sankey(
        predicted_labels_list=ctx["predicted_labels_list"],
        out_html=str(html_path),
        time_keys=ctx["time_keys"],
        show_time_axis=True,
        min_flow=None,
        keep_source_cumfrac=0.8,
        normalize_mode=None,
        label_to_color=ctx["label_to_color"],
        style="nature-methods",
        title="Cell Fate Transitions",
    )

    svg_path = output_dir / "lineage_sankey.svg"
    pdf_path = output_dir / "lineage_sankey.pdf"
    try:
        export_plotly_figure(fig_sankey, svg_path=svg_path, pdf_path=pdf_path, vector_scale=3)
        svg_out = svg_path
        pdf_out = pdf_path
    except Exception as exc:
        print("Static Sankey export failed:", exc)
        svg_out = None
        pdf_out = None

    return AristaLegacyLineageResult(
        output_dir=output_dir,
        snapshots_dir=snapshots_dir,
        label_color_json=label_color_path,
        lineage_sankey_html=html_path,
        lineage_sankey_svg=svg_out,
        lineage_sankey_pdf=pdf_out,
        output_files=list_output_files(output_dir),
    )


def _apply_publication_layout(fig) -> None:
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


def _collect_growth_interaction(
    *,
    adata_dict,
    ts_points: Sequence[float],
    f_net,
    device: str = "cpu",
    interaction_m: int = 1024,
    interaction_threshold: float = 1000.0,
    max_cells: Optional[int] = None,
    seed: int = 0,
):
    import pandas as pd
    import torch
    from DeepRUOT.interaction import cal_interaction

    rng = np.random.default_rng(seed)
    rows = []

    for t in ts_points:
        tk = str(t)
        adata_t = adata_dict[tk]
        X = np.asarray(adata_t.X, dtype=np.float32)
        labels = (
            adata_t.obs["Annotation"].astype(str).values
            if "Annotation" in adata_t.obs.columns
            else None
        )

        if max_cells is not None and X.shape[0] > int(max_cells):
            idx = rng.choice(X.shape[0], size=int(max_cells), replace=False)
            X = X[idx]
            if labels is not None:
                labels = labels[idx]

        n = X.shape[0]
        x_t = torch.tensor(X, device=device)
        t_tensor = torch.full((n, 1), float(t), device=device)

        with torch.no_grad():
            _, g, _, _ = f_net(t_tensor, x_t)
        g = g.detach().cpu().numpy().reshape(-1)

        lnw = torch.log(torch.ones(n, 1, device=device) / n)
        with torch.no_grad():
            inter = cal_interaction(
                x_t,
                lnw,
                f_net.interaction_net,
                torch.tensor([float(t)], dtype=torch.float32, device=device),
                m=int(interaction_m),
                threshold=float(interaction_threshold),
            )
        inter_mag = np.linalg.norm(inter.detach().cpu().numpy(), axis=1)

        block = {
            "time": tk,
            "growth": g,
            "interaction": inter_mag,
        }
        if labels is not None:
            block["celltype"] = labels
        rows.append(pd.DataFrame(block))

    return pd.concat(rows, ignore_index=True)


def run_arista_legacy_growth_interaction_celltype_bubble(
    config: AristaLegacyLineageConfig,
    *,
    interaction_m: int = 1024,
    interaction_threshold: float = 1000.0,
    max_cells: Optional[int] = None,
    seed: int = 0,
) -> AristaLegacyGrowthInteractionBubbleResult:
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = _resolve_path(config.output_dir)
    ctx = _build_legacy_context(config)

    raw_df = _collect_growth_interaction(
        adata_dict=ctx["adata_dict"],
        ts_points=ctx["ts_points"],
        f_net=ctx["f_net"],
        device=ctx["device"],
        interaction_m=int(interaction_m),
        interaction_threshold=float(interaction_threshold),
        max_cells=max_cells,
        seed=int(seed),
    )
    raw_csv = output_dir / "arista_growth_interaction_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    if "celltype" not in raw_df.columns:
        raise ValueError("The collected dataframe does not contain a 'celltype' column.")

    grouped_df = (
        raw_df.groupby(["time", "celltype"])
        .agg(
            growth_mean=("growth", "mean"),
            inter_mean=("interaction", "mean"),
            n=("growth", "size"),
        )
        .reset_index()
    )
    grouped_csv = output_dir / "arista_growth_interaction_celltype_summary.csv"
    grouped_df.to_csv(grouped_csv, index=False)

    time_keys_ordered = [str(t) for t in ctx["ts_points"]]
    time_to_idx = {t: i for i, t in enumerate(time_keys_ordered)}
    grouped_df["time_idx"] = grouped_df["time"].map(time_to_idx)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    sc = ax.scatter(
        grouped_df["inter_mean"],
        grouped_df["growth_mean"],
        s=np.clip(grouped_df["n"], 20, 400),
        c=grouped_df["time_idx"],
        cmap="plasma",
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.set_xlabel("Mean interaction magnitude")
    ax.set_ylabel("Mean growth (g)")
    ax.grid(False)

    n_vals = grouped_df["n"].to_numpy()
    if n_vals.size > 0:
        examples = np.unique(np.round(np.percentile(n_vals, [0, 50, 100])).astype(int))
        examples = examples[examples > 0]
        handles = [
            ax.scatter(
                [],
                [],
                s=np.clip(v, 20, 400),
                color="#777777",
                alpha=0.9,
                edgecolors="white",
                linewidths=0.35,
            )
            for v in examples
        ]
        labels = [f"{v} cells" for v in examples]
        size_leg = ax.legend(
            handles,
            labels,
            title="Dot = one (time, celltype)\nSize = cells in group",
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.02, 1.0),
        )
        ax.add_artist(size_leg)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Time index")
    ax.text(
        0.01,
        0.01,
        "Each dot: mean of a celltype at one timepoint;\nsize = group cell count.",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
    )

    plt.tight_layout()
    svg_path = output_dir / "arista_growth_interaction_celltype_bubble.svg"
    png_path = output_dir / "arista_growth_interaction_celltype_bubble.png"
    fig.savefig(svg_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return AristaLegacyGrowthInteractionBubbleResult(
        output_dir=output_dir,
        bubble_svg=svg_path,
        bubble_png=png_path,
        raw_csv=raw_csv,
        grouped_csv=grouped_csv,
        output_files=list_output_files(output_dir),
    )


def run_arista_legacy_velocity_spatial_direction_correlation_roi(
    config: AristaLegacyVelocityCorrelationConfig,
) -> AristaLegacyVelocityCorrelationResult:
    import anndata as ad
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scanpy as sc
    import scvelo as scv
    import seaborn as sns
    import pandas as pd

    output_dir = _resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _load_legacy_velocity_inputs(
        config_path=config.config_path,
        annotation_csv=config.annotation_csv,
    )
    df = inputs["df"]
    dim = inputs["dim"]
    f_net = inputs["f_net"]
    score_net = inputs["score_net"]
    device = inputs["device"]
    helpers = inputs["helpers"]

    timepoints = np.array(sorted(df["samples"].unique()))
    t = float(timepoints[np.argmin(np.abs(timepoints - float(config.target_timepoint)))])

    df_t = df[df["samples"] == t]
    data_t = df_t.iloc[:, 1 : dim + 1].values
    coords = data_t[:, :2]
    labels = df_t["Annotation"].astype(str).values if "Annotation" in df_t.columns else None

    def smooth_velocity_spatial(all_data, coord_xy, velocity_high_dim, n_neighbors: int = 30):
        adata = ad.AnnData(X=np.asarray(all_data, dtype=np.float32).copy())
        adata.obsm["X_spatial"] = np.asarray(coord_xy, dtype=np.float32)
        adata.layers["Ms"] = adata.X.copy()
        adata.layers["velocity"] = np.asarray(velocity_high_dim, dtype=np.float32).copy()
        sc.pp.neighbors(adata, n_neighbors=int(n_neighbors), use_rep="X_spatial")
        scv.tl.velocity_graph(adata, vkey="velocity", xkey="Ms")
        scv.tl.velocity_embedding(adata, basis="spatial", vkey="velocity")
        return adata.obsm["velocity_spatial"]

    vel = helpers.compute_velocity_components(
        data_t,
        float(t),
        f_net,
        score_net,
        interaction_m=int(config.interaction_m),
        interaction_threshold=int(config.interaction_threshold),
        device=device,
    )
    v_full_s = smooth_velocity_spatial(data_t, coords, vel["full"], n_neighbors=int(config.n_neighbors))
    v_inter_s = smooth_velocity_spatial(data_t, coords, vel["interaction"], n_neighbors=int(config.n_neighbors))

    dot = np.einsum("ij,ij->i", v_full_s, v_inter_s)
    denom = np.linalg.norm(v_full_s, axis=1) * np.linalg.norm(v_inter_s, axis=1)
    cos = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)

    labels_lower = np.char.lower(np.asarray(labels, dtype=str)) if labels is not None else None
    if labels_lower is not None and np.any(np.char.find(labels_lower, str(config.label_keyword).lower()) >= 0):
        mask_lab = np.char.find(labels_lower, str(config.label_keyword).lower()) >= 0
        x0, x1 = coords[mask_lab, 0].min(), coords[mask_lab, 0].max()
        y0, y1 = coords[mask_lab, 1].min(), coords[mask_lab, 1].max()
        dx, dy = x1 - x0, y1 - y0
        if dx == 0:
            dx = 1e-6
        if dy == 0:
            dy = 1e-6
        x_min, x_max = x0 - float(config.pad_ratio) * dx, x1 + float(config.pad_ratio) * dx
        y_min, y_max = y0 - float(config.pad_ratio) * dy, y1 + float(config.pad_ratio) * dy
    else:
        x_min, x_max = np.quantile(coords[:, 0], 0.72), np.quantile(coords[:, 0], 0.92)
        y_min, y_max = np.quantile(coords[:, 1], 0.40), np.quantile(coords[:, 1], 0.70)

    roi_mask = (
        (coords[:, 0] >= x_min)
        & (coords[:, 0] <= x_max)
        & (coords[:, 1] >= y_min)
        & (coords[:, 1] <= y_max)
    )

    roi_df = pd.DataFrame(
        {
            "timepoint": str(t),
            "x": coords[roi_mask, 0],
            "y": coords[roi_mask, 1],
            "cos": cos[roi_mask],
        }
    )
    if labels is not None:
        roi_df["celltype"] = labels[roi_mask]
    roi_csv = output_dir / "velocity_spatial_direction_correlation_roi_t1_scvelo_only.csv"
    roi_df.to_csv(roi_csv, index=False)

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=150)
    cmap = plt.cm.plasma
    norm = mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    ax.scatter(
        roi_df["x"],
        roi_df["y"],
        c=roi_df["cos"],
        s=22,
        linewidths=0,
        cmap=cmap,
        norm=norm,
        alpha=0.95,
    )
    ax.set_xlabel("X (spatial)")
    ax.set_ylabel("Y (spatial)")
    ax.tick_params(axis="both", labelsize=9, length=3)
    ax.set_aspect("equal")
    ax.set_title(f"ROI t={t} | correlation only")

    cax = fig.add_axes([0.86, 0.16, 0.03, 0.68])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("Cosine similarity (full vs interaction)")

    svg_path = output_dir / "velocity_spatial_direction_correlation_roi_t1_scvelo_only.svg"
    png_path = output_dir / "velocity_spatial_direction_correlation_roi_t1_scvelo_only.png"
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return AristaLegacyVelocityCorrelationResult(
        output_dir=output_dir,
        svg_path=svg_path,
        png_path=png_path,
        roi_csv=roi_csv,
        output_files=list_output_files(output_dir),
    )


def run_arista_legacy_spatiotemporal_3d(config: AristaLegacy3DConfig) -> AristaLegacy3DResult:
    _bootstrap_legacy_stack()
    from evaluation.arista_code.arista_helpers import (
        analyze_attention_by_celltype,
        save_interpolated_attention,
    )
    from evaluation.arista_code.arista_helpers_focus_anchor import (
        plot_3d_spatial_sankey_style_focus_anchor,
    )

    output_dir = _resolve_path(config.output_dir)
    ctx = _build_legacy_context(config)

    label_color_path = output_dir / "label_to_color.json"
    with open(label_color_path, "w", encoding="utf-8") as f:
        json.dump(ctx["label_to_color"], f, indent=2)

    attn_dir = output_dir / "attention"
    attn_dir.mkdir(parents=True, exist_ok=True)

    all_time_communications = {}
    for t in ctx["ts_points"]:
        key = str(t)
        adata_t = ctx["adata_dict"][key]
        attn_out = save_interpolated_attention(
            adata_t,
            time_value=float(t),
            f_net=ctx["f_net"],
            device=ctx["device"],
            out_dir=str(attn_dir),
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

    comm_path = output_dir / "arista_all_time_communications.pkl"
    with open(comm_path, "wb") as f:
        pickle.dump(all_time_communications, f)

    common_plot_kwargs = dict(
        adata_dict=ctx["adata_dict"],
        all_time_communications=all_time_communications,
        time_keys=ctx["time_keys"],
        label_to_color=ctx["label_to_color"],
        predicted_labels_list=ctx["predicted_labels_list"],
        spatial_key="spatial",
        z_spacing=float(config.z_spacing),
        reverse_time_order=False,
        intra_threshold=0.0,
        edge_focus_celltype=str(config.focus_label),
        edge_top_k=5,
        edge_top_k_focus_label=str(config.focus_label),
        ribbon_min_count=10,
        ribbon_keep_source_cumfrac=0.85,
        ribbon_focus_celltype=[str(config.focus_label)],
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
        focus_anchor_label=str(config.focus_label),
        focus_anchor_k=None,
        focus_anchor_frac=0.2,
        focus_anchor_radius=None,
        focus_anchor_min_count=None,
    )

    html_path = output_dir / "spatiotemporal_3d.html"
    fig_3d = plot_3d_spatial_sankey_style_focus_anchor(
        **common_plot_kwargs,
        width=int(11.69 * 300),
        height=int(8.27 * 300),
        out_html=str(html_path),
    )
    _apply_publication_layout(fig_3d)
    fig_3d.write_html(str(html_path))

    html_path_large = output_dir / "spatiotemporal_3d_2.html"
    fig_3d_large = plot_3d_spatial_sankey_style_focus_anchor(
        **common_plot_kwargs,
        width=1200,
        height=900,
        out_html=None,
    )
    _apply_publication_layout(fig_3d_large)
    fig_3d_large.write_html(str(html_path_large))

    svg_path = output_dir / "spatiotemporal_3d.svg"
    pdf_path = output_dir / "spatiotemporal_3d.pdf"
    png_path = output_dir / "spatiotemporal_3d.png"
    try:
        export_plotly_figure(
            fig_3d,
            svg_path=svg_path,
            pdf_path=pdf_path,
            png_path=png_path,
            vector_scale=3,
            png_scale=2,
        )
        svg_out = svg_path
        pdf_out = pdf_path
        png_out = png_path
    except Exception as exc:
        print("Static 3D export failed:", exc)
        svg_out = None
        pdf_out = None
        png_out = None

    return AristaLegacy3DResult(
        output_dir=output_dir,
        label_color_json=label_color_path,
        communications_pkl=comm_path,
        html_path=html_path,
        html_path_large=html_path_large,
        svg_path=svg_out,
        pdf_path=pdf_out,
        png_path=png_out,
        output_files=list_output_files(output_dir),
    )
