from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYTOBRIDGE_REPO = Path(
    os.environ.get("CYTOBRIDGE_REPO", str(PROJECT_ROOT.parent / "cytobridge-spatial-clone"))
).resolve()
VENDOR_ROOT = PROJECT_ROOT / "vendor"
ASSET_ROOT = PROJECT_ROOT / "assets" / "zebrafish"
RUNTIME_ROOT = PROJECT_ROOT / "assets" / "zebrafish_runtime"


def _bootstrap_module_paths() -> None:
    for path in (PROJECT_ROOT, CYTOBRIDGE_REPO, VENDOR_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_module_paths()

from CytoBridge.pl import plot_3d_spatial_sankey_style, plot_growth_per_time, plot_velocity_component  # noqa: E402
from CytoBridge.tl import (  # noqa: E402
    LRMultipanelSpec,
    build_dynamical_runtime,
    collect_top_variable_heatmaps,
    compute_timepoint_communications,
    export_ablation_gifs,
    export_ablation_panel_series,
    load_label_to_color,
    load_legacy_dynamical_model_from_dir,
    plot_spatiotemporal_3d,
    render_lr_expression_panels,
    render_lr_incoming_multipanel,
    render_top_receivers_barplot,
    run_interpolation_workflow,
    save_timepoint_snapshots,
)

from .runner import export_plotly_figure  # noqa: E402


@dataclass(frozen=True)
class ZebrafishApiConfig:
    zebrafish_root: Path | None = RUNTIME_ROOT
    config_path: Path | None = None
    data_csv: Path | None = None
    model_dir: Path | None = None
    color_h5ad: Path | None = None
    label_color_json: Path = ASSET_ROOT / "label_to_color.json"
    classifier_cache_path: Path | None = None
    lr_db_path: Path = ASSET_ROOT / "CellChatDB.ligrec.zebrafish.csv"
    annotation_key: str = "Annotation"
    output_name: str = "zebrafish_api_subfigures"


@dataclass(frozen=True)
class ZebrafishResolvedPaths:
    zebrafish_root: Path
    config_path: Path
    data_csv: Path
    model_dir: Path
    color_h5ad: Path | None
    label_color_json: Path
    classifier_cache_path: Path
    lr_db_path: Path
    results_root: Path
    figures_root: Path


@dataclass(frozen=True)
class ZebrafishApiResult:
    output_dir: Path
    trajectory_dir: Path
    gene_velocity_dir: Path
    spatial_velocity_dir: Path
    lr_dir: Path
    gene_program_dir: Path
    ablation_dir: Path
    run_summary_path: Path


def resolve_output_dir(config: ZebrafishApiConfig) -> Path:
    return PROJECT_ROOT / "results" / config.output_name


def _require_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _resolve_paths(config: ZebrafishApiConfig) -> ZebrafishResolvedPaths:
    if config.zebrafish_root is None and any(
        path is None
        for path in (
            config.config_path,
            config.data_csv,
            config.model_dir,
            config.classifier_cache_path,
        )
    ):
        raise ValueError(
            "Provide `zebrafish_root` with the standard runtime layout or explicitly pass "
            "config_path, data_csv, model_dir, and classifier_cache_path."
        )

    root = config.zebrafish_root.resolve() if config.zebrafish_root is not None else Path(".").resolve()
    results_root = root / "results"
    figures_root = results_root / "zebrafish_all_figures"

    resolved = ZebrafishResolvedPaths(
        zebrafish_root=root,
        config_path=_require_exists(
            (config.config_path or (root / "config" / "zebrafish_1111.yaml")).resolve(),
            "zebrafish config",
        ),
        data_csv=_require_exists(
            (config.data_csv or (root / "data" / "zebrafish_1108_with_annotation.csv")).resolve(),
            "zebrafish annotated CSV",
        ),
        model_dir=_require_exists(
            (config.model_dir or (results_root / "zebrafish_1111")).resolve(),
            "trained model directory",
        ),
        color_h5ad=(
            _require_exists(config.color_h5ad.resolve(), "spatial reference h5ad")
            if config.color_h5ad is not None
            else ((root / "spatial_data" / "spatial_sixtime_slice_stereoseq.h5ad").resolve()
                  if (root / "spatial_data" / "spatial_sixtime_slice_stereoseq.h5ad").exists()
                  else None)
        ),
        label_color_json=_require_exists(config.label_color_json.resolve(), "label-color mapping JSON"),
        classifier_cache_path=_require_exists(
            (
                config.classifier_cache_path
                or (figures_root / "focus_anchor" / "classifier_cache" / "mlp_classifier_c4f1314711602dff.pt")
            ).resolve(),
            "classifier cache",
        ),
        lr_db_path=_require_exists(config.lr_db_path.resolve(), "zebrafish LR database CSV"),
        results_root=results_root.resolve(),
        figures_root=figures_root.resolve(),
    )
    return resolved


def _load_df(data_csv: Path, annotation_key: str) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(data_csv)
    x_cols = sorted([c for c in df.columns if c.startswith("x")], key=lambda x: int(x[1:]))
    if not x_cols:
        raise ValueError(f"No x1..xD columns found in {data_csv}")
    dim = len(x_cols)
    df = df[["samples", *x_cols, annotation_key, "Color", "time_label"]].copy()
    df["samples"] = df["samples"].astype(float)
    return df, dim


def _load_runtime(model_dir: Path):
    loaded = load_legacy_dynamical_model_from_dir(model_dir, device="cpu")
    runtime = build_dynamical_runtime(loaded)
    return loaded, runtime


def _compute_velocity_components_reference(data: np.ndarray, time_value: float, runtime, *, device: str = "cpu") -> dict[str, np.ndarray]:
    import torch

    from CytoBridge.tl.core.interaction import cal_interaction

    data = np.asarray(data, dtype=np.float32)
    n_cells = data.shape[0]
    data_tensor = torch.tensor(data, device=device, dtype=torch.float32, requires_grad=True)
    t_tensor = torch.full((n_cells, 1), float(time_value), device=device, dtype=torch.float32)

    with torch.no_grad():
        drift = runtime.f_net.v_net(t_tensor, data_tensor)
    drift_np = drift.detach().cpu().numpy()

    lnw = torch.log(torch.ones(n_cells, 1, device=device, dtype=torch.float32) / float(n_cells))
    with torch.no_grad():
        interaction_t = cal_interaction(
            z=data_tensor.detach(),
            lnw=lnw,
            interaction_potential=runtime.f_net.interaction_net,
            m=1024,
            cutoff=1000.0,
            use_mass=True,
            t=torch.tensor([float(time_value)], dtype=torch.float32, device=device),
        )
    interaction_np = interaction_t.detach().cpu().numpy()

    score_grad = runtime.score_net.compute_gradient(t_tensor, data_tensor)
    score_np = score_grad.detach().cpu().numpy()

    return {
        "drift": drift_np,
        "interaction": interaction_np,
        "score": score_np,
        "full": drift_np + interaction_np + score_np,
    }


def _load_colors(df: pd.DataFrame, paths: ZebrafishResolvedPaths, annotation_key: str) -> dict[str, str]:
    if paths.label_color_json.exists():
        return load_label_to_color(
            df[annotation_key].astype(str).values,
            label_color_json=str(paths.label_color_json),
            color_h5ad=str(paths.color_h5ad),
            annotation_key=annotation_key,
        )
    mapping = {}
    if "Color" in df.columns:
        for _, row in df[[annotation_key, "Color"]].drop_duplicates().iterrows():
            mapping[str(row[annotation_key])] = str(row["Color"])
    if mapping:
        return mapping
    return load_label_to_color(df[annotation_key].astype(str).values, annotation_key=annotation_key)


def _time_label_slug(df: pd.DataFrame, t: float) -> str:
    labels = df.loc[df["samples"] == float(t), "time_label"].dropna().astype(str).unique().tolist()
    if labels:
        raw = labels[0].replace(".", "p").replace(" ", "_").lower()
        if raw.endswith("hpf") and not raw.endswith("_hpf"):
            raw = raw[:-3] + "_hpf"
        return raw
    return f"t{str(t).replace('.', 'p')}"


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_run_summary(path: Path, config: ZebrafishApiConfig, paths: ZebrafishResolvedPaths, result: ZebrafishApiResult) -> None:
    payload = {
        "output_dir": str(result.output_dir),
        "inputs": {
            "config_path": str(paths.config_path),
            "data_csv": str(paths.data_csv),
            "model_dir": str(paths.model_dir),
            "color_h5ad": str(paths.color_h5ad) if paths.color_h5ad is not None else None,
            "label_color_json": str(paths.label_color_json),
            "classifier_cache_path": str(paths.classifier_cache_path),
            "lr_db_path": str(paths.lr_db_path),
        },
        "annotation_key": config.annotation_key,
        "subfigure_groups": {
            "gene_velocity_streamlines": str(result.gene_velocity_dir),
            "spatial_velocity_streamlines": str(result.spatial_velocity_dir),
            "trajectory_3d_interp_growth": str(result.trajectory_dir),
            "lr_cxcl12a_cxcr4a_timecourse": str(result.lr_dir),
            "gene_program_top250_heatmaps": str(result.gene_program_dir),
            "ablation_ysl_vs_evl": str(result.ablation_dir),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_velocity_subfigures(*, df: pd.DataFrame, dim: int, runtime, colors: dict[str, str], annotation_key: str, output_dir: Path) -> tuple[Path, Path]:
    gene_dir = output_dir / "gene_velocity_streamlines"
    spatial_dir = output_dir / "spatial_velocity_streamlines"
    gene_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = [f"x{i}" for i in range(1, dim + 1)]
    keep_times = [0.0, 2.0, 4.0]
    rows_gene: list[dict[str, Any]] = []
    rows_spatial: list[dict[str, Any]] = []

    def _project_gene_velocity(expr_t: np.ndarray, coords: np.ndarray, arr: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
        import anndata as ad
        import scanpy as sc
        import scvelo as scv

        tmp_ad = ad.AnnData(X=expr_t.copy())
        tmp_ad.layers["Ms"] = expr_t.copy()
        tmp_ad.layers["velocity"] = arr.copy()
        tmp_ad.obsm["X_spatial"] = coords.copy()
        try:
            sc.pp.neighbors(tmp_ad, n_neighbors=n_neighbors, use_rep="X", transformer="sklearn")
        except TypeError:
            sc.pp.neighbors(tmp_ad, n_neighbors=n_neighbors, use_rep="X")
        scv.tl.velocity_graph(tmp_ad, vkey="velocity", xkey="Ms", n_jobs=-1)
        scv.tl.velocity_embedding(tmp_ad, basis="spatial", vkey="velocity")
        return np.nan_to_num(tmp_ad.obsm["velocity_spatial"], nan=0.0, posinf=0.0, neginf=0.0)

    for t in keep_times:
        subset = df[df["samples"] == float(t)].copy()
        data_t = subset[feature_cols].values
        coords = data_t[:, :2]
        expr_t = data_t[:, 2:]
        labels_t = subset[annotation_key].astype(str).values
        vel = _compute_velocity_components_reference(data_t, float(t), runtime, device="cpu")
        slug = _time_label_slug(df, t)

        spatial_specs = [
            ("intrinsic", vel["drift"][:, :2]),
            ("interaction", vel["interaction"][:, :2]),
            ("full", vel["full"][:, :2]),
        ]
        for component, arr in spatial_specs:
            file_name = f"spatial_{component}__{slug}.svg"
            out_path = spatial_dir / file_name
            title = f"Spatial {component.title()} (t={t:.1f})"
            plot_velocity_component(
                coords=coords,
                velocity=arr,
                labels=labels_t,
                label_to_color=colors,
                title=title,
                out_path=str(out_path),
                basis="spatial",
                show_legend=False,
            )
            rows_spatial.append({"panel_id": f"spatial_{component}_t{int(t)}", "file": file_name, "time": float(t), "component": component})

        gene_specs = [
            ("intrinsic", vel["drift"][:, 2:]),
            ("interaction", vel["interaction"][:, 2:]),
            ("full", vel["full"][:, 2:]),
        ]
        for component, arr in gene_specs:
            file_name = f"genetospatial_{component}__{slug}.svg"
            out_path = gene_dir / file_name
            projected = _project_gene_velocity(expr_t, coords, arr)
            title = f"Gene {component.title()} Projected To Spatial (t={t:.1f})"
            plot_velocity_component(
                coords=coords,
                velocity=projected,
                labels=labels_t,
                label_to_color=colors,
                title=title,
                out_path=str(out_path),
                basis="spatial",
                show_legend=False,
            )
            rows_gene.append({"panel_id": f"gene_{component}_t{int(t)}", "file": file_name, "time": float(t), "component": component})

    _write_manifest(gene_dir / "manifest.csv", rows_gene)
    _write_manifest(spatial_dir / "manifest.csv", rows_spatial)
    return gene_dir, spatial_dir


def generate_trajectory_growth_subfigures(*, config: ZebrafishApiConfig, paths: ZebrafishResolvedPaths, df: pd.DataFrame, dim: int, runtime, colors: dict[str, str], output_dir: Path) -> Path:
    traj_dir = output_dir / "trajectory_3d_interp_growth"
    traj_dir.mkdir(parents=True, exist_ok=True)

    focus_result = run_interpolation_workflow(
        df=df,
        dim=dim,
        annotation_key=config.annotation_key,
        runtime=runtime,
        device="cpu",
        output_dir=str(traj_dir / "focus_anchor"),
        requested_plot_points=None,
        interp_time_points=[0.5, 1.5, 2.5, 3.5],
        no_interp=False,
        use_real_for_observed=True,
        classifier_n_pcs=12,
        classifier_knn_neighbors=10,
        classifier_cache_path=str(paths.classifier_cache_path),
        split_sigma_scalar=0.03,
        skip_nonsplit_sde=False,
        random_seed=42,
    )
    comm = compute_timepoint_communications(
        adata_dict=focus_result.adata_dict,
        time_points=focus_result.plot_3d_ts_points,
        annotation_key=config.annotation_key,
        f_net=runtime.f_net,
        device="cpu",
        out_dir=str(traj_dir / "focus_anchor" / "attention"),
        save_dense_attention_matrix=False,
        save_pickle_path=str(traj_dir / "focus_anchor" / "communications.pkl"),
    )
    fig3d = plot_spatiotemporal_3d(
        plot_fn=plot_3d_spatial_sankey_style,
        adata_dict=focus_result.adata_dict,
        all_time_communications=comm,
        time_keys=focus_result.plot_3d_time_keys,
        plot_time_points=focus_result.plot_3d_ts_points,
        ts_points=focus_result.ts_points,
        observed_time_points=focus_result.observed_time_points,
        interp_points=focus_result.interp_points,
        annotation_key=config.annotation_key,
        label_to_color=colors,
        out_html=str(traj_dir / "spatiotemporal_3d.html"),
        predicted_labels_list=focus_result.predicted_labels_split,
        background_color="white",
        font_color="black",
        point_size=1.0,
        point_alpha=0.6,
        width=1400,
        height=1000,
        show_title=False,
        show_time_axis=True,
    )
    try:
        export_plotly_figure(fig3d, png_path=traj_dir / "spatiotemporal_3d_preview.png", png_scale=2.0)
    except Exception as exc:
        (traj_dir / "spatiotemporal_3d_preview.error.txt").write_text(str(exc), encoding="utf-8")

    piecewise_result = run_interpolation_workflow(
        df=df,
        dim=dim,
        annotation_key=config.annotation_key,
        runtime=runtime,
        device="cpu",
        output_dir=str(traj_dir / "focus_anchor_piecewise_warp"),
        requested_plot_points=[0.0, 2.0, 4.0],
        interp_time_points=[0.5, 1.5, 2.5, 3.5],
        no_interp=False,
        use_real_for_observed=True,
        classifier_n_pcs=12,
        classifier_knn_neighbors=10,
        classifier_cache_path=str(paths.classifier_cache_path),
        split_sigma_scalar=0.03,
        skip_nonsplit_sde=True,
        spatial_warp_to_observed_piecewise=True,
        spatial_warp_k=8,
        spatial_warp_eps=1e-6,
        slice_max_cells_per_timepoint=4000,
        random_seed=42,
    )
    save_timepoint_snapshots(
        adata_dict=piecewise_result.adata_dict,
        time_keys=piecewise_result.time_keys,
        annotation_key=config.annotation_key,
        label_to_color=colors,
        snapshot_dir=str(traj_dir),
        background_color="white",
        font_color="black",
        snapshot_point_size=3.0,
        snapshot_alpha=0.95,
        mosaic_cols=3,
        mosaic_cell_size=2.8,
        mosaic_show_title=False,
        save_pdf=True,
    )
    plot_growth_per_time(
        df=df,
        dim=dim,
        model=runtime.model,
        out_dir=str(traj_dir),
        device="cpu",
        point_size=2.4,
        point_alpha=1.0,
    )

    rows: list[dict[str, Any]] = [{"panel_id": "spatiotemporal_3d", "file": "spatiotemporal_3d_preview.png", "kind": "3d_preview"}]
    for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        rows.append({"panel_id": f"interp_t{t:.1f}_svg", "file": f"time_{t:.1f}.svg", "kind": "interpolation"})
        rows.append({"panel_id": f"interp_t{t:.1f}_pdf", "file": f"time_{t:.1f}.pdf", "kind": "interpolation"})
    for idx in range(5):
        rows.append({"panel_id": f"growth_t{idx}", "file": f"growth_t{idx}.pdf", "kind": "growth"})
    _write_manifest(traj_dir / "manifest.csv", rows)
    return traj_dir


def generate_lr_subfigures(*, paths: ZebrafishResolvedPaths, output_dir: Path) -> Path:
    lr_dir = output_dir / "lr_cxcl12a_cxcr4a_timecourse"
    lr_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = paths.figures_root / "lr_scores"

    ligand_pdf, receptor_pdf = render_lr_expression_panels(
        scores_pkl=scores_dir / "lr_scores_1.5.pkl",
        h5ad_path=paths.results_root / "zebrafish_1111" / "adata_t1p500.h5ad",
        lr_pair="cxcl12a_cxcr4a",
        lr_db_path=paths.lr_db_path,
        out_dir=lr_dir,
    )
    render_top_receivers_barplot(
        type_scores_csv=paths.figures_root / "lr_single_timepoint" / "cxcl12a_cxcr4a_t1.5_type_scores.csv",
        out_prefix=lr_dir / "lr_top_receivers",
    )
    labels = ["t0", "t0.5", "t1", "t1.5", "t2", "t2.5", "t3", "t3.5", "t4"]
    time_tokens = ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0"]
    score_tokens = ["5.25hpf", "0.5", "10hpf", "1.5", "12hpf", "2.5", "18hpf", "3.5", "24hpf"]

    def _adata_name(tok: str) -> str:
        return f"adata_t{float(tok):0.3f}".replace(".", "p") + ".h5ad"

    specs = [
        LRMultipanelSpec(
            label=labels[i],
            scores_pkl=scores_dir / f"lr_scores_{score_tokens[i]}.pkl",
            h5ad_path=paths.results_root / "zebrafish_1111" / _adata_name(time_tokens[i]),
            annotation_col="Annotation",
            no_filter_time=True,
            invert_y=False,
        )
        for i in range(len(labels))
    ]
    multipanel_pdf = render_lr_incoming_multipanel(
        specs=specs,
        lr_pair="cxcl12a_cxcr4a",
        out_prefix=lr_dir / "multipanel_3x3_modelspace",
    )
    manifest = [
        {"panel_id": "ligand", "file": ligand_pdf.name, "kind": "lr_expression"},
        {"panel_id": "receptor", "file": receptor_pdf.name, "kind": "lr_expression"},
        {"panel_id": "top_receivers", "file": "lr_top_receivers.png", "kind": "lr_summary"},
        {"panel_id": "incoming_multipanel_3x3", "file": multipanel_pdf.name, "kind": "lr_hotspot"},
    ]
    _write_manifest(lr_dir / "manifest.csv", manifest)
    return lr_dir


def generate_gene_program_subfigures(*, paths: ZebrafishResolvedPaths, output_dir: Path) -> Path:
    gp_dir = output_dir / "gene_program_top250_heatmaps"
    gp_dir.mkdir(parents=True, exist_ok=True)
    result = collect_top_variable_heatmaps(
        figures_dir=paths.figures_root / "gene_pathway" / "figures",
        out_dir=gp_dir,
    )
    _write_manifest(
        gp_dir / "manifest.csv",
        [
            {"panel_id": "mean", "file": result.mean_pdf.name, "kind": "gene_program"},
            {"panel_id": "zscore", "file": result.zscore_pdf.name, "kind": "gene_program"},
        ],
    )
    return gp_dir


def generate_ablation_subfigures(*, paths: ZebrafishResolvedPaths, output_dir: Path) -> Path:
    abl_dir = output_dir / "ablation_ysl_vs_evl"
    abl_dir.mkdir(parents=True, exist_ok=True)
    ysl_frames = paths.figures_root / "zebrafish_ablation_yolk" / "trajectory" / "frames"
    evl_frames = paths.figures_root / "zebrafish_ablation_evl_blackbox" / "trajectory" / "frames"
    result = export_ablation_panel_series(
        baseline_frames_dir=ysl_frames,
        comparison_frame_dirs={
            "ysl_ko": ysl_frames,
            "evl_ko": evl_frames,
        },
        frame_ids={"5.25hpf": "000", "10hpf": "020", "12hpf": "040", "18hpf": "060", "24hpf": "080"},
        out_dir=abl_dir,
    )
    gif_result = export_ablation_gifs(
        frame_dirs={
            "ysl_vs_baseline": ysl_frames,
            "evl_vs_baseline": evl_frames,
        },
        out_dir=abl_dir,
        frame_step=2,
        resize_factor=0.5,
        duration_ms=120,
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(result.files):
        stem = path.stem
        kind, time = stem.split("__", 1)
        rows.append({"panel_id": stem, "file": path.name, "kind": kind, "time": time})
    for path in sorted(gif_result.gifs):
        rows.append({"panel_id": path.stem, "file": path.name, "kind": "gif"})
    _write_manifest(abl_dir / "manifest.csv", rows)
    return abl_dir


def run_zebrafish_api_subfigures(config: ZebrafishApiConfig | None = None) -> ZebrafishApiResult:
    config = config or ZebrafishApiConfig()
    paths = _resolve_paths(config)
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, dim = _load_df(paths.data_csv, config.annotation_key)
    _, runtime = _load_runtime(paths.model_dir)
    colors = _load_colors(df, paths, config.annotation_key)

    gene_dir, spatial_dir = generate_velocity_subfigures(
        df=df,
        dim=dim,
        runtime=runtime,
        colors=colors,
        annotation_key=config.annotation_key,
        output_dir=output_dir,
    )
    trajectory_dir = generate_trajectory_growth_subfigures(
        config=config,
        paths=paths,
        df=df,
        dim=dim,
        runtime=runtime,
        colors=colors,
        output_dir=output_dir,
    )
    lr_dir = generate_lr_subfigures(paths=paths, output_dir=output_dir)
    gene_program_dir = generate_gene_program_subfigures(paths=paths, output_dir=output_dir)
    ablation_dir = generate_ablation_subfigures(paths=paths, output_dir=output_dir)
    result = ZebrafishApiResult(
        output_dir=output_dir,
        trajectory_dir=trajectory_dir,
        gene_velocity_dir=gene_dir,
        spatial_velocity_dir=spatial_dir,
        lr_dir=lr_dir,
        gene_program_dir=gene_program_dir,
        ablation_dir=ablation_dir,
        run_summary_path=output_dir / "run_summary.json",
    )
    _write_run_summary(result.run_summary_path, config, paths, result)
    return result
