"""Package-backed ARISTA downstream workflows.

This module deliberately contains no model architecture or numerical solver
implementation.  Legacy ARISTA checkpoints are loaded through the compatibility
API shipped by :mod:`CytoBridge`; all inference and plotting is then performed
through public CytoBridge entrypoints.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

from CytoBridge.pl import (
    plot_celltype_composition,
    plot_growth_interaction_bubble,
    plot_growth_timepoint_grid,
    plot_spatial_component_direction_correlation_roi_from_adata,
    plot_temporal_gene_heatmap,
    plot_temporal_pattern_prototypes,
    plot_temporal_profile_small_multiples,
    plot_velocity_component,
)
from CytoBridge.tl import (
    adata_to_aligned_dataframe,
    build_dynamical_runtime,
    compute_timepoint_communications,
    evaluate_growth_by_timepoint,
    compute_velocity_components,
    compute_velocity_components_from_adata,
    infer_feature_columns,
    infer_time_key,
    load_dynamical_model_from_dir,
    load_label_to_color,
    load_legacy_dynamical_model_from_dir,
    load_pca_reconstruction_spec,
    plot_lineage_sankey,
    plot_spatiotemporal_3d,
    project_communication_to_lr_timecourses,
    run_interpolation_workflow,
    save_timepoint_snapshots,
    set_global_random_seed,
    summarize_growth_interaction_by_celltype,
    summarize_label_composition,
    summarize_temporal_gene_patterns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class AristaApiPaths:
    data_csv: Path
    model_dir: Path
    edge_predictor_root: Path
    label_color_json: Path


@dataclass(frozen=True)
class AristaVelocityConfig:
    """Configuration for the API-backed ARISTA velocity-stream example."""

    output_name: str = "arista_velocity_t1_streams_api"
    output_dir: str | Path | None = None
    workspace_root: str | Path | None = None
    data_csv: str | Path | None = None
    model_dir: str | Path | None = None
    edge_predictor_root: str | Path | None = None
    label_color_json: str | Path | None = None
    target_timepoint: float = 1.0
    annotation_key: str = "Annotation"
    max_cells: int | None = None
    random_seed: int = 42
    device: str = "cpu"
    interaction_m: int = 1024
    interaction_threshold: float = 1000.0
    density: float = 2.0
    show_legend: bool = True


@dataclass(frozen=True)
class AristaVelocityResult:
    output_dir: Path
    manifest_path: Path
    component_summary_csv: Path
    component_arrays_npz: Path
    figure_paths: dict[str, Path]
    n_cells: int
    dim: int


@dataclass(frozen=True)
class AristaSpatiotemporalConfig:
    """Dataset definitions for package-backed ARISTA lineage/communication/3D."""

    output_name: str = "arista_spatiotemporal_api"
    output_dir: str | Path | None = None
    aligned_h5ad: str | Path | None = None
    data_csv: str | Path | None = None
    model_dir: str | Path | None = None
    model_format: str = "legacy"
    edge_predictor_root: str | Path | None = None
    label_color_json: str | Path | None = None
    annotation_key: str = "Annotation"
    time_key: str | None = None
    obsm_key: str = "X_latent"
    spatial_key: str = "spatial_aligned"
    concat_spatial: bool = True
    time_points: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)
    interp_time_points: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)
    plot_3d_time_points: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    n_samples: int = 7668
    skip_nonsplit_sde: bool = False
    sde_dt: float = 0.05
    split_sde_dt: float = 0.01
    split_sigma: float = 0.03
    split_growth_alpha: float = 1.0
    use_real_for_observed: bool = True
    spatial_warp_to_observed_piecewise: bool = True
    spatial_warp_k: int = 8
    spatial_warp_eps: float = 1e-6
    classifier_cache_path: str | Path | None = None
    classifier_cache_dir: str | Path | None = None
    classifier_epochs: int = 1000
    classifier_hidden_size: int = 128
    classifier_best_metric: str = "bacc"
    classifier_train_on_full_data: bool = False
    classifier_knn_neighbors: int = 1
    random_seed: int = 42
    device: str = "cpu"
    run_communication: bool = True
    run_3d: bool = True


@dataclass(frozen=True)
class AristaSpatiotemporalResult:
    output_dir: Path
    manifest_path: Path
    snapshots_dir: Path
    lineage_html: Path
    composition_csv: Path
    composition_figure: Path
    growth_csv: Path
    growth_figure: Path
    communications_pickle: Path | None
    spatiotemporal_html: Path | None
    classifier_cache_dir: Path
    time_points: tuple[float, ...]
    generated_time_points: tuple[float, ...]


@dataclass(frozen=True)
class AristaGrowthInteractionResult:
    output_dir: Path
    manifest_path: Path
    raw_csv: Path
    grouped_csv: Path
    bubble_path: Path


@dataclass(frozen=True)
class AristaDirectionCorrelationResult:
    output_dir: Path
    manifest_path: Path
    roi_csv: Path
    figure_path: Path
    spatial_velocity_figure: Path
    pca_velocity_figure: Path
    n_roi_cells: int


@dataclass(frozen=True)
class AristaTemporalProgramsResult:
    output_dir: Path
    manifest_path: Path
    tables_dir: Path
    figures_dir: Path
    communication_pickle: Path
    gene_heatmap: Path
    gene_pattern_figure: Path
    lr_prototype_figure: Path
    lr_profiles_figure: Path


def _require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _require_dir(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def resolve_arista_api_paths(config: AristaVelocityConfig) -> AristaApiPaths:
    """Resolve portable reviewer assets or an explicitly supplied workspace."""

    if config.workspace_root is None:
        data_root = PROJECT_ROOT / "data" / "arista"
        asset_root = PROJECT_ROOT / "assets" / "arista"
        defaults = {
            "data_csv": data_root / "arista_1108_with_annotation.csv",
            "model_dir": asset_root / "model",
            "edge_predictor_root": asset_root / "edge_classifier",
            "label_color_json": asset_root / "label_to_color.json",
        }
    else:
        workspace = Path(config.workspace_root).expanduser().resolve()
        defaults = {
            "data_csv": workspace / "data" / "arista_1108_with_annotation.csv",
            "model_dir": workspace / "results" / "arista_1110",
            "edge_predictor_root": workspace / "edge_classifier",
            # Plot colors are reviewer metadata rather than a training artifact.
            "label_color_json": PROJECT_ROOT
            / "assets"
            / "arista"
            / "label_to_color.json",
        }

    return AristaApiPaths(
        data_csv=_require_file(
            Path(config.data_csv)
            if config.data_csv is not None
            else defaults["data_csv"],
            "ARISTA annotated model-input CSV",
        ),
        model_dir=_require_dir(
            Path(config.model_dir)
            if config.model_dir is not None
            else defaults["model_dir"],
            "ARISTA trained model directory",
        ),
        edge_predictor_root=_require_dir(
            Path(config.edge_predictor_root)
            if config.edge_predictor_root is not None
            else defaults["edge_predictor_root"],
            "ARISTA edge-predictor directory",
        ),
        label_color_json=_require_file(
            Path(config.label_color_json)
            if config.label_color_json is not None
            else defaults["label_color_json"],
            "ARISTA label-color JSON",
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loaded_model_manifest(loaded) -> dict[str, Any]:
    weight_path = getattr(loaded, "weight_path", None)
    score_path = getattr(loaded, "score_path", None)
    return {
        "class": type(loaded.model).__name__,
        "weight_stage": loaded.weight_stage,
        "weight_checkpoint": (
            None if weight_path is None else str(Path(weight_path).resolve())
        ),
        "weight_checkpoint_sha256": (
            None if weight_path is None else _sha256(Path(weight_path))
        ),
        "score_stage": loaded.score_stage,
        "score_checkpoint": (
            None if score_path is None else str(Path(score_path).resolve())
        ),
        "score_checkpoint_sha256": (
            None if score_path is None else _sha256(Path(score_path))
        ),
        "components": list(getattr(loaded.model, "components", [])),
    }


def _checkpoint_hashes(paths: AristaApiPaths) -> dict[str, str]:
    files = {
        "data_csv": paths.data_csv,
        "params_yml": paths.model_dir / "params.yml",
        "model_final": paths.model_dir / "model_final",
        "score_model": paths.model_dir / "score_model",
        "edge_predictor": paths.edge_predictor_root / "arista.pt",
        "label_color_json": paths.label_color_json,
    }
    return {name: _sha256(_require_file(path, name)) for name, path in files.items()}


def assert_package_only_runtime() -> None:
    """Fail if the external vendored DeepRUOT stack entered the process."""

    forbidden = sorted(
        name
        for name in sys.modules
        if name == "DeepRUOT"
        or name.startswith("DeepRUOT.")
        or name == "vendor.legacy_arista_stack"
        or name.startswith("vendor.legacy_arista_stack.")
    )
    if forbidden:
        raise RuntimeError(
            f"Legacy vendor modules were imported unexpectedly: {forbidden}"
        )


def load_arista_api_context(config: AristaVelocityConfig) -> dict[str, Any]:
    """Load the annotated model inputs and trained checkpoint via CytoBridge."""

    paths = resolve_arista_api_paths(config)
    df = pd.read_csv(paths.data_csv)
    if config.annotation_key not in df.columns:
        raise KeyError(
            f"Missing annotation column '{config.annotation_key}' in {paths.data_csv}"
        )
    if "samples" not in df.columns:
        raise KeyError(f"Missing samples column in {paths.data_csv}")

    feature_cols = list(infer_feature_columns(df))
    loaded = load_legacy_dynamical_model_from_dir(
        paths.model_dir,
        device=config.device,
        edge_predictor_root=paths.edge_predictor_root,
    )
    runtime = build_dynamical_runtime(loaded)
    expected_dim = int(loaded.config["legacy"]["data"]["dim"])
    if len(feature_cols) != expected_dim:
        raise ValueError(
            f"Model expects {expected_dim} features, but {paths.data_csv} contains "
            f"{len(feature_cols)} model feature columns."
        )

    label_to_color = load_label_to_color(
        df[config.annotation_key].astype(str).to_numpy(),
        label_color_json=str(paths.label_color_json),
        annotation_key=config.annotation_key,
    )
    assert_package_only_runtime()
    return {
        "paths": paths,
        "df": df,
        "feature_cols": feature_cols,
        "loaded": loaded,
        "runtime": runtime,
        "label_to_color": label_to_color,
        "dim": expected_dim,
    }


def _select_timepoint_rows(
    df: pd.DataFrame,
    *,
    timepoint: float,
    max_cells: int | None,
    random_seed: int,
) -> pd.DataFrame:
    times = pd.to_numeric(df["samples"], errors="raise").to_numpy(dtype=float)
    selected = df.loc[np.isclose(times, float(timepoint), rtol=0.0, atol=1e-9)].copy()
    if selected.empty:
        available = sorted(pd.unique(times).tolist())
        raise ValueError(
            f"No rows found at timepoint {timepoint}; available={available}"
        )
    if max_cells is not None:
        max_cells = int(max_cells)
        if max_cells <= 0:
            raise ValueError(f"max_cells must be positive or None, got {max_cells}")
        if len(selected) > max_cells:
            selected = selected.sample(
                n=max_cells, random_state=int(random_seed)
            ).sort_index()
    return selected


def _component_summary(components: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, values in components.items():
        values = np.asarray(values, dtype=np.float32)
        norm = np.linalg.norm(values, axis=1)
        rows.append(
            {
                "component": name,
                "n_cells": int(values.shape[0]),
                "dim": int(values.shape[1]),
                "all_finite": bool(np.isfinite(values).all()),
                "mean_norm": float(norm.mean()),
                "median_norm": float(np.median(norm)),
                "max_norm": float(norm.max()),
            }
        )
    return pd.DataFrame(rows)


def run_arista_velocity_t1_streams(
    config: AristaVelocityConfig | None = None,
) -> AristaVelocityResult:
    """Reproduce the six ARISTA t1 velocity panels through CytoBridge APIs."""

    config = config or AristaVelocityConfig()
    set_global_random_seed(config.random_seed)
    context = load_arista_api_context(config)
    paths: AristaApiPaths = context["paths"]
    df_t = _select_timepoint_rows(
        context["df"],
        timepoint=float(config.target_timepoint),
        max_cells=config.max_cells,
        random_seed=int(config.random_seed),
    )
    feature_cols = context["feature_cols"]
    data = df_t[feature_cols].to_numpy(dtype=np.float32)
    coords = data[:, :2]
    labels = df_t[config.annotation_key].astype(str).to_numpy()

    components = compute_velocity_components(
        data=data,
        time_value=float(config.target_timepoint),
        model=context["loaded"].model,
        interaction_m=int(config.interaction_m),
        interaction_threshold=float(config.interaction_threshold),
        device=config.device,
    )
    assert_package_only_runtime()

    full_identity_error = float(
        np.max(
            np.abs(
                components["full"]
                - components["drift"]
                - components["interaction"]
                - components["score"]
            )
        )
    )
    if not all(np.isfinite(values).all() for values in components.values()):
        raise FloatingPointError(
            "Non-finite values found in an ARISTA velocity component."
        )

    panel_components = {
        "spatial_intrinsic": components["drift"][:, :2],
        "spatial_interaction": components["interaction"][:, :2],
        "spatial_full": components["full"][:, :2],
        # Preserve the published notebook's definition: show the first two
        # non-spatial model dimensions over the same tissue coordinates.
        "gene_intrinsic": components["drift"][:, 2:4],
        "gene_interaction": components["interaction"][:, 2:4],
        "gene_full": components["full"][:, 2:4],
    }

    output_dir = (
        Path(config.output_dir).expanduser().resolve()
        if config.output_dir is not None
        else (RESULTS_ROOT / config.output_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    time_tag = f"{float(config.target_timepoint):g}".replace(".", "p")
    figure_paths: dict[str, Path] = {}
    plot_fallbacks: dict[str, str | None] = {}
    for name, velocity in panel_components.items():
        out_path = output_dir / f"velocity_scvelo_{name}_t{time_tag}.svg"
        panel_adata = plot_velocity_component(
            coords=coords,
            velocity=velocity,
            labels=labels,
            label_to_color=context["label_to_color"],
            title=f"{name.replace('_', ' ').title()} (t={config.target_timepoint:g})",
            out_path=str(out_path),
            basis="spatial",
            density=float(config.density),
            show_legend=bool(config.show_legend),
        )
        figure_paths[name] = out_path
        plot_fallbacks[name] = panel_adata.uns.get("velocity_plot_fallback")

    component_summary_csv = output_dir / "velocity_component_summary.csv"
    _component_summary(components).to_csv(component_summary_csv, index=False)

    component_arrays_npz = output_dir / "velocity_components.npz"
    np.savez_compressed(
        component_arrays_npz,
        row_index=df_t.index.to_numpy(),
        labels=labels,
        features=data,
        drift=components["drift"],
        interaction=components["interaction"],
        score=components["score"],
        full=components["full"],
    )

    manifest = {
        "workflow": "arista_velocity_t1_streams_api",
        "runtime_contract": "installed CytoBridge public API; no vendor/legacy_arista_stack imports",
        "cytobridge_version": importlib.metadata.version("CytoBridge"),
        "cytobridge_module": str(Path(sys.modules["CytoBridge"].__file__).resolve()),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "resolved_paths": {key: str(value) for key, value in asdict(paths).items()},
        "input_hashes_sha256": _checkpoint_hashes(paths),
        "model": _loaded_model_manifest(context["loaded"]),
        "n_cells": int(data.shape[0]),
        "dim": int(data.shape[1]),
        "full_identity_max_error": full_identity_error,
        "plot_fallbacks": plot_fallbacks,
        "figures": {key: str(value) for key, value in figure_paths.items()},
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    return AristaVelocityResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        component_summary_csv=component_summary_csv,
        component_arrays_npz=component_arrays_npz,
        figure_paths=figure_paths,
        n_cells=int(data.shape[0]),
        dim=int(data.shape[1]),
    )


def load_arista_spatiotemporal_context(
    config: AristaSpatiotemporalConfig | None = None,
) -> dict[str, Any]:
    """Load either a current aligned H5AD or the portable legacy 52D table.

    The returned runtime always comes from public :mod:`CytoBridge` loaders.
    The CSV-to-AnnData conversion below is only a dataset adapter; it does not
    implement a model, solver, classifier, interaction calculation, or plot.
    """
    import anndata as ad

    config = config or AristaSpatiotemporalConfig()
    model_format = str(config.model_format).strip().lower()
    if model_format not in {"current", "legacy"}:
        raise ValueError("model_format must be 'current' or 'legacy'.")

    portable_asset_root = PROJECT_ROOT / "assets" / "arista"
    data_csv = (
        Path(config.data_csv)
        if config.data_csv is not None
        else PROJECT_ROOT / "data" / "arista" / "arista_1108_with_annotation.csv"
    )
    model_dir = (
        Path(config.model_dir)
        if config.model_dir is not None
        else portable_asset_root / "model"
    )
    edge_predictor_root = (
        Path(config.edge_predictor_root)
        if config.edge_predictor_root is not None
        else portable_asset_root / "edge_classifier"
    )
    label_color_json = (
        Path(config.label_color_json)
        if config.label_color_json is not None
        else portable_asset_root / "label_to_color.json"
    )

    if config.aligned_h5ad is not None:
        aligned_h5ad = _require_file(Path(config.aligned_h5ad), "ARISTA aligned H5AD")
        adata = ad.read_h5ad(aligned_h5ad)
        df, resolved_time_key = adata_to_aligned_dataframe(
            adata,
            time_key=config.time_key,
            obsm_key=config.obsm_key,
            spatial_key=config.spatial_key,
            concat_spatial=config.concat_spatial,
            annotation_key=config.annotation_key,
        )
    else:
        aligned_h5ad = None
        data_csv = _require_file(data_csv, "ARISTA annotated model-input CSV")
        df = pd.read_csv(data_csv)
        if config.annotation_key not in df.columns or "samples" not in df.columns:
            raise KeyError(
                f"{data_csv} must contain 'samples' and '{config.annotation_key}'."
            )
        feature_cols_csv = list(infer_feature_columns(df))
        if len(feature_cols_csv) < 3:
            raise ValueError(
                "ARISTA model-input CSV must contain two spatial and latent columns."
            )
        adata = ad.AnnData(
            X=df[feature_cols_csv[2:]].to_numpy(dtype=np.float32),
            obs=df[["samples", config.annotation_key]].copy(),
        )
        adata.obs_names = pd.Index([f"arista_{idx}" for idx in range(adata.n_obs)])
        adata.obsm[config.spatial_key] = df[feature_cols_csv[:2]].to_numpy(
            dtype=np.float32
        )
        adata.obsm[config.obsm_key] = df[feature_cols_csv[2:]].to_numpy(
            dtype=np.float32
        )
        resolved_time_key = "samples"

    if config.annotation_key not in adata.obs.columns:
        raise KeyError(f"adata.obs is missing '{config.annotation_key}'.")
    feature_cols = list(infer_feature_columns(df))
    dim = len(feature_cols)
    model_dir = _require_dir(model_dir, "ARISTA model directory")
    if model_format == "current":
        loaded = load_dynamical_model_from_dir(
            model_dir,
            dim=dim,
            device=config.device,
        )
    else:
        edge_predictor_root = _require_dir(
            edge_predictor_root,
            "ARISTA edge-predictor directory",
        )
        loaded = load_legacy_dynamical_model_from_dir(
            model_dir,
            device=config.device,
            edge_predictor_root=edge_predictor_root,
        )
    runtime = build_dynamical_runtime(loaded)

    label_color_json = _require_file(label_color_json, "ARISTA label-color JSON")
    label_to_color = load_label_to_color(
        adata.obs[config.annotation_key].astype(str).to_numpy(),
        label_color_json=str(label_color_json),
        color_h5ad=str(aligned_h5ad) if aligned_h5ad is not None else None,
        annotation_key=config.annotation_key,
    )

    if config.classifier_cache_path is not None:
        classifier_cache_path = _require_file(
            Path(config.classifier_cache_path),
            "ARISTA classifier cache",
        )
    elif model_format == "legacy" and config.aligned_h5ad is None:
        candidates = sorted(
            (portable_asset_root / "classifier_cache").glob("classifier_resmlp_*.pt")
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Expected one portable ARISTA classifier cache; "
                f"found {[str(path) for path in candidates]}."
            )
        classifier_cache_path = candidates[0].resolve()
    else:
        classifier_cache_path = None

    assert_package_only_runtime()
    return {
        "adata": adata,
        "df": df,
        "feature_cols": feature_cols,
        "dim": dim,
        "resolved_time_key": resolved_time_key,
        "loaded": loaded,
        "runtime": runtime,
        "label_to_color": label_to_color,
        "classifier_cache_path": classifier_cache_path,
        "aligned_h5ad": aligned_h5ad,
        "data_csv": data_csv.resolve() if Path(data_csv).exists() else None,
        "model_dir": model_dir.resolve(),
        "edge_predictor_root": edge_predictor_root.resolve(),
        "label_color_json": label_color_json.resolve(),
    }


def _export_plotly_bundle(fig, stem: Path) -> dict[str, str]:
    from .runner import export_plotly_figure

    paths = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
    }
    try:
        export_plotly_figure(
            fig,
            svg_path=paths["svg"],
            pdf_path=paths["pdf"],
            png_path=paths["png"],
            vector_scale=3,
            png_scale=2,
        )
        return {key: str(value) for key, value in paths.items()}
    except Exception as exc:
        return {"error": str(exc)}


def _run_arista_interpolation(
    *,
    config: AristaSpatiotemporalConfig,
    context: dict[str, Any],
    output_dir: Path,
    classifier_cache_dir: Path,
):
    requested_points = sorted(
        set(map(float, config.time_points)) | set(map(float, config.interp_time_points))
    )
    return run_interpolation_workflow(
        df=context["df"],
        dim=context["dim"],
        annotation_key=config.annotation_key,
        runtime=context["runtime"],
        device=config.device,
        output_dir=str(output_dir),
        requested_plot_points=requested_points,
        interp_time_points=config.interp_time_points,
        max_observed_timepoints=len(config.time_points),
        use_real_for_observed=bool(config.use_real_for_observed),
        classifier_cache_path=(
            str(context["classifier_cache_path"])
            if context["classifier_cache_path"] is not None
            else None
        ),
        classifier_cache_dir=str(classifier_cache_dir),
        classifier_adata=(
            context["adata"] if context["classifier_cache_path"] is None else None
        ),
        classifier_time_key=context["resolved_time_key"],
        classifier_obsm_key=config.obsm_key,
        classifier_spatial_key=config.spatial_key,
        classifier_concat_spatial=config.concat_spatial,
        classifier_epochs=int(config.classifier_epochs),
        classifier_hidden_size=int(config.classifier_hidden_size),
        classifier_best_metric=config.classifier_best_metric,
        classifier_train_on_full_data=bool(config.classifier_train_on_full_data),
        classifier_knn_neighbors=int(config.classifier_knn_neighbors),
        sde_n_samples=int(config.n_samples),
        skip_nonsplit_sde=bool(config.skip_nonsplit_sde),
        sde_dt=float(config.sde_dt),
        split_sde_dt=float(config.split_sde_dt),
        split_sigma_scalar=float(config.split_sigma),
        split_growth_alpha=float(config.split_growth_alpha),
        spatial_warp_to_observed_piecewise=bool(
            config.spatial_warp_to_observed_piecewise
        ),
        spatial_warp_k=int(config.spatial_warp_k),
        spatial_warp_eps=float(config.spatial_warp_eps),
        random_seed=int(config.random_seed),
    )


def run_arista_spatiotemporal_api(
    config: AristaSpatiotemporalConfig | None = None,
) -> AristaSpatiotemporalResult:
    """Run ARISTA interpolation, lineage, communication, and 3D package APIs."""
    config = config or AristaSpatiotemporalConfig()
    set_global_random_seed(config.random_seed)
    context = load_arista_spatiotemporal_context(config)
    output_dir = (
        Path(config.output_dir).expanduser().resolve()
        if config.output_dir is not None
        else (RESULTS_ROOT / config.output_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_cache_dir = (
        Path(config.classifier_cache_dir).expanduser().resolve()
        if config.classifier_cache_dir is not None
        else output_dir / "classifier_cache"
    )

    interpolation = _run_arista_interpolation(
        config=config,
        context=context,
        output_dir=output_dir,
        classifier_cache_dir=classifier_cache_dir,
    )

    observed_variants = {}
    if (
        interpolation.sde_points_split is not None
        and interpolation.predicted_labels_split is not None
    ):
        for time_value in config.time_points:
            idx = interpolation.ts_points.index(float(time_value))
            observed_df = context["df"][
                np.isclose(context["df"]["samples"], float(time_value))
            ]
            observed_variants[float(time_value)] = {
                "Observed": (
                    observed_df[context["feature_cols"]].to_numpy(dtype=np.float32)[
                        :, :2
                    ],
                    observed_df[config.annotation_key].astype(str).to_numpy(),
                ),
                "Generated": (
                    np.asarray(interpolation.sde_points_split[idx], dtype=np.float32)[
                        :, :2
                    ],
                    np.asarray(interpolation.predicted_labels_split[idx]).astype(str),
                ),
            }

    snapshots_dir = output_dir / "snapshots"
    save_timepoint_snapshots(
        adata_dict=interpolation.adata_dict,
        time_keys=interpolation.time_keys,
        annotation_key=config.annotation_key,
        label_to_color=context["label_to_color"],
        observed_variants=observed_variants or None,
        snapshot_dir=str(snapshots_dir),
        background_color=None,
        font_color="#1a1a1a",
        snapshot_point_size=2.5,
        snapshot_alpha=0.9,
        mosaic_cols=4,
        mosaic_cell_size=2.2,
        mosaic_show_title=True,
        save_pdf=True,
    )

    lineage_labels = (
        interpolation.predicted_labels_list
        if interpolation.predicted_labels_list is not None
        else interpolation.predicted_labels_split
    )
    if lineage_labels is None:
        raise RuntimeError("Interpolation did not produce trajectory labels.")
    lineage_html = output_dir / "lineage_sankey.html"
    lineage_fig = plot_lineage_sankey(
        predicted_labels_list=lineage_labels,
        time_keys=interpolation.time_keys,
        label_to_color=context["label_to_color"],
        out_html=str(lineage_html),
        keep_source_cumfrac=0.8,
        style="nature-methods",
        title="Cell Fate Transitions",
    )
    static_exports = {
        "lineage": _export_plotly_bundle(lineage_fig, output_dir / "lineage_sankey")
    }
    composition = summarize_label_composition(
        lineage_labels,
        interpolation.time_keys,
    )
    composition_csv = output_dir / "celltype_composition.csv"
    composition_figure = output_dir / "celltype_composition.svg"
    composition.to_csv(composition_csv, index=False)
    plot_celltype_composition(
        composition,
        out_path=composition_figure,
        label_to_color=context["label_to_color"],
        title="Cell-type composition across observed and interpolated stages",
    )
    growth = evaluate_growth_by_timepoint(
        interpolation.adata_dict,
        context["loaded"].model,
        time_points=interpolation.ts_points,
        time_keys=interpolation.time_keys,
        annotation_key=config.annotation_key,
        spatial_key="spatial",
        device=config.device,
    )
    growth_csv = output_dir / "growth_dense_time_grid.csv"
    growth_figure = output_dir / "growth_dense_time_grid.svg"
    growth.to_csv(growth_csv, index=False)
    observed_set = set(map(float, interpolation.observed_time_points))
    plot_growth_timepoint_grid(
        interpolation.adata_dict,
        time_points=interpolation.ts_points,
        time_keys=interpolation.time_keys,
        out_path=str(growth_figure),
        source_by_time={
            float(value): "Observed" if float(value) in observed_set else "Generated"
            for value in interpolation.ts_points
        },
        title="ARISTA growth-rate maps across dense time grid",
    )

    communications_pickle = None
    communications = None
    if config.run_communication:
        communications_pickle = output_dir / "all_time_communications.pkl"
        communications = compute_timepoint_communications(
            adata_dict=interpolation.adata_dict,
            time_points=interpolation.ts_points,
            annotation_key=config.annotation_key,
            f_net=context["runtime"].f_net,
            device=config.device,
            out_dir=str(output_dir / "attention"),
            remove_self_loop=False,
            winsor_quantile=0.995,
            save_pickle_path=str(communications_pickle),
        )

    spatiotemporal_html = None
    if config.run_3d:
        if communications is None:
            raise ValueError("run_3d=True requires run_communication=True.")
        spatiotemporal_html = output_dir / "spatiotemporal_3d.html"
        plot_points = list(map(float, config.plot_3d_time_points))
        plot_fig = plot_spatiotemporal_3d(
            adata_dict=interpolation.adata_dict,
            all_time_communications=communications,
            time_keys=[str(value) for value in plot_points],
            plot_time_points=plot_points,
            ts_points=interpolation.ts_points,
            observed_time_points=interpolation.observed_time_points,
            interp_points=interpolation.interp_points,
            annotation_key=config.annotation_key,
            label_to_color=context["label_to_color"],
            out_html=str(spatiotemporal_html),
            predicted_labels_list=lineage_labels,
            z_spacing=3.8,
            intra_threshold=0.0,
            edge_focus_celltype="reaEGC",
            edge_top_k=5,
            edge_top_k_focus_label="reaEGC",
            edge_color="rgba(25,25,25,0.75)",
            edge_show_arrows=True,
            edge_arrow_position=0.7,
            edge_arrow_in_slice_plane=True,
            edge_arrow_length_scale=0.14,
            edge_arrow_width_scale=0.65,
            edge_line_width_base=5,
            edge_line_width_scale=0.7,
            ribbon_min_count=10,
            ribbon_keep_source_cumfrac=0.85,
            ribbon_focus_celltype=["reaEGC"],
            ribbon_focus_source_only=True,
            focus_anchor_label="reaEGC",
            focus_anchor_frac=0.2,
            focus_anchor_k=None,
            focus_anchor_min_count=None,
            bidirectional_offset=0.2,
            bidirectional_curve=True,
            bidirectional_curve_points=18,
            ribbon_line_width_base=6,
            ribbon_line_width_scale=1.0,
            ribbon_line_alpha=0.55,
            ribbon_line_curve=0.12,
            ribbon_line_points=18,
            point_size=1.0,
            observed_point_alpha=0.7,
            generated_point_alpha=0.7,
            show_slice_border=True,
            slice_border_width=5,
            slice_border_color_observed="#5f6a72",
            slice_border_color_generated="#8c6d5a",
            slice_fill_color_observed="#e6f0f6",
            slice_fill_color_generated="#f6eee5",
            slice_fill_opacity=0.5,
            show_time_axis=False,
            show_legend=False,
            show_title=False,
            width=1200,
            height=900,
        )
        static_exports["spatiotemporal_3d"] = _export_plotly_bundle(
            plot_fig,
            output_dir / "spatiotemporal_3d",
        )

    assert_package_only_runtime()
    manifest = {
        "workflow": "arista_spatiotemporal_api",
        "runtime_contract": "installed CytoBridge public API; no vendor/legacy_arista_stack imports",
        "cytobridge_version": importlib.metadata.version("CytoBridge"),
        "cytobridge_module": str(Path(sys.modules["CytoBridge"].__file__).resolve()),
        "config": {
            key: [*value]
            if isinstance(value, tuple)
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in asdict(config).items()
        },
        "resolved": {
            "aligned_h5ad": str(context["aligned_h5ad"])
            if context["aligned_h5ad"]
            else None,
            "data_csv": str(context["data_csv"]) if context["data_csv"] else None,
            "model_dir": str(context["model_dir"]),
            "edge_predictor_root": str(context["edge_predictor_root"]),
            "classifier_cache_path": (
                str(context["classifier_cache_path"])
                if context["classifier_cache_path"] is not None
                else None
            ),
        },
        "model": _loaded_model_manifest(context["loaded"]),
        "dim": int(context["dim"]),
        "observed_time_points": interpolation.observed_time_points,
        "interpolated_time_points": interpolation.interp_points,
        "classifier_knn_neighbors": int(config.classifier_knn_neighbors),
        "spatial_warp_to_observed_piecewise": bool(
            config.spatial_warp_to_observed_piecewise
        ),
        "static_exports": static_exports,
        "composition": {
            "table": str(composition_csv),
            "figure": str(composition_figure),
        },
        "growth_dense_time_grid": {
            "table": str(growth_csv),
            "figure": str(growth_figure),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return AristaSpatiotemporalResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        snapshots_dir=snapshots_dir,
        lineage_html=lineage_html,
        composition_csv=composition_csv,
        composition_figure=composition_figure,
        growth_csv=growth_csv,
        growth_figure=growth_figure,
        communications_pickle=communications_pickle,
        spatiotemporal_html=spatiotemporal_html,
        classifier_cache_dir=classifier_cache_dir,
        time_points=tuple(interpolation.observed_time_points),
        generated_time_points=tuple(interpolation.interp_points),
    )


def run_arista_temporal_programs_api(
    config: AristaSpatiotemporalConfig,
    *,
    lr_database: str | Path,
    reference_h5ad: str | Path | None = None,
    n_top_genes: int = 250,
    n_gene_clusters: int = 2,
    n_lr_clusters: int = 2,
    pca_components_csv: str | Path | None = None,
    pca_center_csv: str | Path | None = None,
    preferred_species_tag: str | None = "hs",
    gene_profile_linkage_method: str = "average",
    gene_profile_cluster_order: str = "peak_time",
    lr_profile_linkage_method: str = "average",
    lr_profile_cluster_order: str = "peak_time",
    communication_max_cells_per_timepoint: int | None = None,
    communication_random_seed: int = 42,
    communication_rng_warmup_max_cells_per_timepoint: int | None = None,
) -> AristaTemporalProgramsResult:
    """Recompute ARISTA gene and LR temporal programs through package APIs."""
    import anndata as ad

    set_global_random_seed(config.random_seed)
    context = load_arista_spatiotemporal_context(config)
    output_dir = _secondary_output_dir(config)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    classifier_cache_dir = (
        Path(config.classifier_cache_dir).expanduser().resolve()
        if config.classifier_cache_dir is not None
        else output_dir / "classifier_cache"
    )
    interpolation = _run_arista_interpolation(
        config=config,
        context=context,
        output_dir=output_dir,
        classifier_cache_dir=classifier_cache_dir,
    )

    communication_indices_by_time = None
    if communication_rng_warmup_max_cells_per_timepoint is not None:
        warmup_n = int(communication_rng_warmup_max_cells_per_timepoint)
        if warmup_n <= 0:
            raise ValueError(
                "communication_rng_warmup_max_cells_per_timepoint must be positive."
            )
        sampling_rng = np.random.default_rng(int(communication_random_seed))
        for time_value in interpolation.ts_points:
            adata_t = interpolation.adata_dict[str(float(time_value))]
            if adata_t.n_obs > warmup_n:
                sampling_rng.choice(adata_t.n_obs, size=warmup_n, replace=False)
        if communication_max_cells_per_timepoint is not None:
            cap = int(communication_max_cells_per_timepoint)
            communication_indices_by_time = {}
            for time_value in interpolation.ts_points:
                key = str(float(time_value))
                n_obs = interpolation.adata_dict[key].n_obs
                if n_obs > cap:
                    communication_indices_by_time[key] = np.sort(
                        sampling_rng.choice(n_obs, size=cap, replace=False)
                    )

    communication_pickle = output_dir / "all_time_communications.pkl"
    communications = compute_timepoint_communications(
        adata_dict=interpolation.adata_dict,
        time_points=interpolation.ts_points,
        annotation_key=config.annotation_key,
        f_net=context["runtime"].f_net,
        device=config.device,
        out_dir=str(output_dir / "attention"),
        remove_self_loop=True,
        winsor_quantile=0.995,
        save_pickle_path=str(communication_pickle),
        max_cells_per_timepoint=communication_max_cells_per_timepoint,
        random_seed=int(communication_random_seed),
        cell_indices_by_time=communication_indices_by_time,
    )

    if reference_h5ad is None:
        reference_adata = context["adata"]
        reference_path = context["aligned_h5ad"]
    else:
        reference_path = _require_file(Path(reference_h5ad), "ARISTA reference H5AD")
        reference_adata = ad.read_h5ad(reference_path)
    if (pca_components_csv is None) != (pca_center_csv is None):
        raise ValueError(
            "pca_components_csv and pca_center_csv must be supplied together."
        )
    pca_components_path = None
    pca_center_path = None
    pca_reconstruction = None
    if pca_components_csv is not None:
        pca_components_path = _require_file(
            Path(pca_components_csv), "PCA components CSV"
        )
        pca_center_path = _require_file(Path(pca_center_csv), "PCA center CSV")
        pca_reconstruction = load_pca_reconstruction_spec(
            pca_components_path,
            pca_center_path,
        )
    if pca_reconstruction is None and "PCs" not in reference_adata.varm:
        raise KeyError(
            "The gene/LR workflow requires PCA loadings in reference_adata.varm['PCs']; "
            "pass reference_h5ad or the PCA components/center CSV pair for portable "
            "legacy-checkpoint runs."
        )
    lr_path = _require_file(Path(lr_database), "ligand-receptor database")

    gene = summarize_temporal_gene_patterns(
        interpolation.adata_dict,
        reference_adata,
        time_points=interpolation.ts_points,
        spatial_dim=2,
        n_top_genes=int(n_top_genes),
        n_clusters=int(n_gene_clusters),
        preferred_species_tag=preferred_species_tag,
        pca_reconstruction=pca_reconstruction,
        profile_linkage_method=gene_profile_linkage_method,
        profile_cluster_order=gene_profile_cluster_order,
    )
    lr = project_communication_to_lr_timecourses(
        interpolation.adata_dict,
        reference_adata,
        communications,
        lr_path,
        time_points=interpolation.ts_points,
        annotation_key=config.annotation_key,
        matrix_key="M_per_source",
        spatial_dim=2,
        expression_space="count",
        complex_mode="min",
        require_all_subunits=False,
        duplicate_policy="first",
        preferred_species_tag=preferred_species_tag,
        n_clusters=int(n_lr_clusters),
        pca_reconstruction=pca_reconstruction,
        profile_linkage_method=lr_profile_linkage_method,
        profile_cluster_order=lr_profile_cluster_order,
    )

    gene.expression.to_csv(tables_dir / "gene_expression_by_time.csv")
    gene.top_variable_genes.to_csv(
        tables_dir / "gene_top_variable_patterns.csv", index=False
    )
    gene.gene_name_map.to_csv(tables_dir / "gene_name_map.csv", index=False)
    gene.clustering.normalized_profiles.to_csv(
        tables_dir / "gene_normalized_profiles.csv"
    )
    gene.clustering.prototypes.to_csv(
        tables_dir / "gene_pattern_prototypes.csv", index=False
    )
    gene.clustering.diagnostics.to_csv(
        tables_dir / "gene_pattern_diagnostics.csv", index=False
    )
    lr.pair_timecourse.to_csv(tables_dir / "lr_pair_timecourse.csv", index=False)
    lr.celltype_timecourse.to_csv(
        tables_dir / "lr_celltype_timecourse.csv", index=False
    )
    lr.pattern_summary.to_csv(tables_dir / "lr_pattern_summary.csv", index=False)
    lr.coverage.to_csv(tables_dir / "lr_projection_coverage.csv", index=False)
    lr.clustering.normalized_profiles.to_csv(
        tables_dir / "lr_normalized_profiles.csv"
    )
    lr.clustering.assignments.to_csv(
        tables_dir / "lr_pattern_assignments.csv", index=False
    )
    lr.clustering.prototypes.to_csv(
        tables_dir / "lr_pattern_prototypes.csv", index=False
    )
    lr.clustering.diagnostics.to_csv(
        tables_dir / "lr_pattern_diagnostics.csv", index=False
    )

    gene_heatmap = plot_temporal_gene_heatmap(
        gene.expression,
        gene.top_variable_genes,
        out_path=figures_dir / "gene_temporal_heatmap.svg",
        title="ARISTA temporal gene programs",
    )
    gene_pattern_figure = plot_temporal_pattern_prototypes(
        gene.clustering.prototypes,
        out_path=figures_dir / "gene_pattern_prototypes.svg",
        title="ARISTA temporal gene-pattern prototypes",
        y_label="Mean gene-wise z-score",
    )
    lr_prototype_figure = plot_temporal_pattern_prototypes(
        lr.clustering.prototypes,
        out_path=figures_dir / "lr_pattern_prototypes.svg",
        title="ARISTA ligand-receptor pattern prototypes",
        y_label="Mean normalized LR score",
    )
    lr_profiles_figure = plot_temporal_profile_small_multiples(
        lr.clustering.normalized_profiles,
        lr.clustering.assignments,
        out_path=figures_dir / "lr_pair_small_multiples.svg",
        title=f"ARISTA LR profiles (n={lr.pattern_summary.shape[0]})",
    )

    assert_package_only_runtime()
    manifest = {
        "workflow": "arista_temporal_programs_api",
        "runtime_contract": "installed CytoBridge public API; no vendor/legacy_arista_stack imports",
        "cytobridge_module": str(Path(sys.modules["CytoBridge"].__file__).resolve()),
        "config": {
            key: [*value]
            if isinstance(value, tuple)
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in asdict(config).items()
        },
        "inputs": {
            "reference_h5ad": str(reference_path) if reference_path else None,
            "reference_h5ad_sha256": _sha256(Path(reference_path))
            if reference_path
            else None,
            "pca_components_csv": str(pca_components_path)
            if pca_components_path
            else None,
            "pca_components_csv_sha256": _sha256(pca_components_path)
            if pca_components_path
            else None,
            "pca_center_csv": str(pca_center_path) if pca_center_path else None,
            "pca_center_csv_sha256": _sha256(pca_center_path)
            if pca_center_path
            else None,
            "lr_database": str(lr_path),
            "lr_database_sha256": _sha256(lr_path),
            "model_dir": str(context["model_dir"]),
        },
        "settings": {
            "n_top_genes": int(n_top_genes),
            "n_gene_clusters": int(n_gene_clusters),
            "n_lr_clusters": int(n_lr_clusters),
            "preferred_species_tag": preferred_species_tag,
            "gene_profile_linkage_method": gene_profile_linkage_method,
            "gene_profile_cluster_order": gene_profile_cluster_order,
            "lr_profile_linkage_method": lr_profile_linkage_method,
            "lr_profile_cluster_order": lr_profile_cluster_order,
            "communication_max_cells_per_timepoint": communication_max_cells_per_timepoint,
            "communication_random_seed": int(communication_random_seed),
            "communication_rng_warmup_max_cells_per_timepoint": (
                communication_rng_warmup_max_cells_per_timepoint
            ),
            "remove_self_loop": True,
            "communication_matrix": "M_per_source",
        },
        "summary": {
            "time_points": interpolation.ts_points,
            "n_gene_profiles": int(gene.top_variable_genes.shape[0]),
            "n_lr_pairs": int(lr.pattern_summary.shape[0]),
            "lr_cluster_counts": lr.pattern_summary["cluster"]
            .value_counts()
            .sort_index()
            .to_dict(),
        },
        "outputs": {
            "tables_dir": str(tables_dir),
            "figures_dir": str(figures_dir),
            "communication_pickle": str(communication_pickle),
            "gene_heatmap": str(gene_heatmap),
            "gene_pattern_figure": str(gene_pattern_figure),
            "lr_prototype_figure": str(lr_prototype_figure),
            "lr_profiles_figure": str(lr_profiles_figure),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return AristaTemporalProgramsResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        communication_pickle=communication_pickle,
        gene_heatmap=gene_heatmap,
        gene_pattern_figure=gene_pattern_figure,
        lr_prototype_figure=lr_prototype_figure,
        lr_profiles_figure=lr_profiles_figure,
    )


def _secondary_output_dir(config: AristaSpatiotemporalConfig) -> Path:
    output_dir = (
        Path(config.output_dir).expanduser().resolve()
        if config.output_dir is not None
        else (RESULTS_ROOT / config.output_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _secondary_manifest(
    *,
    workflow: str,
    config: AristaSpatiotemporalConfig,
    context: dict[str, Any],
    outputs: dict[str, str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    assert_package_only_runtime()
    return {
        "workflow": workflow,
        "runtime_contract": "installed CytoBridge public API; no vendor/legacy_arista_stack imports",
        "cytobridge_version": importlib.metadata.version("CytoBridge"),
        "cytobridge_module": str(Path(sys.modules["CytoBridge"].__file__).resolve()),
        "model": _loaded_model_manifest(context["loaded"]),
        "model_format": config.model_format,
        "resolved": {
            "aligned_h5ad": str(context["aligned_h5ad"])
            if context["aligned_h5ad"]
            else None,
            "data_csv": str(context["data_csv"]) if context["data_csv"] else None,
            "model_dir": str(context["model_dir"]),
            "edge_predictor_root": str(context["edge_predictor_root"]),
        },
        "settings": settings,
        "outputs": outputs,
    }


def run_arista_growth_interaction_api(
    config: AristaSpatiotemporalConfig | None = None,
    *,
    max_cells_per_timepoint: int | None = None,
) -> AristaGrowthInteractionResult:
    """Regenerate the ARISTA growth/interaction cell-type bubble via package APIs."""
    config = config or AristaSpatiotemporalConfig(
        output_name="arista_growth_interaction_api"
    )
    set_global_random_seed(config.random_seed)
    context = load_arista_spatiotemporal_context(config)
    output_dir = _secondary_output_dir(config)
    cutoff = float(
        getattr(context["loaded"].model.interaction_net, "cutoff", 1000.0)
    )
    summary = summarize_growth_interaction_by_celltype(
        context["adata"],
        context["loaded"].model,
        annotation_key=config.annotation_key,
        time_key=context["resolved_time_key"],
        obsm_key=config.obsm_key,
        spatial_key=config.spatial_key,
        concat_spatial=config.concat_spatial,
        interaction_m=1024,
        interaction_threshold=cutoff,
        max_cells_per_timepoint=max_cells_per_timepoint,
        random_seed=config.random_seed,
        device=config.device,
    )
    raw_csv = output_dir / "growth_interaction_raw.csv"
    grouped_csv = output_dir / "growth_interaction_celltype_summary.csv"
    bubble_path = output_dir / "growth_interaction_celltype_bubble.svg"
    summary.raw.to_csv(raw_csv, index=False)
    summary.grouped.to_csv(grouped_csv, index=False)
    plot_growth_interaction_bubble(
        summary.grouped,
        out_path=str(bubble_path),
    )
    manifest = _secondary_manifest(
        workflow="arista_growth_interaction_api",
        config=config,
        context=context,
        outputs={
            "raw_csv": str(raw_csv),
            "grouped_csv": str(grouped_csv),
            "bubble": str(bubble_path),
        },
        settings={
            "interaction_m": 1024,
            "interaction_cutoff": cutoff,
            "max_cells_per_timepoint": max_cells_per_timepoint,
            "random_seed": int(config.random_seed),
        },
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return AristaGrowthInteractionResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        raw_csv=raw_csv,
        grouped_csv=grouped_csv,
        bubble_path=bubble_path,
    )


def run_arista_direction_correlation_api(
    config: AristaSpatiotemporalConfig | None = None,
    *,
    target_timepoint: float = 1.0,
    focus_label_keyword: str = "reaEGC",
    pad_ratio: float = 0.15,
    n_neighbors: int = 30,
    max_cells: int | None = None,
) -> AristaDirectionCorrelationResult:
    """Regenerate the ARISTA full-vs-interaction spatial direction ROI."""
    config = config or AristaSpatiotemporalConfig(
        output_name="arista_direction_correlation_api"
    )
    set_global_random_seed(config.random_seed)
    context = load_arista_spatiotemporal_context(config)
    output_dir = _secondary_output_dir(config)
    cutoff = float(
        getattr(context["loaded"].model.interaction_net, "cutoff", 1000.0)
    )
    raw_times = np.asarray(
        [
            float(value)
            for value in context["adata"].obs[context["resolved_time_key"]]
        ],
        dtype=float,
    )
    available_times = np.unique(raw_times)
    selected_time = float(
        available_times[
            np.argmin(np.abs(available_times - float(target_timepoint)))
        ]
    )
    positions = np.flatnonzero(np.isclose(raw_times, selected_time))
    if max_cells is not None and positions.size > int(max_cells):
        rng = np.random.default_rng(int(config.random_seed))
        positions = np.sort(
            rng.choice(positions, size=int(max_cells), replace=False)
        )
    direction_adata = context["adata"][positions].copy()
    components = compute_velocity_components_from_adata(
        direction_adata,
        context["loaded"].model,
        dim=context["dim"],
        interaction_m=1024,
        interaction_threshold=cutoff,
        device=config.device,
        time_key=context["resolved_time_key"],
        obsm_key=config.obsm_key,
        spatial_key=config.spatial_key,
        concat_spatial=config.concat_spatial,
        write_to_adata=True,
        reuse_if_present=False,
    )
    labels = direction_adata.obs[config.annotation_key].astype(str).to_numpy()
    spatial_coordinates = np.asarray(
        direction_adata.obsm[config.spatial_key], dtype=np.float32
    )
    spatial_dim = int(spatial_coordinates.shape[1])
    if spatial_dim < 2 or components["full"].shape[1] < spatial_dim + 2:
        raise ValueError(
            "ARISTA velocity panels require two spatial and at least two PCA dimensions."
        )
    pca_coordinates = np.asarray(components["features"], dtype=np.float32)[
        :, spatial_dim : spatial_dim + 2
    ]
    spatial_velocity_figure = output_dir / "full_velocity_spatial.svg"
    pca_velocity_figure = output_dir / "full_velocity_pca.svg"
    plot_velocity_component(
        coords=spatial_coordinates[:, :2],
        velocity=np.asarray(components["full"], dtype=np.float32)[:, :2],
        labels=labels,
        label_to_color=context["label_to_color"],
        title=f"Spatial migration velocity (t={selected_time:g})",
        out_path=str(spatial_velocity_figure),
        basis="spatial",
        density=1.8,
        show_legend=True,
    )
    plot_velocity_component(
        coords=pca_coordinates,
        velocity=np.asarray(components["full"], dtype=np.float32)[
            :, spatial_dim : spatial_dim + 2
        ],
        labels=labels,
        label_to_color=context["label_to_color"],
        title=f"Gene/PCA velocity (t={selected_time:g})",
        out_path=str(pca_velocity_figure),
        basis="spatial",
        density=1.6,
        show_legend=False,
    )
    roi_csv = output_dir / "full_vs_interaction_direction_roi.csv"
    figure_path = output_dir / "full_vs_interaction_direction_roi.svg"
    result = plot_spatial_component_direction_correlation_roi_from_adata(
        direction_adata,
        str(figure_path),
        target_timepoint=float(target_timepoint),
        component_a_key="full_drift_model",
        component_b_key="interaction_model",
        component_a_label="full",
        component_b_label="interaction",
        time_key=context["resolved_time_key"],
        annotation_key=config.annotation_key,
        focus_label_keyword=focus_label_keyword,
        pad_ratio=float(pad_ratio),
        n_neighbors=int(n_neighbors),
        obsm_key=config.obsm_key,
        spatial_key=config.spatial_key,
        concat_spatial=config.concat_spatial,
        csv_path=str(roi_csv),
    )
    component_summary = output_dir / "velocity_component_norms.csv"
    pd.DataFrame(
        {
            "component": ["drift", "interaction", "score", "full"],
            "mean_norm": [
                float(np.linalg.norm(components[name], axis=1).mean())
                for name in ("drift", "interaction", "score", "full")
            ],
        }
    ).to_csv(component_summary, index=False)
    manifest = _secondary_manifest(
        workflow="arista_direction_correlation_api",
        config=config,
        context=context,
        outputs={
            "roi_csv": str(roi_csv),
            "figure": str(figure_path),
            "spatial_velocity_figure": str(spatial_velocity_figure),
            "pca_velocity_figure": str(pca_velocity_figure),
            "component_summary": str(component_summary),
        },
        settings={
            "target_timepoint": float(target_timepoint),
            "selected_timepoint": selected_time,
            "focus_label_keyword": focus_label_keyword,
            "pad_ratio": float(pad_ratio),
            "n_neighbors": int(n_neighbors),
            "interaction_m": 1024,
            "interaction_cutoff": cutoff,
            "roi_bounds": list(result.roi_bounds),
            "n_roi_cells": int(len(result.table)),
            "max_cells": max_cells,
        },
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return AristaDirectionCorrelationResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        roi_csv=roi_csv,
        figure_path=figure_path,
        spatial_velocity_figure=spatial_velocity_figure,
        pca_velocity_figure=pca_velocity_figure,
        n_roi_cells=int(len(result.table)),
    )


__all__ = [
    "AristaApiPaths",
    "AristaDirectionCorrelationResult",
    "AristaGrowthInteractionResult",
    "AristaSpatiotemporalConfig",
    "AristaSpatiotemporalResult",
    "AristaVelocityConfig",
    "AristaVelocityResult",
    "assert_package_only_runtime",
    "load_arista_api_context",
    "load_arista_spatiotemporal_context",
    "resolve_arista_api_paths",
    "run_arista_velocity_t1_streams",
    "run_arista_spatiotemporal_api",
    "run_arista_direction_correlation_api",
    "run_arista_growth_interaction_api",
]
