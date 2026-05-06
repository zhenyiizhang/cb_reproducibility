from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


DOWNSTREAM_ROOT = Path(__file__).resolve().parents[1]
MOSTA_ASSET_ROOT = DOWNSTREAM_ROOT / "assets" / "mosta"
MOSTA_DATA_ROOT = DOWNSTREAM_ROOT / "data" / "mosta"


@dataclass(frozen=True)
class MostaAssets:
    config_path: Path
    data_csv: Path
    annotation_key: str
    model_dir: Path
    edge_predictor_root: Path
    classifier_cache_path: Path
    label_color_json: Path
    color_h5ad: str | None = None


@dataclass(frozen=True)
class MostaContext:
    assets: MostaAssets
    df: pd.DataFrame
    dim: int
    device: str
    runtime: object


def build_mosta_snapshot_variants(
    *,
    context: MostaContext,
    interpolation,
    config: "MostaRunConfig",
) -> dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] | None:
    from CytoBridge.tl import predict_labels_for_points
    from CytoBridge.tl.downstream.pipeline_utils import downsample_xy

    observed_time_points = interpolation.observed_time_points
    if len(observed_time_points) == 0:
        return None

    dim = int(context.dim)
    df = context.df
    annotation_key = context.assets.annotation_key
    ts_points = interpolation.ts_points
    ts_index = {float(t): i for i, t in enumerate(ts_points)}

    rng_snapshot = np.random.default_rng(142)
    feature_cols_snapshot = [f"x{i}" for i in range(1, dim + 1)]
    compare_variants: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    has_generated_compare = False

    for t_obs in observed_time_points:
        t_obs_f = float(t_obs)
        subset_obs = df[df["samples"] == t_obs_f]
        if subset_obs.empty:
            continue

        X_gen = None
        labels_gen = None
        idx = ts_index.get(t_obs_f)
        use_final_generated = (
            config.piecewise_spatial_warp
            or (not config.split_sde_piecewise if hasattr(config, "split_sde_piecewise") else True)
            or (
                hasattr(config, "split_sde_piecewise")
                and config.split_sde_piecewise
                and getattr(config, "spatial_warp_to_observed", False)
                and (not config.use_real_for_observed)
            )
        )
        if (
            use_final_generated
            and interpolation.sde_points_split is not None
            and interpolation.predicted_labels_split is not None
            and idx is not None
        ):
            X_gen = np.asarray(interpolation.sde_points_split[idx], dtype=np.float32)
            labels_gen = np.asarray(interpolation.predicted_labels_split[idx]).astype(str)
        elif (
            interpolation.piecewise_endpoint_by_observed is not None
            and t_obs_f in interpolation.piecewise_endpoint_by_observed
        ):
            X_gen = np.asarray(interpolation.piecewise_endpoint_by_observed[t_obs_f], dtype=np.float32)
            labels_gen = predict_labels_for_points(
                points=X_gen,
                time_value=t_obs_f,
                model=interpolation.classifier_model,
                label_encoder=interpolation.label_encoder,
                feature_dim=int(interpolation.classifier_feature_dim),
                device=context.device,
                knn_neighbors=int(config.classifier_knn_neighbors),
            )

        if X_gen is not None and labels_gen is not None:
            X_gen, labels_gen = downsample_xy(
                X_gen,
                labels_gen,
                config.slice_max_cells_per_timepoint if hasattr(config, "slice_max_cells_per_timepoint") else None,
                rng_snapshot,
            )
            compare_variants.setdefault(t_obs_f, {})["generated"] = (
                np.asarray(X_gen, dtype=np.float32)[:, :2],
                np.asarray(labels_gen).astype(str),
            )
            has_generated_compare = True

    observed_variants = compare_variants if has_generated_compare else None
    if (
        getattr(config, "split_sde_piecewise", False)
        and getattr(config, "split_sde_piecewise_include_end", False)
        and getattr(config, "spatial_warp_to_observed", False)
        and interpolation.sde_points_split_prewarp is not None
        and interpolation.predicted_labels_split_prewarp is not None
        and interpolation.sde_points_split is not None
        and interpolation.predicted_labels_split is not None
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
                np.asarray(interpolation.sde_points_split_prewarp[idx], dtype=np.float32)[:, :2],
                np.asarray(interpolation.predicted_labels_split_prewarp[idx]).astype(str),
            )
            observed_variants.setdefault(t_float, {})["postwarp"] = (
                np.asarray(interpolation.sde_points_split[idx], dtype=np.float32)[:, :2],
                np.asarray(interpolation.predicted_labels_split[idx]).astype(str),
            )

    return observed_variants


@dataclass(frozen=True)
class MostaRunConfig:
    output_name: str
    piecewise_spatial_warp: bool
    skip_export: bool = True
    skip_snapshots: bool = False
    interp_time_points: str = "0.5,1.5,2.5"
    plot_3d_time_points: str = "all"
    sde_n_samples: int = 20000
    classifier_n_pcs: int = 12
    classifier_best_metric: str = "bacc"
    classifier_train_on_full_data: bool = True
    classifier_cache_path: str | None = "assets/mosta/classifier_cache/classifier_resmlp_52fb7dc647bfe334.pt"
    use_real_for_observed: bool = False


def resolve_mosta_classifier_cache_path(path: str | None) -> Path:
    if path is None or not str(path).strip():
        return get_mosta_assets().classifier_cache_path
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = DOWNSTREAM_ROOT / candidate
    return candidate.resolve()


def parse_requested_plot_points(config: MostaRunConfig) -> list[float] | None:
    raw = str(config.plot_3d_time_points).strip().lower()
    if raw in {"", "all", "none"}:
        return None
    return [float(x.strip()) for x in str(config.plot_3d_time_points).split(",") if x.strip()]


def build_mosta_interpolation_kwargs(
    *,
    context: MostaContext,
    config: MostaRunConfig,
    output_dir: Path,
) -> dict:
    classifier_cache_path = resolve_mosta_classifier_cache_path(config.classifier_cache_path)
    return {
        "df": context.df,
        "dim": context.dim,
        "annotation_key": context.assets.annotation_key,
        "runtime": context.runtime,
        "device": context.device,
        "output_dir": str(output_dir),
        "requested_plot_points": parse_requested_plot_points(config),
        "interp_time_points": [float(x) for x in str(config.interp_time_points).split(",") if x],
        "no_interp": False,
        "use_real_for_observed": config.use_real_for_observed,
        "classifier_cache_path": str(classifier_cache_path),
        "classifier_best_metric": config.classifier_best_metric,
        "classifier_n_pcs": config.classifier_n_pcs,
        "classifier_knn_neighbors": 10,
        "sde_n_samples": config.sde_n_samples,
        "sde_dt": 0.05,
        "split_sde_dt": 0.05,
        "split_sigma_scalar": 0.03,
        "split_sigma_vector": None,
        "split_growth_alpha": 1.0,
        "spatial_warp_to_observed": False,
        "spatial_warp_to_observed_piecewise": config.piecewise_spatial_warp,
        "spatial_warp_k": 8,
        "spatial_warp_eps": 1e-6,
        "slice_max_cells_per_timepoint": None,
        "random_seed": 42,
    }


def get_mosta_assets() -> MostaAssets:
    return MostaAssets(
        config_path=MOSTA_ASSET_ROOT / "mosta_config.yaml",
        data_csv=MOSTA_DATA_ROOT / "mosta_four_time_with_celltype_refined.csv",
        annotation_key="Annotation",
        model_dir=MOSTA_ASSET_ROOT / "model",
        edge_predictor_root=MOSTA_ASSET_ROOT / "edge_classifier",
        classifier_cache_path=MOSTA_ASSET_ROOT / "classifier_cache" / "classifier_resmlp_52fb7dc647bfe334.pt",
        label_color_json=MOSTA_ASSET_ROOT / "label_to_color.json",
        color_h5ad=None,
    )


def load_mosta_context(*, random_seed: int | None = 42) -> MostaContext:
    from CytoBridge.tl import (
        build_dynamical_runtime,
        load_legacy_dynamical_model_from_dir,
        set_global_random_seed,
    )
    from CytoBridge.tl.downstream.pipeline_utils import require_columns
    from CytoBridge.utils.config import load_config

    assets = get_mosta_assets()
    set_global_random_seed(random_seed)
    config = load_config(str(assets.config_path))
    dim = int(config["data"]["dim"])

    df = pd.read_csv(assets.data_csv, low_memory=False)
    require_columns(df, ["samples"] + [f"x{i}" for i in range(1, dim + 1)], ctx=str(assets.data_csv))
    if assets.annotation_key not in df.columns:
        raise ValueError(f"Expected '{assets.annotation_key}' column in {assets.data_csv}.")
    df = df.copy()
    df["samples"] = df["samples"].astype(float)
    df[assets.annotation_key] = df[assets.annotation_key].astype(str)
    df = df.sort_values("samples").reset_index(drop=True)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model = load_legacy_dynamical_model_from_dir(
        str(assets.model_dir),
        device=device,
        edge_predictor_root=str(assets.edge_predictor_root),
    )
    runtime = build_dynamical_runtime(loaded_model)
    return MostaContext(
        assets=assets,
        df=df,
        dim=dim,
        device=device,
        runtime=runtime,
    )


def resolve_mosta_output_dir(config: MostaRunConfig) -> Path:
    return DOWNSTREAM_ROOT / "results" / config.output_name


def build_mosta_argv(config: MostaRunConfig) -> list[str]:
    output_dir = resolve_mosta_output_dir(config)
    classifier_cache_path = resolve_mosta_classifier_cache_path(config.classifier_cache_path)
    args: list[str] = [
        "--classifier-best-metric",
        str(config.classifier_best_metric),
        "--interp-time-points",
        str(config.interp_time_points),
        "--sde-n-samples",
        str(config.sde_n_samples),
        "--plot-3d-time-points",
        str(config.plot_3d_time_points),
        "--output-dir",
        str(output_dir),
        "--classifier-n-pcs",
        str(config.classifier_n_pcs),
        "--classifier-cache-path",
        str(classifier_cache_path),
    ]
    if config.classifier_train_on_full_data:
        args.append("--classifier-train-on-full-data")
    if not config.use_real_for_observed:
        args.append("--no-use-real-for-observed")
    if config.piecewise_spatial_warp:
        args.append("--spatial-warp-to-observed-piecewise")
    if config.skip_export:
        args.append("--skip-export")
    if config.skip_snapshots:
        args.append("--skip-snapshots")
    return args


def run_mosta_config(config: MostaRunConfig, *, clean_output: bool = True) -> Path:
    from .mosta_pipeline import main as run_mosta_cli

    output_dir = resolve_mosta_output_dir(config)
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    run_mosta_cli(build_mosta_argv(config))
    return output_dir
