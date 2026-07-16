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

from CytoBridge.pl import plot_velocity_component
from CytoBridge.tl import (
    build_dynamical_runtime,
    compute_velocity_components,
    infer_feature_columns,
    load_label_to_color,
    load_legacy_dynamical_model_from_dir,
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
            "label_color_json": PROJECT_ROOT / "assets" / "arista" / "label_to_color.json",
        }

    return AristaApiPaths(
        data_csv=_require_file(
            Path(config.data_csv) if config.data_csv is not None else defaults["data_csv"],
            "ARISTA annotated model-input CSV",
        ),
        model_dir=_require_dir(
            Path(config.model_dir) if config.model_dir is not None else defaults["model_dir"],
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
        raise RuntimeError(f"Legacy vendor modules were imported unexpectedly: {forbidden}")


def load_arista_api_context(config: AristaVelocityConfig) -> dict[str, Any]:
    """Load the annotated model inputs and trained checkpoint via CytoBridge."""

    paths = resolve_arista_api_paths(config)
    df = pd.read_csv(paths.data_csv)
    if config.annotation_key not in df.columns:
        raise KeyError(f"Missing annotation column '{config.annotation_key}' in {paths.data_csv}")
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
        raise ValueError(f"No rows found at timepoint {timepoint}; available={available}")
    if max_cells is not None:
        max_cells = int(max_cells)
        if max_cells <= 0:
            raise ValueError(f"max_cells must be positive or None, got {max_cells}")
        if len(selected) > max_cells:
            selected = selected.sample(n=max_cells, random_state=int(random_seed)).sort_index()
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
        raise FloatingPointError("Non-finite values found in an ARISTA velocity component.")

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
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "resolved_paths": {key: str(value) for key, value in asdict(paths).items()},
        "input_hashes_sha256": _checkpoint_hashes(paths),
        "model": {
            "class": type(context["loaded"].model).__name__,
            "weight_stage": context["loaded"].weight_stage,
            "score_stage": context["loaded"].score_stage,
            "components": list(context["loaded"].model.components),
        },
        "n_cells": int(data.shape[0]),
        "dim": int(data.shape[1]),
        "full_identity_max_error": full_identity_error,
        "plot_fallbacks": plot_fallbacks,
        "figures": {key: str(value) for key, value in figure_paths.items()},
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return AristaVelocityResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        component_summary_csv=component_summary_csv,
        component_arrays_npz=component_arrays_npz,
        figure_paths=figure_paths,
        n_cells=int(data.shape[0]),
        dim=int(data.shape[1]),
    )


__all__ = [
    "AristaApiPaths",
    "AristaVelocityConfig",
    "AristaVelocityResult",
    "assert_package_only_runtime",
    "load_arista_api_context",
    "resolve_arista_api_paths",
    "run_arista_velocity_t1_streams",
]
