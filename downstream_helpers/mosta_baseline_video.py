#!/usr/bin/env python3
"""Notebook-friendly MOSTA baseline video pipeline."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import imageio.v2 as imageio
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
    for path in (PROJECT_ROOT, CYTOBRIDGE_REPO, VENDOR_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_module_paths()

from evaluation.arista_code.arista_helpers import (  # noqa: E402
    analyze_attention_by_celltype,
    save_interpolated_attention,
    train_mlp_classifier,
)

from .mosta import (  # noqa: E402
    DOWNSTREAM_ROOT,
    MostaContext,
    load_mosta_context,
    resolve_mosta_classifier_cache_path,
)


def apply_nature_methods_mpl_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 8.0,
            "figure.titlesize": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )


def get_nature_scatter_defaults() -> dict[str, float]:
    return {
        "frame_dpi": 300.0,
        "point_size": 2.5,
        "point_alpha": 0.9,
        "title_fontsize": 10.0,
        "suptitle_fontsize": 12.0,
        "legend_fontsize": 7.0,
    }


NATURE_SCATTER_DEFAULTS = get_nature_scatter_defaults()
apply_nature_methods_mpl_style()
MOSTA_RESULTS_ROOT = DOWNSTREAM_ROOT / "results"


@dataclass(frozen=True)
class MostaBaselineVideoConfig:
    output_name: str = "mosta_brain_comm_baseline_notebook"
    time_start: float = 0.0
    time_end: float = 3.0
    time_step: float = 0.05
    sde_n_samples: int = 50000
    split_sde_dt: float = 0.05
    split_sigma: float = 0.03
    split_growth_alpha: float = 1.0
    spatial_warp_to_observed: bool = True
    spatial_warp_k: int = 8
    spatial_warp_eps: float = 1e-6
    interaction_m: int = 1024
    classifier_epochs: int = 500
    classifier_hidden: int = 128
    classifier_n_pcs: int = 12
    classifier_feature_start: int = 1
    classifier_best_metric: str = "bacc"
    classifier_train_on_full_data: bool = True
    classifier_cache: bool = True
    classifier_cache_path: str | None = None
    classifier_cache_dir: str | None = None
    classifier_cache_tag: str | None = None
    classifier_force_load: bool = False
    video_style: str = "fixed_2d"
    video_point_subsample: int = 50000
    gif_fps: int = 4
    frame_dpi: int = int(NATURE_SCATTER_DEFAULTS["frame_dpi"])
    point_size: float = float(NATURE_SCATTER_DEFAULTS["point_size"])
    point_alpha: float = float(NATURE_SCATTER_DEFAULTS["point_alpha"])
    show_titles: bool = False
    draw_focus_interactions: bool = True
    focus_celltype: str = "Brain"
    focus_mode: str = "both"
    focus_top_k: int = 3
    focus_min_weight: float = 0.0
    focus_edge_quantile: float | None = None
    focus_edge_width_min: float = 1.0
    focus_edge_width_max: float = 2.6
    focus_edge_alpha_min: float = 0.25
    focus_edge_alpha_max: float = 0.85
    focus_edge_curve: float = 0.18
    focus_edge_curve_bi: float = 0.35
    focus_neighbor_pct: float = 0.2
    focus_other_desaturate: float = 0.0
    focus_show_endpoints: bool = True
    focus_endpoint_size: float = 26.0
    focus_endpoint_alpha: float = 0.95
    focus_endpoint_edgecolor: str = "#ffffff"
    focus_endpoint_linewidth: float = 0.8
    attention_stride: int = 2
    attention_max_cells: int = 30000
    attention_use_real_observed: bool = False
    axis_limit_mode: str = "per_timepoint"
    axis_pad_frac: float = 0.03
    panel_size: float = 4.2
    camera_elev: float = 28.0
    camera_azim: float = -54.0
    random_seed: int | None = 42
    clean_output: bool = True


@dataclass(frozen=True)
class MostaBaselineVideoResult:
    output_dir: Path
    frames_dir: Path
    gif_path: Path
    summary_path: Path
    baseline_counts_csv: Path
    label_color_json: Path
    all_time_communications_pkl: Path | None
    communication_sampling_csv: Path | None
    frame_count: int
    timepoints: tuple[float, ...]


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _float_close(a: float, b: float, eps: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= eps


def _build_ts_points(time_start: float, time_end: float, time_step: float) -> list[float]:
    if time_step <= 0:
        raise ValueError("time_step must be > 0")
    if time_end < time_start:
        raise ValueError("time_end must be >= time_start")

    n_steps = int(round((time_end - time_start) / time_step))
    points = [round(float(time_start + i * time_step), 10) for i in range(n_steps + 1)]
    if not _float_close(points[-1], time_end):
        points.append(round(float(time_end), 10))

    out: list[float] = []
    for val in points:
        if not out or not _float_close(out[-1], val):
            out.append(val)
    return out


def _tag_float(v: float) -> str:
    s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _build_classifier_feature_spec(*, dim: int, start_idx_1based: int, n_pcs: int) -> tuple[list[str], list[int]]:
    if int(start_idx_1based) < 1:
        raise ValueError("classifier_feature_start must be >= 1")
    if int(n_pcs) <= 0:
        raise ValueError("classifier_n_pcs must be > 0")

    end_idx = int(start_idx_1based) + int(n_pcs) - 1
    if end_idx > int(dim):
        raise ValueError(
            f"Requested classifier feature range x{start_idx_1based}..x{end_idx} exceeds x1..x{dim}"
        )

    cols = ["samples"] + [f"x{i}" for i in range(int(start_idx_1based), end_idx + 1)]
    idxs = [i - 1 for i in range(int(start_idx_1based), end_idx + 1)]
    return cols, idxs


class _ClassifierResidualBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LeakyReLU(0.2),
        )
        self.skip = torch.nn.Linear(in_dim, out_dim) if in_dim != out_dim else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class _ClassifierResidualMLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.LeakyReLU(0.2),
        )
        self.res1 = _ClassifierResidualBlock(512, 512)
        self.res2 = _ClassifierResidualBlock(512, 256)
        self.res3 = _ClassifierResidualBlock(256, 128)
        self.fc_out = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_proj(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return self.fc_out(out)


def _load_classifier_checkpoint_force(
    *,
    cache_path: str,
    expected_input_size: int,
    device: str,
):
    from sklearn.preprocessing import LabelEncoder

    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    meta = payload.get("meta", {}) or {}
    state_dict = payload.get("state_dict")
    classes = list(meta.get("classes", []))
    input_size = int(meta.get("input_size", 0))
    hidden_size = int(meta.get("hidden_size", 128))
    if state_dict is None:
        raise ValueError(f"Classifier checkpoint missing state_dict: {path}")
    if not classes:
        raise ValueError(f"Classifier checkpoint missing classes metadata: {path}")
    if input_size != int(expected_input_size):
        raise ValueError(
            f"Classifier checkpoint input_size={input_size} does not match expected_input_size={expected_input_size}"
        )

    model = _ClassifierResidualMLP(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=len(classes),
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(classes)
    acc = payload.get("acc", None)
    try:
        print(f"[run_mosta_baseline_video] force-loaded classifier: {path}")
    except Exception:
        pass
    return model, label_encoder, float(acc) if acc is not None else float("nan")


def _predict_labels_for_trajectories_with_indices(
    *,
    sde_points: np.ndarray,
    ts_points: Sequence[float],
    model,
    label_encoder,
    feature_indices: Sequence[int],
    device: str,
    knn_neighbors: int = 10,
) -> list[np.ndarray]:
    from sklearn.neighbors import KNeighborsClassifier

    feat_idx = np.asarray(list(feature_indices), dtype=np.int64)
    if feat_idx.size == 0:
        raise ValueError("feature_indices must be non-empty")

    model.eval()
    model.to(device)
    predicted_labels_list: list[np.ndarray] = []
    for i, t in enumerate(ts_points):
        traj_t = np.asarray(sde_points[i], dtype=np.float32)
        n_samples = int(traj_t.shape[0])
        if n_samples == 0:
            predicted_labels_list.append(np.asarray([], dtype=str))
            continue

        feats = traj_t[:, feat_idx]
        traj_feat_t = torch.tensor(feats, dtype=torch.float32)
        samples_t = torch.full((n_samples, 1), fill_value=float(t), dtype=torch.float32)
        input_t = torch.cat((samples_t, traj_feat_t), dim=1)

        with torch.no_grad():
            outputs = model(input_t.float().to(device))
            _, predicted = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted.detach().cpu().numpy())

        coords = traj_t[:, :2]
        k = min(int(knn_neighbors), int(coords.shape[0]))
        if k <= 1:
            refined = np.asarray(predicted_labels).astype(str)
        else:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(coords, predicted_labels)
            refined = np.asarray(knn.predict(coords)).astype(str)
        predicted_labels_list.append(refined)
    return predicted_labels_list


def _downsample_for_render(
    X: np.ndarray,
    labels: np.ndarray,
    max_n: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_n is None or int(max_n) <= 0:
        return X, labels
    n = int(X.shape[0])
    if n <= int(max_n):
        return X, labels
    idx = rng.choice(n, size=int(max_n), replace=False)
    return X[idx], labels[idx]


def _compute_xy_limits(
    points: Sequence[np.ndarray],
    *,
    pad_frac: float = 0.03,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    for arr in points:
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            continue
        x_min = min(x_min, float(np.min(a[:, 0])))
        x_max = max(x_max, float(np.max(a[:, 0])))
        y_min = min(y_min, float(np.min(a[:, 1])))
        y_max = max(y_max, float(np.max(a[:, 1])))

    if not np.isfinite(x_min) or not np.isfinite(y_min):
        raise ValueError("Failed to compute axis limits from simulated points.")

    x_span = max(1e-8, x_max - x_min)
    y_span = max(1e-8, y_max - y_min)
    x_pad = x_span * float(pad_frac)
    y_pad = y_span * float(pad_frac)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = str(hex_color).strip().lstrip("#")
    if len(s) == 8:
        s = s[:6]
    if len(s) != 6:
        return (136, 136, 136)
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (136, 136, 136)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(v))) for v in rgb])


def _blend_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((r1 * (1 - t) + r2 * t, g1 * (1 - t) + g2 * t, b1 * (1 - t) + b2 * t))


def _desaturate_hex(color: str, amount: float = 0.55) -> str:
    a = float(np.clip(amount, 0.0, 1.0))
    return _blend_hex(color, "#c0c0c0", a)


def _edge_contrast_color(color: str, darken: float = 0.35) -> str:
    rgb = np.asarray(_hex_to_rgb(color), dtype=np.float32) / 255.0
    d = float(np.clip(darken, 0.0, 0.85))
    rgb = np.clip(rgb * (1.0 - d), 0.0, 1.0)
    return _rgb_to_hex((rgb[0] * 255.0, rgb[1] * 255.0, rgb[2] * 255.0))


def _label_centroids(coords: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    labels_np = np.asarray(labels).astype(str)
    out: Dict[str, np.ndarray] = {}
    for lab in np.unique(labels_np):
        mask = labels_np == lab
        if int(np.sum(mask)) > 0:
            out[str(lab)] = np.asarray(coords, dtype=np.float32)[mask].mean(axis=0)
    return out


def _focus_neighbor_centroids(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    focus_type: str,
    neighbor_pct: float = 0.1,
) -> Dict[str, np.ndarray]:
    labels_np = np.asarray(labels).astype(str)
    coords_np = np.asarray(coords, dtype=np.float32)
    if focus_type not in set(labels_np.tolist()):
        return _label_centroids(coords_np, labels_np)

    focus_mask = labels_np == str(focus_type)
    focus_centroid = coords_np[focus_mask].mean(axis=0)
    out: Dict[str, np.ndarray] = {}
    pct = float(neighbor_pct)
    if pct > 1.0:
        pct = pct / 100.0
    pct = float(np.clip(pct, 0.0, 1.0))

    for lab in np.unique(labels_np):
        mask = labels_np == lab
        if int(np.sum(mask)) == 0:
            continue
        if str(lab) == str(focus_type):
            out[str(lab)] = coords_np[mask].mean(axis=0)
            continue
        coords_l = coords_np[mask]
        d = np.linalg.norm(coords_l - focus_centroid, axis=1)
        k = max(1, int(np.ceil(pct * len(coords_l))))
        idx = np.argsort(d)[:k]
        out[str(lab)] = coords_l[idx].mean(axis=0)
    return out


def _get_focus_edges(
    comm_result: Optional[dict],
    *,
    focus_type: str,
    mode: str = "both",
    top_k: int = 4,
    min_weight: float = 0.0,
    quantile: Optional[float] = None,
) -> list[tuple[float, str, str]]:
    if not comm_result:
        return []
    M = comm_result.get("M_per_source")
    types = comm_result.get("types")
    if M is None or types is None:
        return []
    types_list = [str(x) for x in list(types)]
    if str(focus_type) not in types_list:
        return []

    idx = types_list.index(str(focus_type))
    flows: list[tuple[float, str, str]] = []
    for j, tname in enumerate(types_list):
        if j == idx:
            continue
        if mode in ("outgoing", "both"):
            flows.append((float(M[idx, j]), str(focus_type), str(tname)))
        if mode in ("incoming", "both"):
            flows.append((float(M[j, idx]), str(tname), str(focus_type)))
    flows = [(w, s, d) for (w, s, d) in flows if float(w) > float(min_weight)]
    if not flows:
        return []
    if quantile is not None:
        weights = np.array([f[0] for f in flows], dtype=float)
        q = float(np.quantile(weights, float(quantile)))
        flows = [f for f in flows if float(f[0]) >= q]
    flows.sort(key=lambda x: float(x[0]), reverse=True)
    return flows[: max(1, int(top_k))]


def _draw_edge(ax, p1, p2, color: str, lw: float, alpha: float, rad: float = 0.15) -> None:
    from matplotlib import patches
    import matplotlib.patheffects as pe

    edge_color = _edge_contrast_color(color)
    arrow = patches.FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=10 + float(lw) * 1.5,
        lw=float(lw),
        color=str(edge_color),
        alpha=float(alpha),
        connectionstyle=f"arc3,rad={float(rad)}",
        shrinkA=4,
        shrinkB=4,
    )
    arrow.set_path_effects(
        [
            pe.Stroke(linewidth=max(1.0, float(lw) + 1.5), foreground="white", alpha=min(1.0, 0.95 * float(alpha))),
            pe.Normal(),
        ]
    )
    ax.add_patch(arrow)


def _ensure_color_map_has_labels(label_to_color: Dict[str, str], labels: Sequence[str]) -> Dict[str, str]:
    missing = [str(x) for x in labels if str(x) not in label_to_color]
    if not missing:
        return label_to_color

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    extra: Dict[str, str] = {}
    uniq_missing = list(dict.fromkeys(missing))
    for idx, lab in enumerate(uniq_missing):
        rgb = cmap(idx % cmap.N)[:3]
        extra[str(lab)] = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    out = dict(label_to_color)
    out.update(extra)
    return out


def _render_frame_single_3d(
    *,
    xy: np.ndarray,
    labels: np.ndarray,
    time_value: float,
    label_to_color: Dict[str, str],
    out_png: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    elev: float,
    azim: float,
    point_size: float,
    alpha: float,
    dpi: int,
    title: str,
    show_titles: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.2, 5.8), dpi=int(dpi))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    labs = np.asarray(labels).astype(str)
    colors = [label_to_color.get(str(l), "#888888") for l in labs]
    z = np.full(xy.shape[0], fill_value=float(time_value), dtype=np.float32)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        z,
        c=colors,
        s=float(point_size),
        linewidths=0.0,
        alpha=float(alpha),
        depthshade=False,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    if bool(show_titles):
        ax.set_title(f"{title}\nt={time_value:.1f}", fontsize=float(NATURE_SCATTER_DEFAULTS["title_fontsize"]))
        fig.suptitle("MOSTA Baseline Trajectory", fontsize=float(NATURE_SCATTER_DEFAULTS["suptitle_fontsize"]))
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("time")
    ax.view_init(elev=float(elev), azim=float(azim))
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _render_frame_single_2d(
    *,
    xy: np.ndarray,
    labels: np.ndarray,
    time_value: float,
    label_to_color: Dict[str, str],
    out_png: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    point_size: float,
    alpha: float,
    dpi: int,
    panel_size: float,
    title: str,
    show_titles: bool,
    comm_result: Optional[dict],
    config: MostaBaselineVideoConfig,
) -> None:
    import matplotlib.pyplot as plt

    panel_size = float(max(2.0, panel_size))
    fig, ax = plt.subplots(1, 1, figsize=(panel_size, panel_size), dpi=int(dpi))
    fig.patch.set_facecolor("white")

    labs = np.asarray(labels).astype(str)
    base_colors = [label_to_color.get(str(l), "#888888") for l in labs]
    if bool(config.draw_focus_interactions):
        colors = [
            (
                label_to_color.get(str(l), "#888888")
                if str(l) == str(config.focus_celltype)
                else _desaturate_hex(label_to_color.get(str(l), "#888888"), config.focus_other_desaturate)
            )
            for l in labs
        ]
    else:
        colors = base_colors

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=float(point_size),
        c=colors,
        linewidths=0,
        alpha=float(alpha),
        rasterized=xy.shape[0] > 30000,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if bool(show_titles):
        ax.set_title(f"{title}\nt={time_value:.1f}", fontsize=float(NATURE_SCATTER_DEFAULTS["title_fontsize"]))
        fig.suptitle("MOSTA Baseline Trajectory", fontsize=float(NATURE_SCATTER_DEFAULTS["suptitle_fontsize"]))

    if bool(config.draw_focus_interactions):
        centroids = _focus_neighbor_centroids(
            xy,
            labs,
            focus_type=str(config.focus_celltype),
            neighbor_pct=float(config.focus_neighbor_pct),
        )
        flows = _get_focus_edges(
            comm_result,
            focus_type=str(config.focus_celltype),
            mode=str(config.focus_mode),
            top_k=int(config.focus_top_k),
            min_weight=float(config.focus_min_weight),
            quantile=config.focus_edge_quantile,
        )
        if flows:
            directed_pairs = {(s, d) for _, s, d in flows}
            bidirectional_pairs = {tuple(sorted((s, d))) for s, d in directed_pairs if (d, s) in directed_pairs}
            max_w = max(float(w) for w, _, _ in flows)
            highlight_nodes: list[str] = []
            for w, src, dst in flows:
                if src not in centroids or dst not in centroids:
                    continue
                t = float(w) / max_w if max_w > 0 else 0.0
                lw = float(config.focus_edge_width_min) + t * (
                    float(config.focus_edge_width_max) - float(config.focus_edge_width_min)
                )
                a = float(config.focus_edge_alpha_min) + t * (
                    float(config.focus_edge_alpha_max) - float(config.focus_edge_alpha_min)
                )
                edge_color = label_to_color.get(str(dst), label_to_color.get(str(src), "#111111"))
                pair = tuple(sorted((src, dst)))
                if pair in bidirectional_pairs:
                    rad = float(config.focus_edge_curve_bi)
                else:
                    rad = float(config.focus_edge_curve) if str(src) < str(dst) else -float(config.focus_edge_curve)
                _draw_edge(ax, centroids[src], centroids[dst], edge_color, lw=lw, alpha=a, rad=rad)
                highlight_nodes.extend([str(src), str(dst)])

            if bool(config.focus_show_endpoints):
                uniq_nodes = [x for x in dict.fromkeys(highlight_nodes) if x in centroids]
                if uniq_nodes:
                    node_xy = np.asarray([centroids[x] for x in uniq_nodes], dtype=np.float32)
                    node_colors = [label_to_color.get(x, "#111111") for x in uniq_nodes]
                    ax.scatter(
                        node_xy[:, 0],
                        node_xy[:, 1],
                        s=float(max(1.0, config.focus_endpoint_size)),
                        c=node_colors,
                        linewidths=float(max(0.0, config.focus_endpoint_linewidth)),
                        edgecolors=str(config.focus_endpoint_edgecolor),
                        alpha=float(np.clip(config.focus_endpoint_alpha, 0.0, 1.0)),
                        zorder=6,
                    )

        ax.text(
            0.02,
            0.98,
            f"Focus: {config.focus_celltype} ({config.focus_mode})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=float(NATURE_SCATTER_DEFAULTS["legend_fontsize"]),
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MOSTA baseline-only video pipeline")
    parser.add_argument("--output-name", default=MostaBaselineVideoConfig.output_name)
    parser.add_argument("--time-start", type=float, default=MostaBaselineVideoConfig.time_start)
    parser.add_argument("--time-end", type=float, default=MostaBaselineVideoConfig.time_end)
    parser.add_argument("--time-step", type=float, default=MostaBaselineVideoConfig.time_step)
    parser.add_argument("--sde-n-samples", type=int, default=MostaBaselineVideoConfig.sde_n_samples)
    parser.add_argument("--split-sde-dt", type=float, default=MostaBaselineVideoConfig.split_sde_dt)
    parser.add_argument("--split-sigma", type=float, default=MostaBaselineVideoConfig.split_sigma)
    parser.add_argument("--split-growth-alpha", type=float, default=MostaBaselineVideoConfig.split_growth_alpha)
    parser.add_argument(
        "--spatial-warp-to-observed",
        action=argparse.BooleanOptionalAction,
        default=MostaBaselineVideoConfig.spatial_warp_to_observed,
    )
    parser.add_argument("--spatial-warp-k", type=int, default=MostaBaselineVideoConfig.spatial_warp_k)
    parser.add_argument("--spatial-warp-eps", type=float, default=MostaBaselineVideoConfig.spatial_warp_eps)
    parser.add_argument("--interaction-m", type=int, default=MostaBaselineVideoConfig.interaction_m)
    parser.add_argument("--classifier-epochs", type=int, default=MostaBaselineVideoConfig.classifier_epochs)
    parser.add_argument("--classifier-hidden", type=int, default=MostaBaselineVideoConfig.classifier_hidden)
    parser.add_argument("--classifier-n-pcs", type=int, default=MostaBaselineVideoConfig.classifier_n_pcs)
    parser.add_argument(
        "--classifier-feature-start",
        type=int,
        default=MostaBaselineVideoConfig.classifier_feature_start,
    )
    parser.add_argument(
        "--classifier-best-metric",
        choices=["accuracy", "bacc"],
        default=MostaBaselineVideoConfig.classifier_best_metric,
    )
    parser.add_argument(
        "--classifier-train-on-full-data",
        action=argparse.BooleanOptionalAction,
        default=MostaBaselineVideoConfig.classifier_train_on_full_data,
    )
    parser.add_argument(
        "--classifier-cache",
        action=argparse.BooleanOptionalAction,
        default=MostaBaselineVideoConfig.classifier_cache,
    )
    parser.add_argument("--classifier-cache-path", default=None)
    parser.add_argument("--classifier-cache-dir", default=None)
    parser.add_argument("--classifier-cache-tag", default=None)
    parser.add_argument(
        "--classifier-force-load",
        action=argparse.BooleanOptionalAction,
        default=MostaBaselineVideoConfig.classifier_force_load,
    )
    parser.add_argument("--video-style", choices=["fixed_2d", "fixed_3d"], default=MostaBaselineVideoConfig.video_style)
    parser.add_argument("--video-point-subsample", type=int, default=MostaBaselineVideoConfig.video_point_subsample)
    parser.add_argument("--gif-fps", type=int, default=MostaBaselineVideoConfig.gif_fps)
    parser.add_argument("--frame-dpi", type=int, default=MostaBaselineVideoConfig.frame_dpi)
    parser.add_argument("--point-size", type=float, default=MostaBaselineVideoConfig.point_size)
    parser.add_argument("--point-alpha", type=float, default=MostaBaselineVideoConfig.point_alpha)
    parser.add_argument(
        "--show-titles",
        action=argparse.BooleanOptionalAction,
        default=MostaBaselineVideoConfig.show_titles,
    )
    parser.add_argument(
        "--draw-focus-interactions",
        action=argparse.BooleanOptionalAction,
        default=MostaBaselineVideoConfig.draw_focus_interactions,
    )
    parser.add_argument("--focus-celltype", default=MostaBaselineVideoConfig.focus_celltype)
    parser.add_argument(
        "--focus-mode",
        choices=["incoming", "outgoing", "both"],
        default=MostaBaselineVideoConfig.focus_mode,
    )
    parser.add_argument("--focus-top-k", type=int, default=MostaBaselineVideoConfig.focus_top_k)
    parser.add_argument("--focus-min-weight", type=float, default=MostaBaselineVideoConfig.focus_min_weight)
    parser.add_argument("--focus-edge-quantile", type=float, default=MostaBaselineVideoConfig.focus_edge_quantile)
    parser.add_argument("--focus-edge-width-min", type=float, default=MostaBaselineVideoConfig.focus_edge_width_min)
    parser.add_argument("--focus-edge-width-max", type=float, default=MostaBaselineVideoConfig.focus_edge_width_max)
    parser.add_argument("--focus-edge-alpha-min", type=float, default=MostaBaselineVideoConfig.focus_edge_alpha_min)
    parser.add_argument("--focus-edge-alpha-max", type=float, default=MostaBaselineVideoConfig.focus_edge_alpha_max)
    parser.add_argument("--focus-edge-curve", type=float, default=MostaBaselineVideoConfig.focus_edge_curve)
    parser.add_argument("--focus-edge-curve-bi", type=float, default=MostaBaselineVideoConfig.focus_edge_curve_bi)
    parser.add_argument("--focus-neighbor-pct", type=float, default=MostaBaselineVideoConfig.focus_neighbor_pct)
    parser.add_argument(
        "--focus-other-desaturate",
        type=float,
        default=MostaBaselineVideoConfig.focus_other_desaturate,
    )
    parser.add_argument(
        "--focus-show-endpoints",
        action=argparse.BooleanOptionalAction,
        default=MostaBaselineVideoConfig.focus_show_endpoints,
    )
    parser.add_argument("--focus-endpoint-size", type=float, default=MostaBaselineVideoConfig.focus_endpoint_size)
    parser.add_argument("--focus-endpoint-alpha", type=float, default=MostaBaselineVideoConfig.focus_endpoint_alpha)
    parser.add_argument("--focus-endpoint-edgecolor", default=MostaBaselineVideoConfig.focus_endpoint_edgecolor)
    parser.add_argument(
        "--focus-endpoint-linewidth",
        type=float,
        default=MostaBaselineVideoConfig.focus_endpoint_linewidth,
    )
    parser.add_argument("--attention-stride", type=int, default=MostaBaselineVideoConfig.attention_stride)
    parser.add_argument("--attention-max-cells", type=int, default=MostaBaselineVideoConfig.attention_max_cells)
    parser.add_argument(
        "--attention-use-real-observed",
        action=argparse.BooleanOptionalAction,
        default=MostaBaselineVideoConfig.attention_use_real_observed,
    )
    parser.add_argument(
        "--axis-limit-mode",
        choices=["global", "per_timepoint"],
        default=MostaBaselineVideoConfig.axis_limit_mode,
    )
    parser.add_argument("--axis-pad-frac", type=float, default=MostaBaselineVideoConfig.axis_pad_frac)
    parser.add_argument("--panel-size", type=float, default=MostaBaselineVideoConfig.panel_size)
    parser.add_argument("--camera-elev", type=float, default=MostaBaselineVideoConfig.camera_elev)
    parser.add_argument("--camera-azim", type=float, default=MostaBaselineVideoConfig.camera_azim)
    parser.add_argument("--random-seed", type=int, default=MostaBaselineVideoConfig.random_seed)
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=True)
    return parser


def resolve_mosta_baseline_video_output_dir(config: MostaBaselineVideoConfig) -> Path:
    return MOSTA_RESULTS_ROOT / config.output_name


def build_mosta_baseline_video_config_from_args(args: argparse.Namespace) -> MostaBaselineVideoConfig:
    values = vars(args).copy()
    return MostaBaselineVideoConfig(**values)


def run_mosta_baseline_video(
    config: MostaBaselineVideoConfig,
    *,
    context: MostaContext | None = None,
) -> MostaBaselineVideoResult:
    _seed_everything(config.random_seed)
    try:
        ctx = load_mosta_context(random_seed=config.random_seed) if context is None else context
        from CytoBridge.tl import (
            load_label_to_color,
            simulate_piecewise_spatially_warped_split,
            simulate_sde_points_split_from_x0,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize the MOSTA baseline video runtime. "
            "This workflow needs the same CytoBridge/graph-model environment as the original MOSTA run "
            "(for example torch_geometric and the legacy dynamical-model dependencies)."
        ) from exc

    output_dir = resolve_mosta_baseline_video_output_dir(config)
    frames_dir = output_dir / "frames"
    branch_dir = output_dir / "branch_baseline"
    if config.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    branch_dir.mkdir(parents=True, exist_ok=True)

    df = ctx.df.copy()
    annotation_key = ctx.assets.annotation_key
    feature_cols_full = [f"x{i}" for i in range(1, int(ctx.dim) + 1)]
    observed_times = sorted(float(x) for x in df["samples"].unique())
    ts_points = _build_ts_points(float(config.time_start), float(config.time_end), float(config.time_step))
    t_start_obs = min(observed_times, key=lambda x: abs(x - float(config.time_start)))
    if not _float_close(t_start_obs, float(config.time_start)):
        raise ValueError(
            f"time_start={config.time_start} must match an observed time. Available observed times: {observed_times}"
        )

    df_t0 = df[df["samples"] == float(t_start_obs)]
    X_base_pool = df_t0[feature_cols_full].values.astype(np.float32)
    y_base_pool = df_t0[annotation_key].astype(str).values
    n_init = int(min(int(config.sde_n_samples), int(X_base_pool.shape[0])))
    if n_init <= 0:
        raise ValueError("No valid initial cells available for baseline simulation.")

    rng = np.random.default_rng(42 if config.random_seed is None else int(config.random_seed))
    idx0 = rng.choice(int(X_base_pool.shape[0]), size=n_init, replace=False)
    X0 = X_base_pool[idx0]
    y0 = y_base_pool[idx0]
    np.savez_compressed(branch_dir / "init_pool.npz", X0=X0, labels=np.asarray(y0).astype(str))

    runtime = ctx.runtime
    t_sim0 = time.perf_counter()
    if bool(config.spatial_warp_to_observed):
        sde_baseline = simulate_piecewise_spatially_warped_split(
            x0=X0,
            f_net=runtime.f_net,
            score_net=runtime.score_net,
            observed_time_points=observed_times,
            ts_points=ts_points,
            df=df,
            feature_cols_full=feature_cols_full,
            label_col=annotation_key,
            dt=float(config.split_sde_dt),
            sigma=float(config.split_sigma),
            sigma_by_dim=None,
            growth_alpha=float(config.split_growth_alpha),
            interaction_m=int(config.interaction_m),
            device=ctx.device,
            rng=np.random.default_rng(43 if config.random_seed is None else int(config.random_seed) + 1),
            k=int(config.spatial_warp_k),
            eps=float(config.spatial_warp_eps),
        )
    else:
        sde_baseline = simulate_sde_points_split_from_x0(
            x0=X0,
            f_net=runtime.f_net,
            score_net=runtime.score_net,
            ts_points=ts_points,
            dt=float(config.split_sde_dt),
            sigma=float(config.split_sigma),
            sigma_by_dim=None,
            growth_alpha=float(config.split_growth_alpha),
            interaction_m=int(config.interaction_m),
            device=ctx.device,
            verbose=True,
        )
    print(f"Baseline simulation finished in {time.perf_counter() - t_sim0:.1f}s")

    classifier_feature_cols, classifier_feature_indices = _build_classifier_feature_spec(
        dim=int(ctx.dim),
        start_idx_1based=int(config.classifier_feature_start),
        n_pcs=int(config.classifier_n_pcs),
    )
    classifier_cache_path = str(resolve_mosta_classifier_cache_path(config.classifier_cache_path))
    classifier_cache_dir = config.classifier_cache_dir or str(ctx.assets.classifier_cache_path.parent)
    if bool(config.classifier_force_load):
        clf_model, label_encoder, clf_acc = _load_classifier_checkpoint_force(
            cache_path=classifier_cache_path,
            expected_input_size=len(classifier_feature_cols),
            device=ctx.device,
        )
    else:
        clf_model, label_encoder, clf_acc = train_mlp_classifier(
            df=df,
            feature_cols=classifier_feature_cols,
            label_col=annotation_key,
            hidden_size=int(config.classifier_hidden),
            epochs=int(config.classifier_epochs),
            lr=1e-3,
            test_size=0.1,
            seed=42 if config.random_seed is None else int(config.random_seed),
            cache_path=classifier_cache_path if bool(config.classifier_cache) else None,
            cache_dir=classifier_cache_dir if bool(config.classifier_cache) else None,
            cache_tag=config.classifier_cache_tag,
            df_source_path=str(ctx.assets.data_csv),
            reuse_if_possible=bool(config.classifier_cache),
            progress=True,
            device=ctx.device,
            best_epoch_metric=str(config.classifier_best_metric),
            train_on_full_data=bool(config.classifier_train_on_full_data),
        )

    pred_baseline = _predict_labels_for_trajectories_with_indices(
        sde_points=sde_baseline,
        ts_points=ts_points,
        model=clf_model,
        label_encoder=label_encoder,
        feature_indices=classifier_feature_indices,
        device=ctx.device,
        knn_neighbors=10,
    )
    np.savez_compressed(
        branch_dir / "trajectory_pred.npz",
        ts=np.asarray(ts_points, dtype=np.float32),
        labels=np.asarray(pred_baseline, dtype=object),
    )

    label_to_color = load_label_to_color(
        df[annotation_key].astype(str).values,
        label_color_json=str(ctx.assets.label_color_json) if ctx.assets.label_color_json else None,
        color_h5ad=ctx.assets.color_h5ad,
        annotation_key=annotation_key,
    )
    all_pred_labels: list[str] = []
    for arr in pred_baseline:
        all_pred_labels.extend(np.asarray(arr).astype(str).tolist())
    label_to_color = _ensure_color_map_has_labels(label_to_color, all_pred_labels)
    label_color_path = output_dir / "label_to_color.json"
    label_color_path.write_text(json.dumps(label_to_color, indent=2, ensure_ascii=False), encoding="utf-8")

    comm_path: Path | None = None
    comm_sampling_csv: Path | None = None
    all_time_communications: Dict[str, dict] = {}
    comm_source_by_time: Dict[str, str] = {}
    comm_cells_by_time: Dict[str, int] = {}
    if bool(config.draw_focus_interactions):
        attn_dir = output_dir / "attention"
        attn_dir.mkdir(parents=True, exist_ok=True)
        max_cells = int(config.attention_max_cells)
        if max_cells <= 0:
            raise ValueError("attention_max_cells must be > 0")

        for idx, t in enumerate(ts_points):
            if int(config.attention_stride) > 1 and (idx % int(config.attention_stride) != 0):
                continue

            source = "simulated"
            if bool(config.attention_use_real_observed) and any(_float_close(float(t), t_obs) for t_obs in observed_times):
                df_t = df[df["samples"] == float(t)]
                X_t = df_t[feature_cols_full].values.astype(np.float32)
                labels_t = df_t[annotation_key].astype(str).values
                source = "observed"
            else:
                X_t = np.asarray(sde_baseline[idx], dtype=np.float32)
                labels_t = np.asarray(pred_baseline[idx]).astype(str)

            n_before = int(X_t.shape[0])
            if n_before > max_cells:
                take = rng.choice(n_before, size=max_cells, replace=False)
                X_t = X_t[take]
                labels_t = labels_t[take]
            n_after = int(X_t.shape[0])
            key = str(float(t))
            comm_source_by_time[key] = source
            comm_cells_by_time[key] = n_after

            print(f"[focus-comm] time={key} source={source} cells={n_after}/{n_before}")
            adata_like = SimpleNamespace(X=X_t)
            attn_out = save_interpolated_attention(
                adata_like,
                time_value=float(t),
                f_net=runtime.model,
                device=ctx.device,
                out_dir=str(attn_dir),
                save_dense_matrix=False,
            )
            comm = analyze_attention_by_celltype(
                edge_index=attn_out["edge_index"],
                attn=attn_out["attn_mean"],
                labels=labels_t,
                spatial_coord=X_t[:, :2],
                time_title=key,
                remove_self_loop=False,
                winsor_quantile=0.995,
                distance_bins=None,
                n_permutations=0,
                plot=False,
            )
            all_time_communications[key] = comm

        if all_time_communications:
            computed_keys = sorted(all_time_communications.keys(), key=lambda x: float(x))

            def _nearest_key(target: str) -> str:
                return min(computed_keys, key=lambda k: abs(float(k) - float(target)))

            for t in ts_points:
                key = str(float(t))
                if key not in all_time_communications:
                    nearest = _nearest_key(key)
                    all_time_communications[key] = all_time_communications[nearest]
                    comm_source_by_time[key] = f"nearest:{comm_source_by_time.get(nearest, 'unknown')}"
                    comm_cells_by_time[key] = int(comm_cells_by_time.get(nearest, 0))

            comm_path = output_dir / "all_time_communications.pkl"
            with comm_path.open("wb") as f:
                pickle.dump(all_time_communications, f)

            comm_rows = []
            for key in sorted(all_time_communications.keys(), key=lambda x: float(x)):
                comm_rows.append(
                    {
                        "time": float(key),
                        "source": str(comm_source_by_time.get(key, "unknown")),
                        "n_cells_used": int(comm_cells_by_time.get(key, 0)),
                    }
                )
            comm_sampling_csv = output_dir / "communication_sampling_by_time.csv"
            pd.DataFrame(comm_rows).to_csv(comm_sampling_csv, index=False)

    xlim_global, ylim_global = _compute_xy_limits(sde_baseline, pad_frac=float(config.axis_pad_frac))
    zlim = (float(config.time_start), float(config.time_end)) if config.video_style == "fixed_3d" else None

    frame_paths: list[Path] = []
    baseline_counts_rows: list[dict[str, float | int]] = []
    render_rng = np.random.default_rng((0 if config.random_seed is None else int(config.random_seed)) + 1024)
    t_render0 = time.perf_counter()
    for i, t in enumerate(ts_points):
        Xb = np.asarray(sde_baseline[i], dtype=np.float32)
        yb = np.asarray(pred_baseline[i]).astype(str)
        if config.axis_limit_mode == "per_timepoint":
            xlim_frame, ylim_frame = _compute_xy_limits([Xb[:, :2]], pad_frac=float(config.axis_pad_frac))
        else:
            xlim_frame, ylim_frame = xlim_global, ylim_global

        Xb_vis, yb_vis = _downsample_for_render(Xb, yb, config.video_point_subsample, render_rng)
        frame_path = frames_dir / f"frame_{i:03d}.png"
        if config.video_style == "fixed_2d":
            comm_t = all_time_communications.get(str(float(t))) if bool(config.draw_focus_interactions) else None
            _render_frame_single_2d(
                xy=Xb_vis[:, :2],
                labels=yb_vis,
                time_value=float(t),
                label_to_color=label_to_color,
                out_png=frame_path,
                xlim=xlim_frame,
                ylim=ylim_frame,
                point_size=float(config.point_size),
                alpha=float(config.point_alpha),
                dpi=int(config.frame_dpi),
                panel_size=float(config.panel_size),
                title="Baseline",
                show_titles=bool(config.show_titles),
                comm_result=comm_t,
                config=config,
            )
        else:
            if zlim is None:
                raise ValueError("zlim must be set for fixed_3d rendering.")
            _render_frame_single_3d(
                xy=Xb_vis[:, :2],
                labels=yb_vis,
                time_value=float(t),
                label_to_color=label_to_color,
                out_png=frame_path,
                xlim=xlim_frame,
                ylim=ylim_frame,
                zlim=zlim,
                elev=float(config.camera_elev),
                azim=float(config.camera_azim),
                point_size=float(config.point_size),
                alpha=float(config.point_alpha),
                dpi=int(config.frame_dpi),
                title="Baseline",
                show_titles=bool(config.show_titles),
            )
        frame_paths.append(frame_path)
        baseline_counts_rows.append({"time": float(t), "n_baseline": int(Xb.shape[0])})
        print(f"Rendered baseline frame {i + 1}/{len(ts_points)} | t={t:.3f}")
    print(f"Baseline frame rendering finished in {time.perf_counter() - t_render0:.1f}s")

    baseline_counts_csv = output_dir / "baseline_counts_by_time.csv"
    pd.DataFrame(baseline_counts_rows).sort_values("time").to_csv(baseline_counts_csv, index=False)

    gif_path = output_dir / (
        f"baseline_{_tag_float(config.time_start)}_to_{_tag_float(config.time_end)}_step{_tag_float(config.time_step)}.gif"
    )
    duration = 1.0 / max(1, int(config.gif_fps))
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, images, duration=duration, loop=0)

    summary = {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "render_mode": "baseline_only",
        "data_csv": str(ctx.assets.data_csv),
        "annotation_key": annotation_key,
        "time_start": float(config.time_start),
        "time_end": float(config.time_end),
        "time_step": float(config.time_step),
        "timepoints": [float(x) for x in ts_points],
        "n_frames": int(len(ts_points)),
        "gif_path": str(gif_path),
        "frames_dir": str(frames_dir),
        "frame_count": int(len(frame_paths)),
        "baseline_counts_by_time_csv": str(baseline_counts_csv),
        "observed_times": [float(x) for x in observed_times],
        "n_observed_rows": int(df.shape[0]),
        "n_t0_pool_baseline": int(X_base_pool.shape[0]),
        "n_init_baseline": int(n_init),
        "classifier": {
            "n_pcs": int(config.classifier_n_pcs),
            "feature_start_x": int(config.classifier_feature_start),
            "feature_end_x": int(config.classifier_feature_start + config.classifier_n_pcs - 1),
            "epochs": int(config.classifier_epochs),
            "hidden": int(config.classifier_hidden),
            "best_metric": str(config.classifier_best_metric),
            "train_on_full_data": bool(config.classifier_train_on_full_data),
            "cache_path": str(classifier_cache_path),
            "accuracy": None if clf_acc is None or not np.isfinite(clf_acc) else float(clf_acc),
        },
        "simulation": {
            "split_sde_dt": float(config.split_sde_dt),
            "split_sigma": float(config.split_sigma),
            "split_growth_alpha": float(config.split_growth_alpha),
            "spatial_warp_to_observed": bool(config.spatial_warp_to_observed),
            "spatial_warp_k": int(config.spatial_warp_k),
            "spatial_warp_eps": float(config.spatial_warp_eps),
            "spatial_warp_mode": "piecewise_rerun" if bool(config.spatial_warp_to_observed) else "off",
            "interaction_m": int(config.interaction_m),
            "sde_n_samples_requested": int(config.sde_n_samples),
            "device": str(ctx.device),
        },
        "camera": {
            "style": str(config.video_style),
            "layout": "single_panel",
            "elev": float(config.camera_elev),
            "azim": float(config.camera_azim),
            "axis_limit_mode": str(config.axis_limit_mode),
            "axis_pad_frac": float(config.axis_pad_frac),
            "xlim_global": [float(xlim_global[0]), float(xlim_global[1])],
            "ylim_global": [float(ylim_global[0]), float(ylim_global[1])],
            "zlim": [float(zlim[0]), float(zlim[1])] if zlim is not None else None,
            "point_size": float(config.point_size),
            "point_alpha": float(config.point_alpha),
            "panel_size": float(config.panel_size),
            "video_point_subsample": int(config.video_point_subsample),
            "gif_fps": int(config.gif_fps),
            "frame_dpi": int(config.frame_dpi),
            "show_titles": bool(config.show_titles),
        },
        "communication": {
            "draw_focus_interactions": bool(config.draw_focus_interactions),
            "focus_celltype": str(config.focus_celltype),
            "focus_mode": str(config.focus_mode),
            "focus_top_k": int(config.focus_top_k),
            "focus_min_weight": float(config.focus_min_weight),
            "focus_edge_quantile": None
            if config.focus_edge_quantile is None
            else float(config.focus_edge_quantile),
            "focus_edge_width": [float(config.focus_edge_width_min), float(config.focus_edge_width_max)],
            "focus_edge_alpha": [float(config.focus_edge_alpha_min), float(config.focus_edge_alpha_max)],
            "focus_edge_curve": float(config.focus_edge_curve),
            "focus_edge_curve_bi": float(config.focus_edge_curve_bi),
            "focus_neighbor_pct": float(config.focus_neighbor_pct),
            "focus_other_desaturate": float(config.focus_other_desaturate),
            "focus_show_endpoints": bool(config.focus_show_endpoints),
            "focus_endpoint_size": float(config.focus_endpoint_size),
            "focus_endpoint_alpha": float(config.focus_endpoint_alpha),
            "focus_endpoint_edgecolor": str(config.focus_endpoint_edgecolor),
            "focus_endpoint_linewidth": float(config.focus_endpoint_linewidth),
            "attention_stride": int(config.attention_stride),
            "attention_max_cells": int(config.attention_max_cells),
            "attention_use_real_observed": bool(config.attention_use_real_observed),
            "all_time_communications_pkl": str(comm_path) if comm_path is not None else None,
            "communication_sampling_by_time_csv": str(comm_sampling_csv) if comm_sampling_csv is not None else None,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return MostaBaselineVideoResult(
        output_dir=output_dir,
        frames_dir=frames_dir,
        gif_path=gif_path,
        summary_path=summary_path,
        baseline_counts_csv=baseline_counts_csv,
        label_color_json=label_color_path,
        all_time_communications_pkl=comm_path,
        communication_sampling_csv=comm_sampling_csv,
        frame_count=len(frame_paths),
        timepoints=tuple(float(x) for x in ts_points),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = build_mosta_baseline_video_config_from_args(args)
    result = run_mosta_baseline_video(config)
    print("Saved GIF:", result.gif_path)
    print("Saved summary:", result.summary_path)


__all__ = [
    "MostaBaselineVideoConfig",
    "MostaBaselineVideoResult",
    "build_mosta_baseline_video_config_from_args",
    "main",
    "resolve_mosta_baseline_video_output_dir",
    "run_mosta_baseline_video",
]
