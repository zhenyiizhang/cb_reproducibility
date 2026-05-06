from __future__ import annotations

import json
import os
import random
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from evaluation.arista_code.arista_helpers import (
    load_arista_df,
    load_config,
    load_models,
    predict_labels_for_trajectories,
    simulate_sde_points,
    simulate_sde_points_split,
    train_mlp_classifier,
)


def set_global_seed(random_seed: Optional[int]) -> None:
    if random_seed is None:
        return
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_label_to_color(
    labels: np.ndarray,
    label_color_json: Optional[str] = None,
    color_h5ad: Optional[str] = None,
    annotation_key: str = "Annotation",
) -> Dict[str, str]:
    if label_color_json and os.path.exists(label_color_json):
        with open(label_color_json, "r", encoding="utf-8") as f:
            return json.load(f)

    if color_h5ad and os.path.exists(color_h5ad):
        try:
            import anndata as ad

            adata = ad.read_h5ad(color_h5ad, backed="r")
            try:
                key = annotation_key if annotation_key in adata.obs else None
                if key is None and annotation_key.lower() in adata.obs:
                    key = annotation_key.lower()
                if key:
                    colors_key = f"{key}_colors"
                    colors = adata.uns.get(colors_key)
                    if colors is not None:
                        categories = (
                            adata.obs[key].cat.categories
                            if hasattr(adata.obs[key], "cat")
                            else sorted(adata.obs[key].unique())
                        )
                        return {str(c): str(col) for c, col in zip(categories, colors)}
            finally:
                try:
                    adata.file.close()
                except Exception:
                    pass
        except Exception as exc:
            print(f"Color map load failed from {color_h5ad}: {exc}")

    import matplotlib.pyplot as plt

    unique_labels = list(dict.fromkeys([str(x) for x in labels]))
    cmap = plt.get_cmap("tab20")
    out = {}
    for idx, lab in enumerate(unique_labels):
        rgb = cmap(idx % cmap.N)[:3]
        out[str(lab)] = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    return out


def save_timepoint_snapshots(
    adata_dict,
    time_keys,
    label_to_color,
    annotation_key="Annotation",
    snapshot_dir=None,
    background_color=None,
    font_color="#1a1a1a",
    snapshot_point_size=2.5,
    snapshot_alpha=0.9,
    mosaic_cols=4,
    mosaic_cell_size=2.2,
    mosaic_show_title=True,
    save_pdf=True,
    panel_title_map: Optional[Dict[str, str]] = None,
):
    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bg = background_color or "white"
    os.makedirs(snapshot_dir, exist_ok=True)

    panels = []
    for tk in time_keys:
        ad = adata_dict[tk]
        coords = np.asarray(ad.obsm["spatial"])
        labels = ad.obs[annotation_key].astype(str).values
        panels.append(
            {
                "tk": tk,
                "title": panel_title_map.get(str(tk), f"t = {tk}") if panel_title_map else f"t = {tk}",
                "coords": coords,
                "labels": np.asarray(labels).astype(str),
            }
        )

    for panel in panels:
        coords = np.asarray(panel["coords"])
        labels = np.asarray(panel["labels"]).astype(str)
        colors = [label_to_color.get(str(l), "#888888") for l in labels]
        rasterized = coords.shape[0] > 30000

        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(snapshot_point_size),
            c=colors,
            linewidths=0,
            alpha=float(snapshot_alpha),
            rasterized=rasterized,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(panel["title"], color=font_color, fontsize=12, pad=6)

        tk = panel["tk"]
        out_path = os.path.join(snapshot_dir, f"time_{tk}.svg")
        fig.savefig(out_path, format="svg", facecolor=bg, bbox_inches="tight")
        if save_pdf:
            out_path = os.path.join(snapshot_dir, f"time_{tk}.pdf")
            fig.savefig(out_path, format="pdf", facecolor=bg, bbox_inches="tight")
        plt.close(fig)

    n_panels = len(panels)
    cols = max(1, int(mosaic_cols))
    rows = math.ceil(n_panels / cols)
    fig_w = cols * float(mosaic_cell_size)
    fig_h = rows * float(mosaic_cell_size)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor(bg)
    axes = axes if isinstance(axes, np.ndarray) else np.array([[axes]])
    axes = axes.reshape(rows, cols)

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        coords = np.asarray(panel["coords"])
        labels = np.asarray(panel["labels"]).astype(str)
        colors = [label_to_color.get(str(l), "#888888") for l in labels]
        rasterized = coords.shape[0] > 30000

        ax.set_facecolor(bg)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(snapshot_point_size),
            c=colors,
            linewidths=0,
            alpha=float(snapshot_alpha),
            rasterized=rasterized,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if mosaic_show_title:
            ax.set_title(panel["tk"], color=font_color, fontsize=8, pad=3)

    for idx in range(n_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
        axes[r, c].set_facecolor(bg)

    mosaic_path = os.path.join(snapshot_dir, "timepoint_mosaic.svg")
    fig.savefig(mosaic_path, format="svg", facecolor=bg, bbox_inches="tight")
    plt.close(fig)

    legend_path = os.path.join(snapshot_dir, "label_legend.svg")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=label_to_color[k], markersize=6, label=k)
        for k in label_to_color.keys()
    ]

    fig, ax = plt.subplots(figsize=(4, 6), facecolor=bg)
    ax.set_facecolor(bg)
    ax.legend(handles=handles, loc="center left", frameon=False, labelcolor=font_color)
    ax.axis("off")
    fig.savefig(legend_path, format="svg", facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def save_timepoint_mosaic_panels(
    *,
    panels: Sequence[dict],
    label_to_color: Dict[str, str],
    snapshot_dir: str,
    background_color: Optional[str] = None,
    font_color: str = "#1a1a1a",
    snapshot_point_size: float = 2.5,
    snapshot_alpha: float = 0.9,
    mosaic_cols: int = 4,
    mosaic_cell_size: float = 2.2,
    mosaic_show_title: bool = True,
    output_name: str = "timepoint_mosaic.svg",
):
    import math
    import matplotlib.pyplot as plt

    bg = background_color or "white"
    os.makedirs(snapshot_dir, exist_ok=True)

    n_panels = len(panels)
    cols = max(1, int(mosaic_cols))
    rows = max(1, math.ceil(n_panels / cols))
    fig_w = cols * float(mosaic_cell_size)
    fig_h = rows * float(mosaic_cell_size)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor(bg)
    axes = axes if isinstance(axes, np.ndarray) else np.array([[axes]])
    axes = axes.reshape(rows, cols)

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        coords = np.asarray(panel["coords"])
        labels = np.asarray(panel["labels"]).astype(str)
        colors = [label_to_color.get(str(l), "#888888") for l in labels]
        rasterized = coords.shape[0] > 30000

        ax.set_facecolor(bg)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(snapshot_point_size),
            c=colors,
            linewidths=0,
            alpha=float(snapshot_alpha),
            rasterized=rasterized,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if mosaic_show_title:
            ax.set_title(str(panel["title"]), color=font_color, fontsize=8, pad=3)

    for idx in range(n_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
        axes[r, c].set_facecolor(bg)

    mosaic_path = os.path.join(snapshot_dir, output_name)
    fig.savefig(mosaic_path, format="svg", facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def _select_evenly_spaced(values: Sequence[float], n_keep: int) -> list[float]:
    if n_keep >= len(values):
        return list(values)
    idx = np.linspace(0, len(values) - 1, num=n_keep)
    idx = [int(round(i)) for i in idx]
    seen = set()
    idx_unique = []
    for i in idx:
        if i not in seen:
            idx_unique.append(i)
            seen.add(i)
    if len(idx_unique) < n_keep:
        for i in range(len(values)):
            if i in seen:
                continue
            idx_unique.append(i)
            if len(idx_unique) == n_keep:
                break
    idx_unique = sorted(idx_unique)
    return [values[i] for i in idx_unique]


def _sample_observed_x0(
    df: pd.DataFrame,
    *,
    time_value: float,
    feature_cols: Sequence[str],
    label_col: str,
    n_samples_cap: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    subset = df[df["samples"] == float(time_value)]
    X = subset[list(feature_cols)].values.astype(np.float32)
    labels = subset[label_col].astype(str).values
    if n_samples_cap is None:
        return X, labels
    cap = int(n_samples_cap)
    if cap <= 0:
        raise ValueError("n_samples_cap must be > 0 when provided")
    if X.shape[0] <= cap:
        return X, labels
    idx = rng.choice(X.shape[0], size=cap, replace=False)
    return X[idx], labels[idx]


def _compute_spatial_warp_displacements(
    query_xy: np.ndarray,
    anchor_source_xy: np.ndarray,
    anchor_target_xy: np.ndarray,
    *,
    k: int,
    eps: float,
) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    query_xy = np.asarray(query_xy, dtype=np.float32)
    anchor_source_xy = np.asarray(anchor_source_xy, dtype=np.float32)
    anchor_target_xy = np.asarray(anchor_target_xy, dtype=np.float32)

    if anchor_source_xy.shape[0] == 0 or anchor_target_xy.shape[0] == 0:
        return np.zeros((query_xy.shape[0], 2), dtype=np.float32)

    target_nn = NearestNeighbors(n_neighbors=1)
    target_nn.fit(anchor_target_xy)
    _, target_idx = target_nn.kneighbors(anchor_source_xy)
    anchor_disp = anchor_target_xy[target_idx[:, 0]] - anchor_source_xy

    k_eff = max(1, min(int(k), anchor_source_xy.shape[0]))
    source_nn = NearestNeighbors(n_neighbors=k_eff)
    source_nn.fit(anchor_source_xy)
    dists, src_idx = source_nn.kneighbors(query_xy[:, :2])

    weights = 1.0 / np.maximum(dists, float(eps))
    weights /= weights.sum(axis=1, keepdims=True)
    disp = (anchor_disp[src_idx] * weights[..., None]).sum(axis=1)
    return disp.astype(np.float32, copy=False)


def _simulate_sde_points_split_from_x0(
    *,
    x0: np.ndarray,
    f_net,
    score_net,
    ts_points: Sequence[float],
    dt: float,
    sigma: float,
    growth_alpha: float,
    interaction_m: int,
    device: str,
    verbose: bool = True,
) -> np.ndarray:
    import torch
    from DeepRUOT.interaction import cal_interaction, euler_sdeint_split

    x0_t = torch.tensor(np.asarray(x0, dtype=np.float32), device=device)
    lnw0 = torch.log(torch.ones(x0_t.shape[0], 1, device=device) / x0_t.shape[0])
    initial_state = (x0_t, lnw0)

    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, ode_drift, g, score, interaction, sigma):
            super().__init__()
            self.drift = ode_drift
            self.score = score
            self.interaction = interaction
            self.g_net = g
            self.sigma = float(sigma)

        def f(self, t, y):
            z, lnw = y
            with torch.no_grad():
                drift = self.drift(t, z)
                dlnw = self.g_net(t, z) * growth_alpha
                net_forces = cal_interaction(z, lnw, self.interaction, t, m=interaction_m)
            t_expand = t.expand(z.shape[0], 1)
            score_grad = self.score.compute_gradient(t_expand, z)
            return (drift + score_grad + net_forces, dlnw)

        def g(self, t, y):
            return torch.ones_like(y) * self.sigma

    if verbose:
        print(
            "[piecewise split-SDE] start | "
            f"n_init={x0_t.shape[0]}, ts_points={len(ts_points)}, dt={dt}, sigma={sigma}, "
            f"growth_alpha={growth_alpha}"
        )

    sde = SDE(f_net.v_net, f_net.g_net, score_net, f_net.interaction_net, sigma=sigma)
    ts_tensor = torch.tensor(list(ts_points), dtype=torch.float32, device=device)
    sde_points, _ = euler_sdeint_split(sde, initial_state, dt=dt, ts=ts_tensor, noise_std=0.0)
    sde_point_np = [p.detach().cpu().numpy() for p in sde_points]
    if verbose:
        print(
            "[piecewise split-SDE] done | "
            f"timepoints={len(sde_point_np)}, shape0={sde_point_np[0].shape if sde_point_np else None}"
        )
    return np.array(sde_point_np, dtype=object)


def _simulate_piecewise_spatially_warped_split(
    *,
    x0: np.ndarray,
    f_net,
    score_net,
    observed_time_points: Sequence[float],
    ts_points: Sequence[float],
    df: pd.DataFrame,
    feature_cols_full: Sequence[str],
    label_col: str,
    dt: float,
    sigma: float,
    growth_alpha: float,
    interaction_m: int,
    device: str,
    rng: np.random.Generator,
    k: int,
    eps: float,
) -> np.ndarray:
    ts_sorted = [float(t) for t in ts_points]
    observed_sorted = [float(t) for t in observed_time_points if ts_sorted[0] <= float(t) <= ts_sorted[-1]]
    if len(observed_sorted) < 2:
        return _simulate_sde_points_split_from_x0(
            x0=x0,
            f_net=f_net,
            score_net=score_net,
            ts_points=ts_sorted,
            dt=dt,
            sigma=sigma,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=True,
        )

    current_x0 = np.asarray(x0, dtype=np.float32)
    points_by_time: Dict[float, np.ndarray] = {}

    for t_start, t_end in zip(observed_sorted[:-1], observed_sorted[1:]):
        seg_ts = [float(t) for t in ts_sorted if float(t_start) <= float(t) <= float(t_end)]
        if len(seg_ts) < 2:
            for t_val in seg_ts:
                points_by_time[float(t_val)] = np.asarray(current_x0, dtype=np.float32).copy()
            continue

        print(f"[spatial-warp piecewise] segment {t_start}->{t_end} | targets={seg_ts}")
        seg_points = _simulate_sde_points_split_from_x0(
            x0=current_x0,
            f_net=f_net,
            score_net=score_net,
            ts_points=seg_ts,
            dt=dt,
            sigma=sigma,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=True,
        )

        source_endpoint_xy = np.asarray(seg_points[-1], dtype=np.float32)[:, :2]
        X_target, _ = _sample_observed_x0(
            df,
            time_value=float(t_end),
            feature_cols=feature_cols_full,
            label_col=label_col,
            n_samples_cap=min(int(source_endpoint_xy.shape[0]), int((df["samples"] == float(t_end)).sum())),
            rng=rng,
        )
        target_endpoint_xy = np.asarray(X_target, dtype=np.float32)[:, :2]

        for t_val, pts_raw in zip(seg_ts, seg_points):
            pts = np.asarray(pts_raw, dtype=np.float32).copy()
            alpha = (float(t_val) - float(t_start)) / max(float(t_end - t_start), float(eps))
            if alpha > 0.0:
                disp = _compute_spatial_warp_displacements(
                    pts[:, :2],
                    source_endpoint_xy,
                    target_endpoint_xy,
                    k=k,
                    eps=eps,
                )
                pts[:, :2] = pts[:, :2] + float(alpha) * disp
            points_by_time[float(t_val)] = pts

        current_x0 = np.asarray(points_by_time[float(t_end)], dtype=np.float32).copy()

    missing = [float(t) for t in ts_sorted if float(t) not in points_by_time]
    if missing:
        raise ValueError(f"Piecewise spatial-warp split-SDE missing timepoints: {missing}")
    return np.array([points_by_time[float(t)] for t in ts_sorted], dtype=object)


def prepare_arista_focus_anchor_context(
    *,
    config_path: str,
    annotation_csv: str,
    label_color_json: Optional[str],
    color_h5ad: Optional[str],
    output_dir: str,
    random_seed: Optional[int],
    n_samples: int,
    sde_dt: float,
    classifier_epochs: int,
    classifier_hidden: int,
    classifier_n_pcs: Optional[int],
    classifier_knn_neighbors: int,
    classifier_best_metric: str,
    classifier_train_on_full_data: bool,
    classifier_save_test_models: bool,
    classifier_last_k_epochs: int,
    classifier_checkpoint_dir: Optional[str],
    classifier_cache: bool,
    classifier_cache_dir: Optional[str],
    classifier_cache_tag: Optional[str],
    classifier_cache_path: Optional[str] = None,
    classifier_force_load: bool = False,
    split_sde_dt: float,
    split_sigma: float,
    split_growth_alpha: float,
    interp_time_points: Sequence[float],
    target_total_slices: Optional[int],
    use_real_for_observed: bool,
    time_points_override: Optional[Sequence[float]] = None,
    spatial_warp_to_observed_piecewise: bool = False,
    spatial_warp_k: int = 8,
    spatial_warp_eps: float = 1e-6,
):
    import anndata as ad

    set_global_seed(random_seed)

    config = load_config(config_path)
    df, _ = load_arista_df(config)

    df_anno = pd.read_csv(annotation_csv)
    if "Annotation" not in df_anno.columns:
        raise ValueError(f"Annotation column missing in {annotation_csv}")
    if len(df_anno) != len(df):
        raise ValueError("Annotation CSV length does not match data CSV")

    df = df.copy()
    df["Annotation"] = df_anno["Annotation"].astype(str).values

    dim = int(config["data"]["dim"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    f_net, score_net, exp_dir, device = load_models(
        config,
        exp_name=config["exp"]["name"],
        device=device,
        model_tag="model_final",
        score_tag="score_model",
    )

    print("Device:", device)
    print("Experiment dir:", exp_dir)

    if classifier_n_pcs is None:
        classifier_feature_dim = dim
    else:
        if classifier_n_pcs < 1:
            raise ValueError("classifier_n_pcs must be >= 1 or None")
        classifier_feature_dim = min(int(classifier_n_pcs), dim)

    feature_cols_cls = ["samples"] + [f"x{i}" for i in range(1, classifier_feature_dim + 1)]
    cache_dir = classifier_cache_dir or os.path.join(output_dir, "classifier_cache") if classifier_cache else None
    checkpoint_dir = None
    if classifier_save_test_models:
        checkpoint_dir = classifier_checkpoint_dir or os.path.join(output_dir, "classifier_checkpoints")

    model, label_encoder, acc = train_mlp_classifier(
        df,
        feature_cols=feature_cols_cls,
        label_col="Annotation",
        hidden_size=classifier_hidden,
        epochs=classifier_epochs,
        cache_path=classifier_cache_path,
        cache_dir=cache_dir,
        cache_tag=classifier_cache_tag,
        force_load_cache_path=classifier_cache_path if classifier_force_load else None,
        df_source_path=annotation_csv,
        reuse_if_possible=bool(classifier_cache and (not classifier_save_test_models)),
        progress=True,
        device=device,
        best_epoch_metric=classifier_best_metric,
        train_on_full_data=bool(classifier_train_on_full_data),
        checkpoint_dir=checkpoint_dir,
        save_best_acc=bool(classifier_save_test_models),
        save_best_bacc=bool(classifier_save_test_models),
        save_last_k_epochs=int(classifier_last_k_epochs),
    )
    eval_name = "train_full_acc" if classifier_train_on_full_data else "val_acc"
    print(f"Classifier {eval_name}:", acc)

    if time_points_override is None:
        time_points = sorted(df["samples"].unique().tolist())
    else:
        time_points = [float(t) for t in time_points_override]
    interp_time_points = [float(t) for t in interp_time_points if float(t) not in time_points]

    if target_total_slices is not None:
        keep_observed = max(1, int(target_total_slices) - len(interp_time_points))
        if keep_observed < len(time_points):
            time_points = _select_evenly_spaced(time_points, keep_observed)
            print("Selected observed timepoints:", time_points)

    ts_points = sorted(set(time_points + interp_time_points))

    sde_points, _ = simulate_sde_points(
        df=df,
        dim=dim,
        f_net=f_net,
        score_net=score_net,
        time_index=0,
        n_samples=n_samples,
        ts_points=ts_points,
        dt=sde_dt,
        sigma=0.0,
        include_score=False,
        device=device,
    )

    sde_points_split = None
    if (len(interp_time_points) > 0) or (not use_real_for_observed):
        if spatial_warp_to_observed_piecewise:
            rng_piecewise = np.random.default_rng(1 if random_seed is None else int(random_seed) + 1)
            feature_cols_full = [f"x{i}" for i in range(1, dim + 1)]
            x0_warp, _ = _sample_observed_x0(
                df,
                time_value=float(min(time_points)),
                feature_cols=feature_cols_full,
                label_col="Annotation",
                n_samples_cap=n_samples,
                rng=rng_piecewise,
            )
            sde_points_split = _simulate_piecewise_spatially_warped_split(
                x0=x0_warp,
                f_net=f_net,
                score_net=score_net,
                observed_time_points=time_points,
                ts_points=ts_points,
                df=df,
                feature_cols_full=feature_cols_full,
                label_col="Annotation",
                dt=split_sde_dt,
                sigma=split_sigma,
                growth_alpha=split_growth_alpha,
                interaction_m=1024,
                device=device,
                rng=rng_piecewise,
                k=int(spatial_warp_k),
                eps=float(spatial_warp_eps),
            )
        else:
            sde_points_split = simulate_sde_points_split(
                df=df,
                dim=dim,
                f_net=f_net,
                score_net=score_net,
                time_index=0,
                n_samples=n_samples,
                ts_points=ts_points,
                dt=split_sde_dt,
                sigma=split_sigma,
                growth_alpha=split_growth_alpha,
                device=device,
            )

    predicted_labels_list = predict_labels_for_trajectories(
        sde_points=sde_points,
        ts_points=ts_points,
        model=model,
        label_encoder=label_encoder,
        feature_dim=classifier_feature_dim,
        device=device,
        knn_neighbors=int(classifier_knn_neighbors),
    )

    predicted_labels_split = None
    if sde_points_split is not None:
        predicted_labels_split = predict_labels_for_trajectories(
            sde_points=sde_points_split,
            ts_points=ts_points,
            model=model,
            label_encoder=label_encoder,
            feature_dim=classifier_feature_dim,
            device=device,
            knn_neighbors=int(classifier_knn_neighbors),
        )

    feature_cols_full = [f"x{i}" for i in range(1, dim + 1)]
    adata_dict = {}
    observed_adata_dict = {}
    generated_adata_dict = {}
    time_keys = [str(t) for t in ts_points]

    sde_map_split = None
    pred_map_split = None
    if sde_points_split is not None and predicted_labels_split is not None:
        sde_map_split = {str(ts_points[i]): np.array(sde_points_split[i], dtype=np.float32) for i in range(len(ts_points))}
        pred_map_split = {str(ts_points[i]): np.asarray(predicted_labels_split[i]).astype(str) for i in range(len(ts_points))}

    for t in ts_points:
        key = str(t)
        if t in time_points:
            subset_obs = df[df["samples"] == t]
            X_obs = subset_obs[feature_cols_full].values.astype(np.float32)
            labels_obs = subset_obs["Annotation"].astype(str).values
            adata_obs = ad.AnnData(X=X_obs)
            adata_obs.obs["Annotation"] = labels_obs
            adata_obs.obsm["spatial"] = X_obs[:, :2]
            observed_adata_dict[key] = adata_obs

        if sde_map_split is not None and pred_map_split is not None:
            X_gen = sde_map_split[key]
            labels_gen = pred_map_split[key]
            adata_gen = ad.AnnData(X=X_gen)
            adata_gen.obs["Annotation"] = labels_gen
            adata_gen.obsm["spatial"] = X_gen[:, :2]
            generated_adata_dict[key] = adata_gen

        if use_real_for_observed and t in time_points:
            X = observed_adata_dict[key].X.astype(np.float32)
            labels = observed_adata_dict[key].obs["Annotation"].astype(str).values
        else:
            if sde_map_split is None or pred_map_split is None:
                raise ValueError("Split SDE points missing; set interpolation or use_real_for_observed=True")
            X = sde_map_split[key]
            labels = pred_map_split[key]
        adata = ad.AnnData(X=X)
        adata.obs["Annotation"] = labels
        adata.obsm["spatial"] = X[:, :2]
        adata_dict[key] = adata

    label_to_color = load_label_to_color(
        df["Annotation"].astype(str).values,
        label_color_json=label_color_json,
        color_h5ad=color_h5ad,
    )

    return {
        "config": config,
        "df": df,
        "dim": dim,
        "device": device,
        "f_net": f_net,
        "score_net": score_net,
        "exp_dir": exp_dir,
        "model": model,
        "label_encoder": label_encoder,
        "classifier_feature_dim": classifier_feature_dim,
        "time_points": time_points,
        "interp_time_points": interp_time_points,
        "ts_points": ts_points,
        "time_keys": time_keys,
        "sde_points": sde_points,
        "sde_points_split": sde_points_split,
        "predicted_labels_list": predicted_labels_list,
        "predicted_labels_split": predicted_labels_split,
        "adata_dict": adata_dict,
        "observed_adata_dict": observed_adata_dict,
        "generated_adata_dict": generated_adata_dict,
        "label_to_color": label_to_color,
    }
