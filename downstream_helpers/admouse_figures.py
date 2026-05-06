"""Figure helpers for the AD mouse downstream notebooks."""

from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
TIMEPOINT_RE = re.compile(r"adata_t(?P<int>\d+)p(?P<frac>\d+)\.h5ad$")

LR_DOTPLOT_TARGET_PAIRS = (
    "Pecam1_Pecam1",
    "Nts_Ntsr2",
    "Cdh4_Cdh4",
    "Col1a1_Cd44",
    "Fn1_Cd44",
    "Col6a1_Cd44",
    "Spp1_Cd44",
)

SPATIAL_COLOR_SEQUENCE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

ORIGINAL_SPATIAL_SAMPLES = (
    "TgCRND8_2_5",
    "TgCRND8_5_7",
    "TgCRND8_17_9",
)


@dataclass(frozen=True)
class LinearTimeMapper:
    real_times: tuple[float, ...] = (2.5, 5.7, 17.9)
    model_times: tuple[float, ...] = (0.0, 1.0, 2.0)

    def model_to_real(self, value: float) -> float:
        real = np.asarray(self.real_times, dtype=float)
        model = np.asarray(self.model_times, dtype=float)
        value = float(value)
        slope = (real[-1] - real[0]) / (model[-1] - model[0])
        if value < model[0]:
            return float(real[0] + (value - model[0]) * slope)
        if value > model[-1]:
            return float(real[-1] + (value - model[-1]) * slope)
        return float(np.interp(value, model, real))


@dataclass(frozen=True)
class AdMouseAssetPaths:
    asset_root: Path
    config_yaml: Path
    label_to_color_json: Path
    pca_basis_csv: Path
    pca_mean_csv: Path
    edge_classifier_path: Path
    model_final_path: Path
    score_model_path: Path
    params_yaml: Path
    lr_dir: Path


@dataclass(frozen=True)
class AdMouseNewLrPaths:
    merged_components_path: Path
    pca_mean_path: Path
    pca_components_path: Path
    review_lr_dir: Path


@dataclass(frozen=True)
class InterpolatedSliceRecord:
    index: int
    model_time: float
    real_time: float
    path: Path


@dataclass(frozen=True)
class AdMouseGeneExpressionConfig:
    output_name: str = "admouse_gene_expression"
    source_root: str | Path | None = None
    cell_type: str = "Microglia"
    n_clusters: int = 4
    smooth_sigma: float = 0.0
    cluster_method: str = "average"


@dataclass(frozen=True)
class AdMouseGeneExpressionResult:
    output_dir: Path
    figure_pdf_path: Path
    figure_png_path: Path
    expr_shape: tuple[int, int]
    n_patterns: int
    time_labels: tuple[str, ...]

    @property
    def figure_path(self) -> Path:
        return self.figure_pdf_path


@dataclass(frozen=True)
class AdMouseLrScoreConfig:
    output_name: str = "admouse_lr_score"
    source_root: str | Path | None = None
    focus_pair: str = "Pecam1_Pecam1"
    trajectory_time_indices: tuple[int, ...] = (4, 13, 21)
    spatial_key: str = "spatial"
    dotplot_target_pairs: tuple[str, ...] = LR_DOTPLOT_TARGET_PAIRS


@dataclass(frozen=True)
class AdMouseLrScoreResult:
    output_dir: Path
    trajectory_pdf_path: Path
    trajectory_png_path: Path
    dotplot_pdf_path: Path
    dotplot_png_path: Path
    trajectory_times: tuple[str, ...]
    pair_order: tuple[str, ...]

    @property
    def trajectory_path(self) -> Path:
        return self.trajectory_pdf_path

    @property
    def dotplot_path(self) -> Path:
        return self.dotplot_pdf_path


@dataclass(frozen=True)
class AdMouseSpatialComparisonConfig:
    output_name: str = "admouse_spatial_comparison"
    source_root: str | Path | None = None
    original_samples: tuple[str, ...] = ORIGINAL_SPATIAL_SAMPLES
    generated_sample_indices: tuple[int, ...] = (4, 10, 12)


@dataclass(frozen=True)
class AdMouseSpatialComparisonResult:
    output_dir: Path
    original_pdf_path: Path
    original_png_path: Path
    generated_pdf_path: Path
    generated_png_path: Path
    original_samples: tuple[str, ...]
    generated_times: tuple[str, ...]


def resolve_admouse_source_root(source_root: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if source_root is not None:
        candidates.append(Path(source_root))
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "admouse",
            PROJECT_ROOT.parent / "Clean_ad",
            PROJECT_ROOT / "Clean_ad",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate AD mouse data under data/admouse or the sibling Clean_ad directory.")


def resolve_admouse_assets() -> AdMouseAssetPaths:
    asset_root = PROJECT_ROOT / "assets" / "admouse"
    data_root = PROJECT_ROOT / "data" / "admouse"
    return AdMouseAssetPaths(
        asset_root=asset_root,
        config_yaml=asset_root / "config" / "admouse.yaml",
        label_to_color_json=asset_root / "label_to_color.json",
        pca_basis_csv=data_root / "feature_basis" / "admouse_pca_components_with_gene_names.csv",
        pca_mean_csv=data_root / "feature_basis" / "admouse_pca_mean.csv",
        edge_classifier_path=asset_root / "edge_classifier" / "admouse_new.pt",
        model_final_path=asset_root / "model" / "model_final",
        score_model_path=asset_root / "model" / "score_model",
        params_yaml=asset_root / "model" / "params.yml",
        lr_dir=data_root / "Figures" / "LR_analysis_interpolated",
    )


def resolve_admouse_output_dir(output_name: str) -> Path:
    return RESULTS_ROOT / output_name


def _matrix_to_numpy(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _parse_model_time(path: Path) -> float:
    match = TIMEPOINT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse model time from {path.name}")
    integer = int(match.group("int"))
    frac = match.group("frac")
    return integer + int(frac) / (10 ** len(frac))


def list_interpolated_slices(source_root: str | Path | None = None) -> list[InterpolatedSliceRecord]:
    root = resolve_admouse_source_root(source_root)
    mapper = LinearTimeMapper()
    files = sorted((root / "results" / "admouse_new").glob("adata_t*.h5ad"), key=_parse_model_time)
    if not files:
        raise FileNotFoundError(f"No interpolated AD mouse slices found under {root / 'results' / 'admouse_new'}")
    records = []
    for index, path in enumerate(files):
        model_time = float(_parse_model_time(path))
        real_time = float(np.round(mapper.model_to_real(model_time), 1))
        records.append(
            InterpolatedSliceRecord(
                index=index,
                model_time=model_time,
                real_time=real_time,
                path=path,
            )
        )
    return records


def load_pca_basis(path: str | Path | None = None) -> tuple[list[str], np.ndarray, dict[str, int]]:
    csv_path = Path(path) if path is not None else resolve_admouse_assets().pca_basis_csv
    pca_df = pd.read_csv(csv_path)
    gene_names = pca_df.iloc[:, 0].astype(str).tolist()
    pca_components = pca_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    return gene_names, pca_components, gene_to_idx


def load_pca_mean(path: str | Path | None = None) -> np.ndarray:
    csv_path = Path(path) if path is not None else resolve_admouse_assets().pca_mean_csv
    return pd.read_csv(csv_path).to_numpy(dtype=np.float32).reshape(-1)


def _extract_gene_pc_embedding(adata_obj: ad.AnnData) -> np.ndarray:
    embedding = _matrix_to_numpy(adata_obj.X)
    if embedding.ndim != 2 or embedding.shape[1] < 3:
        raise ValueError(f"Expected at least 3 embedding columns, found shape {embedding.shape}")
    return np.asarray(embedding[:, 2:], dtype=np.float32)


def _save_figure_variants(
    fig: mpl.figure.Figure,
    pdf_path: str | Path,
    png_path: str | Path | None = None,
    *,
    dpi: int = 300,
    bbox_inches: str | None = None,
    facecolor: str = "white",
    transparent: bool = False,
) -> tuple[Path, Path]:
    pdf_path = Path(pdf_path)
    if png_path is None:
        png_path = pdf_path.with_suffix(".png")
    png_path = Path(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor=facecolor,
        transparent=transparent,
    )
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor=facecolor,
        transparent=transparent,
    )
    return pdf_path, png_path


def collect_cell_type_expression_matrix(
    slice_records: Iterable[InterpolatedSliceRecord],
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    cell_type: str,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    expr_data: list[np.ndarray] = []
    time_data: list[float] = []
    time_labels: list[str] = []

    pca_components = np.asarray(pca_components, dtype=np.float32)
    pca_mean = np.asarray(pca_mean, dtype=np.float32)

    for record in slice_records:
        adata_obj = ad.read_h5ad(record.path)
        annotations = adata_obj.obs["annotation"].astype(str).to_numpy()
        mask = annotations == cell_type
        if mask.sum() == 0:
            continue
        embedding = _extract_gene_pc_embedding(adata_obj)[mask]
        expr_log1p = embedding @ pca_components.T
        expr_log1p = np.asarray(expr_log1p + pca_mean, dtype=np.float32)
        expr_count = np.clip(np.expm1(expr_log1p), 0.0, None)
        expr_data.append(expr_count.mean(axis=0))
        time_data.append(record.real_time)
        time_labels.append(f"{record.real_time:.1f}")

    if not expr_data:
        raise ValueError(f"Could not find any cells annotated as {cell_type!r}")

    expr_matrix = np.vstack(expr_data).T
    time_array = np.asarray(time_data, dtype=float)
    order = np.argsort(time_array)
    expr_matrix = expr_matrix[:, order]
    time_array = time_array[order]
    time_labels = [time_labels[idx] for idx in order]

    if smooth_sigma > 0:
        for idx in range(expr_matrix.shape[0]):
            expr_matrix[idx] = gaussian_filter1d(expr_matrix[idx], sigma=smooth_sigma)

    return expr_matrix, time_array, time_labels


def zscore_normalize_by_gene(expr_matrix: np.ndarray) -> np.ndarray:
    zscore_matrix = np.zeros_like(expr_matrix, dtype=float)
    for idx in range(expr_matrix.shape[0]):
        gene_data = np.asarray(expr_matrix[idx], dtype=float)
        std = float(gene_data.std())
        if std > 0:
            zscore_matrix[idx] = (gene_data - gene_data.mean()) / std
        else:
            zscore_matrix[idx] = gene_data
    return zscore_matrix


def analyze_gene_patterns(
    expr_matrix: np.ndarray,
    gene_names: list[str],
    n_clusters: int = 4,
    method: str = "average",
    min_genes_per_cluster: int = 5,
) -> dict[str, Any]:
    zscore_matrix = zscore_normalize_by_gene(expr_matrix)
    distance_matrix = pdist(zscore_matrix, metric="euclidean")
    linkage_matrix = linkage(distance_matrix, method=method)
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    pattern_genes: dict[int, dict[str, Any]] = {}
    pattern_curves: dict[int, dict[str, Any]] = {}

    for cluster_id in sorted(np.unique(cluster_labels)):
        gene_indices = np.where(cluster_labels == cluster_id)[0]
        if len(gene_indices) < min_genes_per_cluster:
            continue
        genes = [gene_names[idx] for idx in gene_indices]
        pattern_genes[int(cluster_id)] = {
            "gene_indices": gene_indices.tolist(),
            "genes": genes,
            "n_genes": len(genes),
        }
        cluster_data = zscore_matrix[gene_indices, :]
        pattern_curves[int(cluster_id)] = {
            "mean": cluster_data.mean(axis=0),
            "std": cluster_data.std(axis=0),
            "n_genes": int(len(gene_indices)),
        }

    if not pattern_genes:
        raise ValueError("No gene-expression patterns passed the minimum-size filter")

    return {
        "pattern_genes": pattern_genes,
        "pattern_curves": pattern_curves,
    }


def _build_pattern_boundaries(pattern_results: dict[str, Any]) -> tuple[list[int], list[dict[str, float]]]:
    pattern_genes = pattern_results["pattern_genes"]
    ordered_gene_indices: list[int] = []
    pattern_boundaries: list[dict[str, float]] = []
    current_idx = 0

    for pattern_id in sorted(pattern_genes.keys()):
        gene_indices = list(pattern_genes[pattern_id]["gene_indices"])
        ordered_gene_indices.extend(gene_indices)
        start = current_idx
        end = current_idx + len(gene_indices) - 1
        pattern_boundaries.append(
            {
                "pattern_id": int(pattern_id),
                "start": float(start),
                "end": float(end),
                "n_genes": float(len(gene_indices)),
            }
        )
        current_idx = end + 1
    return ordered_gene_indices, pattern_boundaries


def plot_combined_heatmap_uniform_curves(
    expr_matrix: np.ndarray,
    time_array: np.ndarray,
    time_labels: list[str],
    pattern_results: dict[str, Any],
    cell_type: str,
    save_pdf_path: str | Path,
    save_png_path: str | Path | None = None,
) -> tuple[Path, Path]:
    heatmap_data = zscore_normalize_by_gene(expr_matrix)
    pattern_genes = pattern_results["pattern_genes"]
    pattern_curves = pattern_results["pattern_curves"]
    ordered_gene_indices, pattern_boundaries = _build_pattern_boundaries(pattern_results)
    heatmap_data = heatmap_data[ordered_gene_indices, :]

    n_patterns = len(pattern_boundaries)
    if n_patterns <= 4:
        curve_rows, curve_cols = 2, 2
    elif n_patterns <= 6:
        curve_rows, curve_cols = 2, 3
    else:
        curve_rows, curve_cols = 3, 3

    fig = plt.figure(figsize=(30, 16))
    gs = GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.15)
    ax_heatmap = fig.add_subplot(gs[0])

    im = ax_heatmap.imshow(heatmap_data, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    for idx, boundary in enumerate(pattern_boundaries):
        if idx > 0:
            ax_heatmap.axhline(
                y=boundary["start"] - 0.5,
                color="black",
                linestyle="--",
                linewidth=1.5,
            )
        middle = (boundary["start"] + boundary["end"]) / 2
        ax_heatmap.text(
            heatmap_data.shape[1] - 1.5,
            middle,
            f"P{int(boundary['pattern_id'])}",
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        )

    ax_heatmap.set_yticks([])
    ax_heatmap.set_xticks(np.arange(len(time_array)))
    ax_heatmap.set_xticklabels(time_labels, rotation=45, fontsize=10)
    ax_heatmap.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax_heatmap.set_title(f"{cell_type}: Gene Expression Patterns", fontsize=15, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.9, pad=0.02)
    cbar.set_label("Z-score", fontsize=11)

    pattern_colors = plt.cm.tab10(np.linspace(0, 1, max(n_patterns, 1)))
    all_means = [pattern_curves[pattern_id]["mean"] for pattern_id in sorted(pattern_genes.keys())]
    all_means_concat = np.concatenate(all_means)
    y_min = float(all_means_concat.min() - 0.5)
    y_max = float(all_means_concat.max() + 0.5)
    y_range = max(abs(y_min), abs(y_max))

    curve_gs = GridSpec(
        curve_rows,
        curve_cols,
        left=0.55,
        right=0.95,
        wspace=0.3,
        hspace=0.4,
    )

    for idx in range(min(n_patterns, curve_rows * curve_cols)):
        row = idx // curve_cols
        col = idx % curve_cols
        ax_curve = fig.add_subplot(curve_gs[row, col])
        pattern_id = int(pattern_boundaries[idx]["pattern_id"])
        mean_curve = pattern_curves[pattern_id]["mean"]
        std_curve = pattern_curves[pattern_id]["std"]
        pattern_color = pattern_colors[idx % len(pattern_colors)]
        ax_curve.plot(time_array, mean_curve, linewidth=2, color=pattern_color, marker="o", markersize=3)
        ax_curve.fill_between(
            time_array,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=pattern_color,
        )
        ax_curve.set_ylim(-y_range, y_range)
        ax_curve.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax_curve.set_title(f"Pattern {pattern_id}", fontsize=11, fontweight="bold")
        ax_curve.set_xlabel("Time", fontsize=9)
        ax_curve.set_ylabel("Z-score", fontsize=9)
        ax_curve.grid(True, alpha=0.3)

    for idx in range(min(n_patterns, curve_rows * curve_cols), curve_rows * curve_cols):
        row = idx // curve_cols
        col = idx % curve_cols
        ax_empty = fig.add_subplot(curve_gs[row, col])
        ax_empty.axis("off")

    plt.tight_layout()
    pdf_path, png_path = _save_figure_variants(fig, save_pdf_path, save_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def resolve_admouse_new_lr_paths(source_root: str | Path | None = None) -> AdMouseNewLrPaths:
    root = resolve_admouse_source_root(source_root)
    return AdMouseNewLrPaths(
        merged_components_path=root / "merged_ad_adata_components.h5ad",
        pca_mean_path=root / "new_LR" / "pca_mean.csv",
        pca_components_path=root / "admouse_pca_components_with_gene_names.csv",
        review_lr_dir=root / "Figures" / "LR_analysis_interpolated",
    )

# LR score helpers below intentionally follow the direct computation path from
# `Clean_ad/new_LR/Clean_pipeline_ad.ipynb` together with
# `Clean_ad/new_LR/arista_helpers.py`. The compact repo notebook calls these
# functions so the notebook stays slim without changing the source workflow.

def load_new_lr_source_analyze_attention_by_celltype(
    source_root: str | Path | None = None,
):
    new_lr_root = resolve_admouse_source_root(source_root) / "new_LR"
    if str(new_lr_root) not in sys.path:
        sys.path.insert(0, str(new_lr_root))
    from arista_helpers import analyze_attention_by_celltype
    return analyze_attention_by_celltype


def load_review_lr_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def reconstruct_all_adata_new_from_components(
    source_root: str | Path | None = None,
) -> tuple[list[ad.AnnData], tuple[float, ...]]:
    paths = resolve_admouse_new_lr_paths(source_root)
    merged_adata_list = sc.read_h5ad(paths.merged_components_path)

    all_adata: dict[Any, ad.AnnData] = {}
    for real_time in merged_adata_list.obs["real_time"].unique():
        all_adata[real_time] = merged_adata_list[merged_adata_list.obs["real_time"] == real_time].copy()

    pca_mean = pd.read_csv(paths.pca_mean_path).values.flatten()
    pca_df = pd.read_csv(paths.pca_components_path)
    pca_components = pca_df.iloc[:, 1:].values
    var_df = pd.DataFrame(index=pca_df.iloc[:, 0].astype(str).values)

    all_adata_new: list[ad.AnnData] = []
    for _, adata_obj in all_adata.items():
        X_embedding = adata_obj.X
        if hasattr(X_embedding, "toarray"):
            X_embedding = X_embedding.toarray()
        X_embedding = np.asarray(X_embedding, dtype=float)
        X_embedding = X_embedding[:, 2:]

        X_log1p = X_embedding @ pca_components.T
        X_log1p = X_log1p + pca_mean
        X_count = np.expm1(X_log1p)
        X_count = np.clip(X_count, 0, None)

        adata_new = ad.AnnData(
            X=X_count,
            obs=adata_obj.obs.copy(),
            var=var_df.copy(),
            uns=adata_obj.uns.copy(),
            obsm={key: value.copy() for key, value in adata_obj.obsm.items()},
        )
        all_adata_new.append(adata_new)

    time_keys = tuple(sorted(float(adata_obj.obs["real_time"].iloc[0]) for adata_obj in all_adata_new))
    return all_adata_new, time_keys


def collect_lr_timecourses(
    score_dir: Path,
    times: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    type_rows = []
    missing = []

    for time_value in times:
        path = score_dir / f"lr_scores_{float(time_value)}.pkl"
        if not path.exists():
            missing.append(path.name)
            continue
        record = load_review_lr_record(path)
        cell_types = [str(x) for x in np.asarray(record["cell_types"]).tolist()]
        for lr_pair, mat_raw in record["lr_scores"].items():
            mat = np.asarray(mat_raw, dtype=float)
            incoming = mat.sum(axis=0)
            outgoing = mat.sum(axis=1)
            total = incoming + outgoing
            pair_rows.append(
                {
                    "time": float(record.get("comm_time_key", record["time_key"])),
                    "lr_pair": str(lr_pair),
                    "global_score": float(mat.sum()),
                    "max_edge": float(mat.max()),
                    "nonzero_edges": int((mat > 0).sum()),
                    "n_cell_types": int(len(cell_types)),
                    "peak_receiver": cell_types[int(np.argmax(incoming))] if cell_types else None,
                    "peak_sender": cell_types[int(np.argmax(outgoing))] if cell_types else None,
                }
            )
            type_rows.extend(
                {
                    "time": float(record.get("comm_time_key", record["time_key"])),
                    "lr_pair": str(lr_pair),
                    "cell_type": cell_type,
                    "incoming": float(in_v),
                    "outgoing": float(out_v),
                    "total": float(tot_v),
                }
                for cell_type, in_v, out_v, tot_v in zip(cell_types, incoming, outgoing, total)
            )

    pair_df = (
        pd.DataFrame(pair_rows).sort_values(["lr_pair", "time"]).reset_index(drop=True)
        if pair_rows
        else pd.DataFrame(
            columns=[
                "time",
                "lr_pair",
                "global_score",
                "max_edge",
                "nonzero_edges",
                "n_cell_types",
                "peak_receiver",
                "peak_sender",
            ]
        )
    )
    type_df = (
        pd.DataFrame(type_rows)
        .sort_values(["lr_pair", "time", "total"], ascending=[True, True, False])
        .reset_index(drop=True)
        if type_rows
        else pd.DataFrame(columns=["time", "lr_pair", "cell_type", "incoming", "outgoing", "total"])
    )
    if missing:
        print(f"missing lr record files: {missing}")
    return pair_df, type_df


# Backward-compatible alias for earlier refactors.
collect_review_lr_timecourses = collect_lr_timecourses

def build_new_lr_dotplot_df(
    pair_df: pd.DataFrame,
    target_pairs: tuple[str, ...] | list[str] = LR_DOTPLOT_TARGET_PAIRS,
) -> pd.DataFrame:
    """Lift the exact dot-plot preparation flow from Clean_ad/new_LR/Clean_pipeline_ad.ipynb."""
    df_plot = pair_df.copy()
    target_pairs = list(target_pairs)

    df_dot = df_plot[df_plot["lr_pair"].isin(target_pairs)].copy()
    df_dot["lr_pair"] = pd.Categorical(
        df_dot["lr_pair"],
        categories=target_pairs,
        ordered=True,
    )
    df_dot = df_dot.sort_values(["lr_pair", "time"])
    df_dot["score_norm"] = (
        df_dot.groupby("lr_pair")["global_score"]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    )
    df_dot["time"] = df_dot["time"].map(lambda value: f"{float(value):.1f}")
    return df_dot.reset_index(drop=True)



def plot_new_lr_dotplot(
    pair_df: pd.DataFrame,
    output_dir: str | Path,
    target_pairs: tuple[str, ...] | list[str] = LR_DOTPLOT_TARGET_PAIRS,
) -> tuple[pd.DataFrame, Path, Path]:
    """Render the exact 7-pair dot plot requested by the AD mouse LR notebook."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_dot = build_new_lr_dotplot_df(pair_df, target_pairs)

    fig = plt.figure(figsize=(10, 5), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    scatter = ax.scatter(
        x=df_dot["time"],
        y=df_dot["lr_pair"],
        s=150,
        c=df_dot["score_norm"],
        cmap="RdBu_r",
        edgecolors="black",
        linewidth=0.6,
        vmin=-2.5,
        vmax=2.5,
    )

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.xlabel("Time Point", fontsize=12, labelpad=10)
    plt.ylabel("Ligand-Receptor Pairs", fontsize=11, labelpad=10)
    plt.title("Dynamics of LR Interactions", fontsize=14, pad=20, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    cbar = plt.colorbar(scatter, shrink=0.7, aspect=30, pad=0.02)
    cbar.set_label("Score Norm (Relative Strength)", fontsize=9)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    pdf_path, png_path = _save_figure_variants(
        fig,
        output_dir / "LR.pdf",
        output_dir / "LR.png",
        dpi=300,
    )
    plt.close(fig)
    return df_dot, pdf_path, png_path


def plot_temporal_spatial_trajectory(
    lr_pair: str,
    all_adata_new: list[ad.AnnData],
    lr_result_dir: str | Path,
    adata_index: int | list[int] | tuple[int, ...] | None = None,
    spatial_key: str = "spatial",
    figsize_factor: float = 3,
    time_points: Iterable[float] | None = None,
    metrics: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
    save_pdf: bool = True,
    dpi: int = 300,
) -> dict[str, Any]:
    ligand, receptor = lr_pair.split("_")
    lr_result_dir = Path(lr_result_dir)

    if output_dir is None:
        output_dir = resolve_admouse_output_dir("admouse_lr_score") / "lr_plots" / "multi_index"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if adata_index is not None:
        if isinstance(adata_index, int):
            selected_adata = [all_adata_new[adata_index]]
            index_values = [int(adata_index)]
            index_str = f"_idx{adata_index}"
        elif isinstance(adata_index, (list, tuple)):
            selected_adata = [all_adata_new[i] for i in adata_index]
            index_values = [int(i) for i in adata_index]
            index_str = f"_idx{'_'.join(map(str, adata_index))}"
        else:
            raise ValueError("adata_index must be int, list, or tuple")
    else:
        selected_adata = all_adata_new
        index_values = list(range(len(all_adata_new)))
        index_str = "_all"

    if metrics is None:
        metrics = ["ligand", "receptor", "potential", "score"]

    trajectory_data = {
        "time_keys": [],
        "coords": [],
        "ligand": [],
        "receptor": [],
        "potential": [],
        "score": [],
    }

    for adata_obj in selected_adata:
        time_key = float(adata_obj.obs["real_time"].iloc[0])

        if time_points is not None:
            if not any(np.isclose(time_key, t) for t in time_points):
                continue

        score_file = lr_result_dir / f"lr_scores_{time_key}.pkl"
        if not score_file.exists():
            print(f"Skip time {time_key}: score file not found")
            continue

        record = load_review_lr_record(score_file)
        lr_scores = record["lr_scores"]
        cell_types = record["cell_types"]

        if lr_pair not in lr_scores:
            print(f"Skip time {time_key}: {lr_pair} not in lr_scores")
            continue

        lr_matrix = np.asarray(lr_scores[lr_pair])

        if spatial_key in adata_obj.obsm:
            coords = np.asarray(adata_obj.obsm[spatial_key])
        else:
            coords = np.zeros((adata_obj.n_obs, 2))

        l_expr = adata_obj[:, ligand].X
        r_expr = adata_obj[:, receptor].X
        if hasattr(l_expr, "toarray"):
            l_expr = l_expr.toarray().flatten()
        if hasattr(r_expr, "toarray"):
            r_expr = r_expr.toarray().flatten()
        l_expr = np.asarray(l_expr, dtype=float)
        r_expr = np.asarray(r_expr, dtype=float)

        l_vis = np.log1p(l_expr)
        r_vis = np.log1p(r_expr)
        potential_vis = np.log1p(l_expr * r_expr)

        annotations = adata_obj.obs["annotation"].astype(str).values
        cell_scores = np.zeros(adata_obj.n_obs)
        for i, ct in enumerate(cell_types):
            mask = annotations == str(ct)
            if mask.sum() == 0:
                continue
            cell_scores[mask] = lr_matrix[:, i].sum()
        score_vis = np.log1p(cell_scores)

        trajectory_data["time_keys"].append(time_key)
        trajectory_data["coords"].append(coords)
        trajectory_data["ligand"].append(l_vis)
        trajectory_data["receptor"].append(r_vis)
        trajectory_data["potential"].append(potential_vis)
        trajectory_data["score"].append(score_vis)

    if len(trajectory_data["time_keys"]) == 0:
        raise ValueError(f"No valid data for {lr_pair}")

    sort_idx = np.argsort(trajectory_data["time_keys"])
    for key in trajectory_data:
        trajectory_data[key] = [trajectory_data[key][i] for i in sort_idx]

    def rotate_ccw(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        rotated = np.zeros_like(coords)
        rotated[:, 0] = -coords[:, 1]
        rotated[:, 1] = coords[:, 0]
        return rotated

    generated_files: dict[str, dict[str, Path | None]] = {}

    def plot_row(data_list: list[np.ndarray], metric_name: str, cmap: str, metric_key: str) -> dict[str, Path | None]:
        n_times = len(data_list)
        all_vals = np.concatenate(data_list)
        vmin = float(np.min(all_vals))
        vmax = float(np.percentile(all_vals, 99.5))
        if vmax <= vmin:
            vmax = vmin + 1e-5

        if metric_key == "score":
            coords_list = [rotate_ccw(coords) for coords in trajectory_data["coords"]]
        else:
            coords_list = [np.asarray(coords, dtype=float) for coords in trajectory_data["coords"]]

        all_coords = np.vstack(coords_list)
        xmin, xmax = all_coords[:, 0].min(), all_coords[:, 0].max()
        ymin, ymax = all_coords[:, 1].min(), all_coords[:, 1].max()

        fig, axes = plt.subplots(
            1,
            n_times,
            figsize=(n_times * figsize_factor, figsize_factor),
            dpi=dpi,
        )
        if n_times == 1:
            axes = [axes]

        fig.patch.set_facecolor("white")
        sc = None
        for i, ax in enumerate(axes):
            coords = coords_list[i]
            vals = data_list[i]
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=vals,
                s=1,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolors="none",
                rasterized=True,
            )
            ax.set_title(f"T {trajectory_data['time_keys'][i]:.1f}")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.axis("off")
            ax.set_aspect("equal")

        if sc is None:
            raise ValueError("No panels were drawn")

        fig.subplots_adjust(right=0.88, wspace=0.05)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        colorbar.set_label(metric_name)
        colorbar.outline.set_visible(False)

        pdf_path: Path | None = None
        png_path = output_dir / f"{lr_pair}_{metric_key}{index_str}.png"
        if save_pdf:
            pdf_path = output_dir / f"{lr_pair}_{metric_key}{index_str}.pdf"
            pdf_path, png_path = _save_figure_variants(
                fig,
                pdf_path,
                png_path,
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                png_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                transparent=False,
            )
        plt.close(fig)
        return {"pdf_path": pdf_path, "png_path": png_path}

    metric_info = {
        "ligand": ("Reds", f"{ligand} expression", "ligand"),
        "receptor": ("Blues", f"{receptor} expression", "receptor"),
        "potential": ("plasma", f"{lr_pair} potential", "potential"),
        "score": ("Reds", f"{lr_pair} interaction score", "score"),
    }

    for metric in metrics:
        cmap, name, metric_key = metric_info[metric]
        generated_files[metric] = plot_row(
            trajectory_data[metric],
            name,
            cmap,
            metric_key,
        )

    return {
        "time_keys": tuple(float(value) for value in trajectory_data["time_keys"]),
        "selected_indices": tuple(index_values),
        "files": generated_files,
    }

def _build_spatial_color_map(labels: Iterable[str]) -> dict[str, str]:
    sorted_labels = sorted({str(label) for label in labels})
    return {
        label: SPATIAL_COLOR_SEQUENCE[idx % len(SPATIAL_COLOR_SEQUENCE)]
        for idx, label in enumerate(sorted_labels)
    }


def _plot_spatial_panels(
    panels: list[dict[str, Any]],
    label_key: str,
    point_size: float,
    title: str,
    save_pdf_path: str | Path,
    save_png_path: str | Path | None = None,
) -> tuple[Path, Path]:
    all_cell_types: list[str] = []
    for panel in panels:
        all_cell_types.extend(str(value) for value in panel[label_key])
    color_map = _build_spatial_color_map(all_cell_types)
    sorted_cell_types = sorted(color_map.keys())

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 18))
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        spatial_coords = np.asarray(panel["coords"], dtype=float)
        rotated_coords = np.zeros_like(spatial_coords)
        rotated_coords[:, 0] = -spatial_coords[:, 1]
        rotated_coords[:, 1] = spatial_coords[:, 0]
        cell_colors = [color_map[str(cell_type)] for cell_type in panel[label_key]]
        ax.scatter(
            rotated_coords[:, 0],
            rotated_coords[:, 1],
            c=cell_colors,
            s=point_size,
            alpha=0.85,
            edgecolors="none",
            linewidths=0,
            marker="o",
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.015, linestyle="-", linewidth=0.2)

    legend_patches = [
        Patch(facecolor=color_map[cell_type], edgecolor="none", label=cell_type, linewidth=0)
        for cell_type in sorted_cell_types
    ]
    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title="Cell Types",
        title_fontsize=11,
        frameon=False,
        ncol=1 if len(legend_patches) <= 15 else 2,
    )
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    pdf_path, png_path = _save_figure_variants(fig, save_pdf_path, save_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def load_original_spatial_panels(
    source_root: str | Path | None = None,
    samples: tuple[str, ...] | list[str] = ORIGINAL_SPATIAL_SAMPLES,
) -> list[dict[str, Any]]:
    root = resolve_admouse_source_root(source_root)
    adata_obj = ad.read_h5ad(root / "ad_data.h5ad")
    panels: list[dict[str, Any]] = []
    sample_array = adata_obj.obs["sample"].astype(str).to_numpy()
    for sample in samples:
        mask = sample_array == str(sample)
        sample_data = adata_obj[mask].copy()
        panels.append(
            {
                "sample": str(sample),
                "coords": np.asarray(sample_data.obsm["spatial"], dtype=float),
                "labels": sample_data.obs["major_annotation"].astype(str).to_numpy(),
            }
        )
    return panels


def load_generated_spatial_panels(
    source_root: str | Path | None = None,
    sample_indices: tuple[int, ...] | list[int] = (4, 10, 12),
) -> list[dict[str, Any]]:
    slice_records = list_interpolated_slices(source_root)
    panels: list[dict[str, Any]] = []
    for index in sample_indices:
        record = slice_records[int(index)]
        adata_obj = ad.read_h5ad(record.path)
        panels.append(
            {
                "model_time": f"{record.model_time:.1f}",
                "real_time": f"{record.real_time:.1f}",
                "coords": np.asarray(adata_obj.obsm["spatial"], dtype=float),
                "labels": adata_obj.obs["annotation"].astype(str).to_numpy(),
            }
        )
    return panels


def preview_admouse_gene_expression_inputs(
    config: AdMouseGeneExpressionConfig,
    max_slices: int = 3,
) -> dict[str, Any]:
    slice_records = list_interpolated_slices(config.source_root)[:max_slices]
    gene_names, pca_components, _ = load_pca_basis()
    pca_mean = load_pca_mean()
    expr_matrix, time_array, time_labels = collect_cell_type_expression_matrix(
        slice_records,
        pca_components,
        pca_mean,
        cell_type=config.cell_type,
        smooth_sigma=config.smooth_sigma,
    )
    return {
        "n_slices": len(slice_records),
        "n_genes": len(gene_names),
        "expr_shape": tuple(expr_matrix.shape),
        "time_labels": tuple(time_labels),
        "time_array": tuple(float(value) for value in time_array),
    }


def preview_admouse_lr_score_inputs(
    config: AdMouseLrScoreConfig,
    max_slices: int = 3,
) -> dict[str, Any]:
    _ = load_new_lr_source_analyze_attention_by_celltype(config.source_root)
    all_adata_new, time_keys = reconstruct_all_adata_new_from_components(config.source_root)
    paths = resolve_admouse_new_lr_paths(config.source_root)
    pair_df, _ = collect_lr_timecourses(paths.review_lr_dir, time_keys)
    df_dot = build_new_lr_dotplot_df(pair_df, config.dotplot_target_pairs)

    preview_indices = tuple(int(idx) for idx in config.trajectory_time_indices[:max_slices])
    preview_times = tuple(
        f"{float(all_adata_new[idx].obs['real_time'].iloc[0]):.1f}"
        for idx in preview_indices
        if 0 <= int(idx) < len(all_adata_new)
    )
    return {
        "preview_times": preview_times,
        "dotplot_rows": int(len(df_dot)),
        "pair_order": tuple(df_dot["lr_pair"].astype(str).drop_duplicates().tolist()),
    }

def preview_admouse_spatial_inputs(
    config: AdMouseSpatialComparisonConfig,
) -> dict[str, Any]:
    original_panels = load_original_spatial_panels(config.source_root, config.original_samples)
    generated_panels = load_generated_spatial_panels(config.source_root, config.generated_sample_indices)
    return {
        "original_samples": tuple(panel["sample"] for panel in original_panels),
        "generated_times": tuple(panel["real_time"] for panel in generated_panels),
        "n_original_cell_types": len({label for panel in original_panels for label in panel["labels"]}),
        "n_generated_cell_types": len({label for panel in generated_panels for label in panel["labels"]}),
    }


def run_admouse_gene_expression_figure(config: AdMouseGeneExpressionConfig) -> AdMouseGeneExpressionResult:
    output_dir = resolve_admouse_output_dir(config.output_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_records = list_interpolated_slices(config.source_root)
    gene_names, pca_components, _ = load_pca_basis()
    pca_mean = load_pca_mean()
    expr_matrix, time_array, time_labels = collect_cell_type_expression_matrix(
        slice_records,
        pca_components,
        pca_mean,
        cell_type=config.cell_type,
        smooth_sigma=config.smooth_sigma,
    )
    pattern_results = analyze_gene_patterns(
        expr_matrix=expr_matrix,
        gene_names=gene_names,
        n_clusters=config.n_clusters,
        method=config.cluster_method,
    )
    figure_pdf_path, figure_png_path = plot_combined_heatmap_uniform_curves(
        expr_matrix=expr_matrix,
        time_array=time_array,
        time_labels=time_labels,
        pattern_results=pattern_results,
        cell_type=config.cell_type,
        save_pdf_path=output_dir / "microglia_combined_uniform.pdf",
        save_png_path=output_dir / "microglia_combined_uniform.png",
    )
    return AdMouseGeneExpressionResult(
        output_dir=output_dir,
        figure_pdf_path=figure_pdf_path,
        figure_png_path=figure_png_path,
        expr_shape=tuple(expr_matrix.shape),
        n_patterns=len(pattern_results["pattern_genes"]),
        time_labels=tuple(time_labels),
    )


def run_admouse_lr_score_figures(config: AdMouseLrScoreConfig) -> AdMouseLrScoreResult:
    output_dir = resolve_admouse_output_dir(config.output_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ = load_new_lr_source_analyze_attention_by_celltype(config.source_root)
    paths = resolve_admouse_new_lr_paths(config.source_root)
    all_adata_new, time_keys = reconstruct_all_adata_new_from_components(config.source_root)
    pair_df, _ = collect_lr_timecourses(paths.review_lr_dir, time_keys)

    df_dot, dotplot_pdf_path, dotplot_png_path = plot_new_lr_dotplot(
        pair_df=pair_df,
        output_dir=output_dir,
        target_pairs=config.dotplot_target_pairs,
    )

    trajectory_outputs = plot_temporal_spatial_trajectory(
        lr_pair=config.focus_pair,
        all_adata_new=all_adata_new,
        lr_result_dir=paths.review_lr_dir,
        adata_index=list(config.trajectory_time_indices),
        spatial_key=config.spatial_key,
        metrics=["score"],
        output_dir=output_dir / "lr_plots" / "multi_index",
        save_pdf=True,
        dpi=300,
    )

    score_outputs = trajectory_outputs["files"]["score"]
    trajectory_pdf_path = score_outputs["pdf_path"]
    trajectory_png_path = score_outputs["png_path"]
    if trajectory_pdf_path is None:
        raise ValueError("The score trajectory PDF was not generated")

    pair_order = tuple(df_dot["lr_pair"].astype(str).drop_duplicates().tolist())
    trajectory_times = tuple(f"{float(value):.1f}" for value in trajectory_outputs["time_keys"])
    return AdMouseLrScoreResult(
        output_dir=output_dir,
        trajectory_pdf_path=trajectory_pdf_path,
        trajectory_png_path=trajectory_png_path,
        dotplot_pdf_path=dotplot_pdf_path,
        dotplot_png_path=dotplot_png_path,
        trajectory_times=trajectory_times,
        pair_order=pair_order,
    )

def run_admouse_spatial_comparison_figures(
    config: AdMouseSpatialComparisonConfig,
) -> AdMouseSpatialComparisonResult:
    output_dir = resolve_admouse_output_dir(config.output_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_panels = load_original_spatial_panels(config.source_root, config.original_samples)
    generated_panels = load_generated_spatial_panels(config.source_root, config.generated_sample_indices)

    original_pdf_path, original_png_path = _plot_spatial_panels(
        original_panels,
        label_key="labels",
        point_size=2.5,
        title="TgCRND8 Mouse Brain Spatial Distribution",
        save_pdf_path=output_dir / "original_samples_spatial.pdf",
        save_png_path=output_dir / "original_samples_spatial.png",
    )
    generated_pdf_path, generated_png_path = _plot_spatial_panels(
        generated_panels,
        label_key="labels",
        point_size=4,
        title="TgCRND8 Mouse Brain Spatial Distribution",
        save_pdf_path=output_dir / "interpolated_samples_spatial.pdf",
        save_png_path=output_dir / "interpolated_samples_spatial.png",
    )
    return AdMouseSpatialComparisonResult(
        output_dir=output_dir,
        original_pdf_path=original_pdf_path,
        original_png_path=original_png_path,
        generated_pdf_path=generated_pdf_path,
        generated_png_path=generated_png_path,
        original_samples=tuple(panel["sample"] for panel in original_panels),
        generated_times=tuple(panel["real_time"] for panel in generated_panels),
    )
