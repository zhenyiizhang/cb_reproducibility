from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "downstream_helpers" / "arista_api.py"

PL_EXPORTS = (
    "gene_velocity_embeddings_from_adata",
    "plot_enrichment_bar",
    "plot_enrichment_dot",
    "plot_celltype_composition",
    "plot_growth_interaction_bubble",
    "plot_growth_timepoint_grid",
    "plot_spatial_component_direction_correlation_roi_from_adata",
    "plot_temporal_gene_heatmap",
    "plot_temporal_pattern_prototypes",
    "plot_temporal_profile_small_multiples",
    "plot_velocity_component",
)

TL_EXPORTS = (
    "adata_to_aligned_dataframe",
    "build_dynamical_runtime",
    "compute_timepoint_communications",
    "evaluate_growth_by_timepoint",
    "compute_velocity_components",
    "compute_velocity_components_from_adata",
    "infer_feature_columns",
    "infer_time_key",
    "load_dynamical_model_from_dir",
    "load_label_to_color",
    "load_gmt_gene_sets",
    "load_legacy_dynamical_model_from_dir",
    "load_pca_reconstruction_spec",
    "plot_lineage_sankey",
    "plot_spatiotemporal_3d",
    "overrepresentation_analysis",
    "project_communication_to_lr_timecourses",
    "run_interpolation_workflow",
    "save_timepoint_snapshots",
    "set_global_random_seed",
    "summarize_growth_interaction_by_celltype",
    "summarize_label_composition",
    "summarize_temporal_gene_patterns",
)


@pytest.fixture()
def arista_api(monkeypatch):
    """Load the module with tiny CytoBridge stubs; no model stack is imported."""
    def dummy(*args, **kwargs):
        return None

    cytobridge = ModuleType("CytoBridge")
    cytobridge.__path__ = []
    pl = ModuleType("CytoBridge.pl")
    tl = ModuleType("CytoBridge.tl")
    for name in PL_EXPORTS:
        setattr(pl, name, dummy)
    for name in TL_EXPORTS:
        setattr(tl, name, dummy)
    cytobridge.pl = pl
    cytobridge.tl = tl
    monkeypatch.setitem(sys.modules, "CytoBridge", cytobridge)
    monkeypatch.setitem(sys.modules, "CytoBridge.pl", pl)
    monkeypatch.setitem(sys.modules, "CytoBridge.tl", tl)

    module_name = "_arista_api_lineage_contract_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_requires_non_split_labels_even_when_split_labels_exist(arista_api):
    interpolation = SimpleNamespace(
        predicted_labels_list=None,
        predicted_labels_split=[["not-a-persistent-lineage"]],
    )

    with pytest.raises(RuntimeError, match="cannot be paired by row index"):
        arista_api._require_persistent_lineage_labels(interpolation)


def test_returns_non_split_labels_without_consulting_split_rows(arista_api):
    persistent_labels = [["reaEGC"], ["wntEGC"]]
    interpolation = SimpleNamespace(
        predicted_labels_list=persistent_labels,
        predicted_labels_split=[["different-population"]],
    )

    assert (
        arista_api._require_persistent_lineage_labels(interpolation)
        is persistent_labels
    )


def test_main_workflow_rejects_skip_nonsplit_before_loading_data(
    arista_api, monkeypatch
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("data/model loading should not start")

    monkeypatch.setattr(arista_api, "load_arista_spatiotemporal_context", fail_if_called)
    config = arista_api.AristaSpatiotemporalConfig(skip_nonsplit_sde=True)

    with pytest.raises(ValueError, match="skip_nonsplit_sde=True is unsafe"):
        arista_api.run_arista_spatiotemporal_api(config)
