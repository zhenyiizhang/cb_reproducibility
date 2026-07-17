#!/usr/bin/env python3
"""Regenerate ARISTA Figure 5e through the public CytoBridge API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_helpers.arista_api import (  # noqa: E402
    AristaSpatiotemporalConfig,
    assert_package_only_runtime,
    run_arista_growth_interaction_api,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-h5ad", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-seed", "--seed", dest="random_seed", type=int, default=42)
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--max-cells-per-timepoint", type=int, default=None)
    return parser


def run(args: argparse.Namespace):
    if args.max_cells_per_timepoint is not None and args.max_cells_per_timepoint <= 0:
        raise ValueError("--max-cells-per-timepoint must be positive")
    config = AristaSpatiotemporalConfig(
        output_name="arista_clean_growth_interaction_api",
        output_dir=Path(args.output_dir).expanduser().resolve(),
        aligned_h5ad=Path(args.aligned_h5ad).expanduser().resolve(),
        model_dir=Path(args.model_dir).expanduser().resolve(),
        model_format="current",
        annotation_key=str(args.annotation_key),
        spatial_warp_to_observed_piecewise=False,
        random_seed=int(args.random_seed),
        device=str(args.device),
        run_communication=False,
        run_3d=False,
    )
    result = run_arista_growth_interaction_api(
        config,
        max_cells_per_timepoint=args.max_cells_per_timepoint,
    )
    assert_package_only_runtime()
    return result


def main() -> None:
    result = run(_parser().parse_args())
    print(result.manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
