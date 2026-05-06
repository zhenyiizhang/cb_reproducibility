#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream_helpers.zebrafish_api import RUNTIME_ROOT, ZebrafishApiConfig, run_zebrafish_api_subfigures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate zebrafish downstream subfigures through CytoBridge package APIs.")
    parser.add_argument("--zebrafish-root", type=Path, default=RUNTIME_ROOT, help="Path to the zebrafish runtime bundle.")
    parser.add_argument("--cytobridge-repo", type=Path, default=None, help="Optional path to a local cytobridge-spatial checkout.")
    parser.add_argument("--output-name", default="zebrafish_api_subfigures_local")
    args = parser.parse_args()

    if args.cytobridge_repo is not None:
        import os
        os.environ["CYTOBRIDGE_REPO"] = str(args.cytobridge_repo.resolve())

    config = ZebrafishApiConfig(
        zebrafish_root=args.zebrafish_root.resolve(),
        output_name=args.output_name,
    )
    result = run_zebrafish_api_subfigures(config=config)
    print(json.dumps({
        "output_dir": str(result.output_dir),
        "trajectory_dir": str(result.trajectory_dir),
        "gene_velocity_dir": str(result.gene_velocity_dir),
        "spatial_velocity_dir": str(result.spatial_velocity_dir),
        "lr_dir": str(result.lr_dir),
        "gene_program_dir": str(result.gene_program_dir),
        "ablation_dir": str(result.ablation_dir),
        "run_summary": str(result.run_summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
