#!/usr/bin/env bash
set -euo pipefail

repo="zhenyiizhang/cb_reproducibility"
tag="${1:-reviewer-assets-v1}"
base_url="https://github.com/${repo}/releases/download/${tag}"
tmp_dir="$(mktemp -d)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

download_one() {
  local name="$1"
  local dest="$2"

  mkdir -p "$(dirname "${dest}")"
  if command -v gh >/dev/null 2>&1 && gh auth status -h github.com >/dev/null 2>&1; then
    gh release download "${tag}" --repo "${repo}" --pattern "${name}" --dir "${tmp_dir}" --clobber
    mv "${tmp_dir}/${name}" "${dest}"
  else
    curl -L --fail --retry 3 --output "${dest}" "${base_url}/${name}"
  fi
}

download_one "mosta_four_time_with_celltype_refined.csv" \
  "data/mosta/mosta_four_time_with_celltype_refined.csv"
download_one "attn_interp_t0.0.npy" \
  "results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t0.0.npy"
download_one "attn_interp_t0.5.npy" \
  "results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t0.5.npy"
download_one "attn_interp_t1.0.npy" \
  "results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t1.0.npy"
download_one "attn_interp_t1.5.npy" \
  "results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t1.5.npy"
download_one "attn_interp_t2.0.npy" \
  "results/arista_spatiotemporal_3d_focus_anchor_notebook/attention/attn_interp_t2.0.npy"

echo "Reviewer assets downloaded for ${tag}."
