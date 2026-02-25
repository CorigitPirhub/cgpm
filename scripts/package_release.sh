#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RELEASE_NAME="egf_dhmap3d_v1.0_release"
ZIP_NAME="${RELEASE_NAME}.zip"
STAGE_DIR=".release_stage/${RELEASE_NAME}"
SUMMARY_DIR="${STAGE_DIR}/output/summary_tables"

echo "[package] root: $ROOT_DIR"
echo "[package] staging: $STAGE_DIR"

rm -rf "$STAGE_DIR" ".release_stage/${ZIP_NAME}"
mkdir -p "$STAGE_DIR" "$SUMMARY_DIR"

copy_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -f "$src" "$dst"
}

copy_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}

echo "[package] copying docs"
copy_file "README.md" "${STAGE_DIR}/README.md"
copy_file "DATASETS.md" "${STAGE_DIR}/DATASETS.md"
copy_file "BENCHMARK_REPORT.md" "${STAGE_DIR}/BENCHMARK_REPORT.md"
copy_file "IMPLEMENTATION.md" "${STAGE_DIR}/IMPLEMENTATION.md"
copy_file "LICENSE" "${STAGE_DIR}/LICENSE"
copy_file "requirements.txt" "${STAGE_DIR}/requirements.txt"

echo "[package] copying code"
copy_dir "egf_dhmap3d" "${STAGE_DIR}/egf_dhmap3d"
mkdir -p "${STAGE_DIR}/scripts"
copy_file "scripts/run_benchmark.py" "${STAGE_DIR}/scripts/run_benchmark.py"
copy_file "scripts/run_temporal_ablation.py" "${STAGE_DIR}/scripts/run_temporal_ablation.py"
copy_file "scripts/run_benchmark_bonn.py" "${STAGE_DIR}/scripts/run_benchmark_bonn.py"
copy_file "scripts/run_egf_3d_tum.py" "${STAGE_DIR}/scripts/run_egf_3d_tum.py"
copy_file "scripts/run_tsdf_baseline.py" "${STAGE_DIR}/scripts/run_tsdf_baseline.py"
copy_file "scripts/run_simple_removal_baseline.py" "${STAGE_DIR}/scripts/run_simple_removal_baseline.py"
copy_file "scripts/make_paper_ready_figures.py" "${STAGE_DIR}/scripts/make_paper_ready_figures.py"
copy_file "scripts/build_ablation_summary.py" "${STAGE_DIR}/scripts/build_ablation_summary.py"
copy_dir "scripts/data" "${STAGE_DIR}/scripts/data"

echo "[package] copying assets"
copy_dir "assets" "${STAGE_DIR}/assets"

echo "[package] collecting compact summary tables"
copy_file "output/temporal_ablation/summary.csv" "${SUMMARY_DIR}/temporal_ablation_summary.csv"
copy_file "output/benchmark_bonn/summary.csv" "${SUMMARY_DIR}/bonn_summary.csv"
copy_file "output/benchmark_results/v6_final/tables/reconstruction_metrics.csv" "${SUMMARY_DIR}/tum_v6_summary.csv"
copy_file "output/benchmark_results/v6_final/tables/dynamic_metrics.csv" "${SUMMARY_DIR}/tum_v6_dynamic.csv"
copy_file "output/benchmark_results/v6_final/tables/benchmark_summary.json" "${SUMMARY_DIR}/tum_v6_summary.json"
copy_file "output/benchmark_bonn/summary.json" "${SUMMARY_DIR}/bonn_summary.json"

echo "[package] writing release manifest"
cat > "${STAGE_DIR}/RELEASE_NOTES.txt" <<'TXT'
EGF-DHMap 3D v1.0 release package

Included:
- Core implementation (egf_dhmap3d/)
- Reproducibility scripts for TUM/Bonn/Temporal ablation
- Final benchmark report and implementation notes
- Paper-ready figures (assets/)
- Compact CSV/JSON summary tables in output/summary_tables/
TXT

echo "[package] creating zip: ${ZIP_NAME}"
rm -f "$ZIP_NAME"
(cd ".release_stage" && zip -r "../${ZIP_NAME}" "${RELEASE_NAME}" >/dev/null)

echo "[package] writing checksum"
sha256sum "$ZIP_NAME" > release_checksum.txt

echo "[done] ${ZIP_NAME}"
echo "[done] release_checksum.txt"
