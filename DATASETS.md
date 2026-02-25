# Datasets For Reproduction

This release package does not bundle raw RGB-D datasets (too large), but all required benchmark datasets can be downloaded automatically by the provided scripts.

## 1) TUM RGB-D
- Website: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- Sequence used in this release:
  - `rgbd_dataset_freiburg3_walking_xyz`
- Script support:
  - `scripts/run_benchmark.py --dataset_kind tum --download ...`
  - `scripts/run_temporal_ablation.py` (uses local TUM folder)

## 2) Bonn RGB-D Dynamic Dataset
- Website: https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/
- Sequence used in this release:
  - `rgbd_bonn_balloon2`
- Script support:
  - `scripts/run_benchmark_bonn.py --download ...`
  - `scripts/run_benchmark.py --dataset_kind bonn --download ...`

## 3) Local default layout
- TUM root: `data/tum/<sequence_name>/...`
- Bonn root: `data/bonn/<sequence_name>/...`

