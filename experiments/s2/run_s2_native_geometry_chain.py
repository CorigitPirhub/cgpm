from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS_DIR))

from run_s2_write_time_synthesis import dryrun_base_cmds, ensure_flag, run, set_arg
from run_s2_rps_rear_geometry_quality import build_commands, summarize_variant
from run_s2_rps_ray_consistency import apply_ray_args
from run_s2_rps_upstream_geometry import attach_upstream_args


OUT_DIR = PROJECT_ROOT / "output" / "s2"
STAGE_ROOT = PROJECT_ROOT / "output" / "s2_stage"
CSV_PATH = OUT_DIR / "S2_NATIVE_BASELINE_111.csv"
MD_PATH = OUT_DIR / "S2_NATIVE_BASELINE_111.md"
TUM_REFERENCE_ROOT = STAGE_ROOT / "72_local_geometric_conflict_resolution_semantic" / "tum_oracle" / "oracle"


def build_spec(frames: int, stride: int, max_points: int) -> dict:
    return {
        "frames": frames,
        "stride": stride,
        "max_points": max_points,
        "name": "111_native_geometry_chain_direct",
        "output_name": "111_native_geometry_chain_direct",
        "depth_bias_offset_m": -0.01,
        "surface_geometry_chain_coupling_enable": True,
        "surface_geometry_chain_coupling_mode": "direct",
        "surface_geometry_chain_coupling_donor_root": str(STAGE_ROOT / "99_manhattan_plane_completion"),
        "surface_geometry_chain_coupling_project_dist": 0.05,
        "rps_candidate_rescue_enable": True,
        "rps_candidate_support_gain": 0.36,
        "rps_candidate_bg_gain": 0.30,
        "rps_candidate_rho_gain": 0.22,
        "rps_candidate_front_relax": 0.38,
        "joint_bg_state_enable": True,
        "joint_bg_state_on": 0.15,
        "joint_bg_state_gain": 0.76,
        "joint_bg_state_rho_gain": 0.30,
        "joint_bg_state_front_penalty": 0.10,
        "rps_commit_activation_enable": True,
        "rps_commit_threshold": 0.34,
        "rps_commit_release": 0.22,
        "rps_commit_age_threshold": 1.0,
        "rps_commit_rho_ref": 0.035,
        "rps_commit_weight_ref": 0.35,
        "rps_commit_min_cand_rho": 0.006,
        "rps_commit_min_cand_w": 0.025,
        "rps_commit_evidence_weight": 0.38,
        "rps_commit_geometry_weight": 0.32,
        "rps_commit_bg_weight": 0.20,
        "rps_commit_static_weight": 0.10,
        "rps_commit_front_penalty": 0.12,
        "rps_surface_bank_enable": True,
        "rps_bank_margin": 0.02,
        "rps_bank_separation_ref": 0.02,
        "rps_bank_rear_min_score": 0.18,
        "bonn_rps_admission_support_enable": True,
        "bonn_rps_admission_support_on": 0.36,
        "bonn_rps_admission_support_gain": 0.60,
        "bonn_rps_admission_score_relax": 0.06,
        "bonn_rps_admission_active_floor": 0.28,
        "bonn_rps_admission_rho_ref": 0.07,
        "bonn_rps_admission_weight_ref": 0.30,
        "bonn_rps_rear_state_protect_enable": True,
        "bonn_rps_rear_state_decay_relax": 0.65,
        "bonn_rps_rear_state_active_floor": 0.34,
        "bonn_rps_bg_manifold_state_enable": True,
        "bonn_rps_bg_dense_state_enable": True,
        "bonn_rps_bg_surface_constrained_enable": True,
        "bonn_rps_bg_bridge_enable": True,
        "bonn_rps_bg_bridge_min_visible": 0.20,
        "bonn_rps_bg_bridge_min_obstruction": 0.22,
        "bonn_rps_bg_bridge_min_step": 2,
        "bonn_rps_bg_bridge_max_step": 5,
        "bonn_rps_bg_bridge_gain": 0.86,
        "bonn_rps_bg_bridge_target_dyn_max": 0.28,
        "bonn_rps_bg_bridge_target_surface_max": 0.30,
        "bonn_rps_bg_bridge_ghost_suppress_enable": True,
        "bonn_rps_bg_bridge_ghost_suppress_weight": 0.18,
        "bonn_rps_bg_bridge_relaxed_dyn_max": 0.44,
        "bonn_rps_bg_bridge_keep_multi_hits": True,
        "bonn_rps_bg_bridge_max_hits_per_source": 18,
        "bonn_rps_bg_bridge_patch_radius_cells": 1,
        "bonn_rps_bg_bridge_patch_gain_scale": 0.58,
        "bonn_rps_bg_bridge_depth_hypothesis_count": 1,
        "bonn_rps_bg_bridge_depth_step_scale": 0.30,
        "bonn_rps_bg_bridge_cone_enable": True,
        "bonn_rps_bg_bridge_cone_radius_cells": 1,
        "bonn_rps_bg_bridge_cone_gain_scale": 0.52,
        "bonn_rps_bg_bridge_rear_synth_enable": True,
        "bonn_rps_bg_bridge_rear_support_gain": 0.18,
        "bonn_rps_bg_bridge_rear_rho_gain": 0.09,
        "bonn_rps_bg_bridge_rear_phi_blend": 0.86,
        "bonn_rps_bg_bridge_rear_score_floor": 0.23,
        "bonn_rps_bg_bridge_rear_active_floor": 0.53,
        "bonn_rps_bg_bridge_rear_age_floor": 1.0,
        "bonn_rps_bg_manifold_history_weight": 0.0,
        "bonn_rps_bg_manifold_obstruction_weight": 0.0,
        "bonn_rps_bg_manifold_visible_lo": 0.0,
        "bonn_rps_bg_manifold_visible_hi": 0.0,
        "bonn_rps_rear_selectivity_enable": True,
        "bonn_rps_rear_selectivity_support_weight": 0.10,
        "bonn_rps_rear_selectivity_history_weight": 0.18,
        "bonn_rps_rear_selectivity_static_weight": 0.12,
        "bonn_rps_rear_selectivity_geom_weight": 0.18,
        "bonn_rps_rear_selectivity_bridge_weight": 0.08,
        "bonn_rps_rear_selectivity_density_weight": 0.08,
        "bonn_rps_rear_selectivity_rear_score_weight": 0.18,
        "bonn_rps_rear_selectivity_front_score_weight": 0.26,
        "bonn_rps_rear_selectivity_competition_weight": 0.38,
        "bonn_rps_rear_selectivity_competition_alpha": 0.90,
        "bonn_rps_rear_selectivity_gap_weight": 0.08,
        "bonn_rps_rear_selectivity_sep_weight": 0.08,
        "bonn_rps_rear_selectivity_dyn_weight": 0.12,
        "bonn_rps_rear_selectivity_ghost_weight": 0.12,
        "bonn_rps_rear_selectivity_front_weight": 0.10,
        "bonn_rps_rear_selectivity_geom_risk_weight": 0.18,
        "bonn_rps_rear_selectivity_history_risk_weight": 0.10,
        "bonn_rps_rear_selectivity_density_risk_weight": 0.08,
        "bonn_rps_rear_selectivity_bridge_relief_weight": 0.08,
        "bonn_rps_rear_selectivity_static_relief_weight": 0.06,
        "bonn_rps_rear_selectivity_gap_risk_weight": 0.06,
        "bonn_rps_rear_selectivity_score_min": 0.40,
        "bonn_rps_rear_selectivity_risk_max": 0.60,
        "bonn_rps_rear_selectivity_geom_floor": 0.36,
        "bonn_rps_rear_selectivity_history_floor": 0.30,
        "bonn_rps_rear_selectivity_bridge_floor": 0.12,
        "bonn_rps_rear_selectivity_competition_floor": 0.02,
        "bonn_rps_rear_selectivity_front_score_max": 0.72,
        "bonn_rps_rear_selectivity_gap_valid_min": 0.10,
        "bonn_rps_rear_selectivity_topk": 92,
        "bonn_rps_rear_selectivity_rank_risk_weight": 0.55,
        "bonn_rps_rear_selectivity_penetration_weight": 0.55,
        "bonn_rps_rear_selectivity_penetration_floor": 0.18,
        "bonn_rps_rear_selectivity_penetration_risk_weight": 0.20,
        "bonn_rps_rear_selectivity_penetration_free_ref": 0.07,
        "bonn_rps_rear_selectivity_penetration_max_steps": 12,
    }


def write_md(path: Path, row: dict) -> None:
    lines = [
        "# S2 Native Baseline 111",
        "",
        "当前稳定原生基线：`111_native_geometry_chain_direct`",
        "",
        f"- `tum_acc_cm`: `{float(row['tum_acc_cm']):.4f}`",
        f"- `tum_comp_r_5cm`: `{float(row['tum_comp_r_5cm']):.2f}`",
        f"- `bonn_acc_cm`: `{float(row['bonn_acc_cm']):.4f}`",
        f"- `bonn_comp_r_5cm`: `{float(row['bonn_comp_r_5cm']):.2f}`",
        f"- `bonn_ghost_reduction_vs_tsdf`: `{float(row['bonn_ghost_reduction_vs_tsdf']):.2f}`",
        f"- `bonn_rear_true_background_sum`: `{float(row['bonn_rear_true_background_sum']):.0f}`",
        f"- `bonn_rear_ghost_sum`: `{float(row['bonn_rear_ghost_sum']):.0f}`",
        f"- `bonn_rear_hole_or_noise_sum`: `{float(row['bonn_rear_hole_or_noise_sum']):.0f}`",
        "",
        "说明：该脚本不再依赖旧 `processes/` 目录，输出统一写入 `output/s2/` 与 `output/s2_stage/`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run stable S2 native baseline 111.")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_points_per_frame", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    spec = build_spec(args.frames, args.stride, args.max_points_per_frame)
    base_tum, base_bonn = dryrun_base_cmds(args.frames, args.stride, args.max_points_per_frame)
    stage_root = STAGE_ROOT / spec["output_name"]
    if stage_root.exists():
        shutil.rmtree(stage_root)

    tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec)
    bonn_cmd = apply_ray_args(bonn_cmd, spec)
    bonn_cmd = attach_upstream_args(bonn_cmd, spec)
    bonn_cmd = set_arg(bonn_cmd, "--bonn_dynamic_preset", "all3")
    bonn_cmd = set_arg(bonn_cmd, "--dynamic_sequences", "rgbd_bonn_balloon2,rgbd_bonn_balloon,rgbd_bonn_crowd2")
    bonn_cmd = ensure_flag(bonn_cmd, "--egf_surface_geometry_chain_coupling_enable")
    bonn_cmd = set_arg(bonn_cmd, "--egf_surface_geometry_chain_coupling_mode", "direct")
    bonn_cmd = set_arg(bonn_cmd, "--egf_surface_geometry_chain_coupling_donor_root", spec["surface_geometry_chain_coupling_donor_root"])
    bonn_cmd = set_arg(bonn_cmd, "--egf_surface_geometry_chain_coupling_project_dist", str(spec["surface_geometry_chain_coupling_project_dist"]))

    run(bonn_cmd)
    if TUM_REFERENCE_ROOT.exists():
        shutil.copytree(TUM_REFERENCE_ROOT, root / "tum_oracle" / "oracle", dirs_exist_ok=True)
    row, _tum_details, _bonn_payload = summarize_variant(
        root,
        str(spec["output_name"]),
        frames=args.frames,
        stride=args.stride,
        max_points=args.max_points_per_frame,
        seed=args.seed,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    write_md(MD_PATH, row)
    print("[done]", CSV_PATH)


if __name__ == "__main__":
    main()
