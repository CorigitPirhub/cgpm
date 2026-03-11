from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d

from run_s2_write_time_synthesis import PROJECT_ROOT, dryrun_base_cmds, ensure_flag, run, set_arg
from run_s2_rear_bg_state_recovery import build_variant_cmds as build_rearbg
from run_benchmark import build_dynamic_references


TUM_ALL3 = [
    'rgbd_dataset_freiburg3_walking_xyz',
    'rgbd_dataset_freiburg3_walking_static',
    'rgbd_dataset_freiburg3_walking_halfsphere',
]
BONN_ALL3 = [
    'rgbd_bonn_balloon2',
    'rgbd_bonn_balloon',
    'rgbd_bonn_crowd2',
]
GHOST_THRESH = 0.08
BG_THRESH = 0.05
DATA_BONN = PROJECT_ROOT / 'data' / 'bonn'


def load_csv(path: Path) -> List[dict]:
    return list(csv.DictReader(path.open('r', encoding='utf-8')))


def pick_row(rows: List[dict], sequence: str, method: str) -> dict:
    method = method.lower()
    for row in rows:
        if str(row.get('sequence', '')) == sequence and str(row.get('method', '')).lower() == method:
            return row
    raise KeyError(f'missing row sequence={sequence} method={method}')


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def read_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=float)
    return pts if pts.ndim == 2 else np.zeros((0, 3), dtype=float)


def seq_summary(root: Path, family: str, sequence: str) -> dict:
    if family == 'tum':
        return load_summary(root / 'tum_oracle' / 'oracle' / sequence / 'egf' / 'summary.json')
    return load_summary(root / 'bonn_slam' / 'slam' / sequence / 'egf' / 'summary.json')


def family_metrics(root: Path, family: str, sequences: List[str]) -> Tuple[Dict[str, float], Dict[str, dict]]:
    sub = root / ('tum_oracle/oracle' if family == 'tum' else 'bonn_slam/slam')
    rec = load_csv(sub / 'tables' / 'reconstruction_metrics.csv')
    dyn = load_csv(sub / 'tables' / 'dynamic_metrics.csv')
    accs: List[float] = []
    comps: List[float] = []
    ghost_reds: List[float] = []
    details: Dict[str, dict] = {}
    for seq in sequences:
        regf = pick_row(rec, seq, 'egf')
        rdyn = pick_row(dyn, seq, 'egf')
        rtsdf = pick_row(dyn, seq, 'tsdf')
        acc_cm = float(regf['accuracy']) * 100.0
        comp_r = float(regf['recall_5cm']) * 100.0
        ghost_ratio = float(rdyn['ghost_ratio'])
        tsdf_ghost = float(rtsdf['ghost_ratio'])
        ghost_red = ((tsdf_ghost - ghost_ratio) / max(1e-9, tsdf_ghost)) * 100.0
        accs.append(acc_cm)
        comps.append(comp_r)
        ghost_reds.append(ghost_red)
        details[seq] = {'acc_cm': acc_cm, 'comp_r_5cm': comp_r, 'ghost_reduction_vs_tsdf': ghost_red}
    return {
        'acc_cm_mean': float(mean(accs)),
        'comp_r_5cm_mean': float(mean(comps)),
        'ghost_reduction_vs_tsdf_mean': float(mean(ghost_reds)),
    }, details


def classify_rear_points(sequence: str, rear_points: np.ndarray, *, frames: int, stride: int, max_points: int, seed: int) -> dict:
    stable_bg, tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
        sequence_dir=DATA_BONN / sequence,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points,
        seed=seed,
    )
    pts = np.asarray(rear_points, dtype=float)
    if pts.shape[0] == 0:
        return {
            'rear_points': 0.0,
            'ghost_region_points': 0.0,
            'true_background_points': 0.0,
            'hole_or_noise_points': 0.0,
            'ghost_ratio': 0.0,
            'true_background_ratio': 0.0,
            'hole_or_noise_ratio': 0.0,
        }
    bg_tree = None
    if stable_bg.shape[0] > 0:
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(stable_bg)
        bg_tree = o3d.geometry.KDTreeFlann(bg_pcd)
    ghost = 0
    true_bg = 0
    holes = 0
    for p in pts:
        v = tuple(np.floor(p / float(dynamic_voxel)).astype(np.int32).tolist())
        if v in dynamic_region:
            ghost += 1
            continue
        hit_bg = False
        if bg_tree is not None:
            _, idx, dist2 = bg_tree.search_knn_vector_3d(p.astype(float), 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < BG_THRESH:
                hit_bg = True
        if hit_bg:
            true_bg += 1
        else:
            holes += 1
    total = max(1.0, float(pts.shape[0]))
    return {
        'rear_points': float(pts.shape[0]),
        'ghost_region_points': float(ghost),
        'true_background_points': float(true_bg),
        'hole_or_noise_points': float(holes),
        'ghost_ratio': float(ghost / total),
        'true_background_ratio': float(true_bg / total),
        'hole_or_noise_ratio': float(holes / total),
    }


def summarize_variant(root: Path, name: str, *, frames: int, stride: int, max_points: int, seed: int) -> Tuple[dict, Dict[str, dict], Dict[str, dict]]:
    tum_family, tum_details = family_metrics(root, 'tum', TUM_ALL3)
    bonn_family, bonn_details = family_metrics(root, 'bonn', BONN_ALL3)
    rear_dist: Dict[str, dict] = {}
    for seq in BONN_ALL3:
        rear_pts = read_points(root / 'bonn_slam' / 'slam' / seq / 'egf' / 'rear_surface_points.ply')
        rear_dist[seq] = classify_rear_points(seq, rear_pts, frames=frames, stride=stride, max_points=max_points, seed=seed)
    row = {
        'variant': name,
        'tum_acc_cm': tum_family['acc_cm_mean'],
        'tum_comp_r_5cm': tum_family['comp_r_5cm_mean'],
        'bonn_acc_cm': bonn_family['acc_cm_mean'],
        'bonn_comp_r_5cm': bonn_family['comp_r_5cm_mean'],
        'bonn_ghost_reduction_vs_tsdf': bonn_family['ghost_reduction_vs_tsdf_mean'],
        'bonn_rear_points_sum': sum(v['rear_points'] for v in rear_dist.values()),
        'bonn_rear_true_background_sum': sum(v['true_background_points'] for v in rear_dist.values()),
        'bonn_rear_ghost_sum': sum(v['ghost_region_points'] for v in rear_dist.values()),
        'bonn_rear_hole_or_noise_sum': sum(v['hole_or_noise_points'] for v in rear_dist.values()),
        'decision': 'pending',
    }
    return row, tum_details, {'metrics': bonn_details, 'rear_dist': rear_dist}


def build_commands(base_tum: List[str], base_bonn: List[str], spec: dict) -> Tuple[List[str], List[str], Path]:
    spec_for_build = dict(spec)
    if 'output_name' in spec_for_build:
        spec_for_build['name'] = str(spec_for_build['output_name'])
    tum_cmd, bonn_cmd, root = build_rearbg(base_tum, base_bonn, spec_for_build)
    tum_cmd = set_arg(tum_cmd, '--dynamic_sequences', ','.join(TUM_ALL3))
    bonn_cmd = set_arg(bonn_cmd, '--bonn_dynamic_preset', 'all3')
    bonn_cmd = set_arg(bonn_cmd, '--dynamic_sequences', ','.join(BONN_ALL3))

    common_pairs = [
        ('rps_bank_margin', '--egf_rps_bank_margin'),
        ('rps_bank_separation_ref', '--egf_rps_bank_separation_ref'),
        ('rps_bank_rear_min_score', '--egf_rps_bank_rear_min_score'),
        ('rps_commit_threshold', '--egf_rps_commit_threshold'),
        ('rps_commit_release', '--egf_rps_commit_release'),
        ('rps_commit_age_threshold', '--egf_rps_commit_age_threshold'),
        ('rps_commit_rho_ref', '--egf_rps_commit_rho_ref'),
        ('rps_commit_weight_ref', '--egf_rps_commit_weight_ref'),
        ('rps_commit_min_cand_rho', '--egf_rps_commit_min_cand_rho'),
        ('rps_commit_min_cand_w', '--egf_rps_commit_min_cand_w'),
        ('rps_commit_evidence_weight', '--egf_rps_commit_evidence_weight'),
        ('rps_commit_geometry_weight', '--egf_rps_commit_geometry_weight'),
        ('rps_commit_bg_weight', '--egf_rps_commit_bg_weight'),
        ('rps_commit_static_weight', '--egf_rps_commit_static_weight'),
        ('rps_commit_front_penalty', '--egf_rps_commit_front_penalty'),
    ]
    bonn_only_pairs = [
        ('bonn_rps_admission_support_on', '--egf_rps_admission_support_on'),
        ('bonn_rps_admission_support_gain', '--egf_rps_admission_support_gain'),
        ('bonn_rps_admission_score_relax', '--egf_rps_admission_score_relax'),
        ('bonn_rps_admission_active_floor', '--egf_rps_admission_active_floor'),
        ('bonn_rps_admission_rho_ref', '--egf_rps_admission_rho_ref'),
        ('bonn_rps_admission_weight_ref', '--egf_rps_admission_weight_ref'),
        ('bonn_rps_admission_geometry_weight', '--egf_rps_admission_geometry_weight'),
        ('bonn_rps_admission_geometry_floor', '--egf_rps_admission_geometry_floor'),
        ('bonn_rps_admission_occlusion_weight', '--egf_rps_admission_occlusion_weight'),
        ('bonn_rps_space_redirect_history_weight', '--egf_rps_space_redirect_history_weight'),
        ('bonn_rps_space_redirect_history_bg_weight', '--egf_rps_space_redirect_history_bg_weight'),
        ('bonn_rps_space_redirect_history_static_weight', '--egf_rps_space_redirect_history_static_weight'),
        ('bonn_rps_space_redirect_history_floor', '--egf_rps_space_redirect_history_floor'),
        ('bonn_rps_space_redirect_ghost_suppress_weight', '--egf_rps_space_redirect_ghost_suppress_weight'),
        ('bonn_rps_space_redirect_visual_anchor_weight', '--egf_rps_space_redirect_visual_anchor_weight'),
        ('bonn_rps_space_redirect_visual_anchor_min', '--egf_rps_space_redirect_visual_anchor_min'),
        ('bonn_rps_history_visible_min', '--egf_rps_history_visible_min'),
        ('bonn_rps_obstruction_min', '--egf_rps_obstruction_min'),
        ('bonn_rps_non_hole_min', '--egf_rps_non_hole_min'),
        ('bonn_rps_history_manifold_visible_min', '--egf_rps_history_manifold_visible_min'),
        ('bonn_rps_history_manifold_obstruction_min', '--egf_rps_history_manifold_obstruction_min'),
        ('bonn_rps_history_manifold_bg_weight', '--egf_rps_history_manifold_bg_weight'),
        ('bonn_rps_history_manifold_geo_weight', '--egf_rps_history_manifold_geo_weight'),
        ('bonn_rps_history_manifold_static_weight', '--egf_rps_history_manifold_static_weight'),
        ('bonn_rps_history_manifold_blend', '--egf_rps_history_manifold_blend'),
        ('bonn_rps_history_manifold_max_offset', '--egf_rps_history_manifold_max_offset'),
        ('bonn_rps_bg_manifold_alpha_up', '--egf_rps_bg_manifold_alpha_up'),
        ('bonn_rps_bg_manifold_alpha_down', '--egf_rps_bg_manifold_alpha_down'),
        ('bonn_rps_bg_manifold_rho_alpha', '--egf_rps_bg_manifold_rho_alpha'),
        ('bonn_rps_bg_manifold_weight_gain', '--egf_rps_bg_manifold_weight_gain'),
        ('bonn_rps_bg_manifold_rho_ref', '--egf_rps_bg_manifold_rho_ref'),
        ('bonn_rps_bg_manifold_weight_ref', '--egf_rps_bg_manifold_weight_ref'),
        ('bonn_rps_bg_manifold_history_weight', '--egf_rps_bg_manifold_history_weight'),
        ('bonn_rps_bg_manifold_obstruction_weight', '--egf_rps_bg_manifold_obstruction_weight'),
        ('bonn_rps_bg_manifold_visible_lo', '--egf_rps_bg_manifold_visible_lo'),
        ('bonn_rps_bg_manifold_visible_hi', '--egf_rps_bg_manifold_visible_hi'),
        ('bonn_rps_bg_dense_neighbor_radius', '--egf_rps_bg_dense_neighbor_radius'),
        ('bonn_rps_bg_dense_neighbor_weight', '--egf_rps_bg_dense_neighbor_weight'),
        ('bonn_rps_bg_dense_geometry_weight', '--egf_rps_bg_dense_geometry_weight'),
        ('bonn_rps_bg_dense_max_weight', '--egf_rps_bg_dense_max_weight'),
        ('bonn_rps_bg_dense_support_floor', '--egf_rps_bg_dense_support_floor'),
        ('bonn_rps_bg_dense_decay', '--egf_rps_bg_dense_decay'),
        ('bonn_rps_bg_surface_min_conf', '--egf_rps_bg_surface_min_conf'),
        ('bonn_rps_bg_surface_agree_weight', '--egf_rps_bg_surface_agree_weight'),
        ('bonn_rps_bg_surface_tangent_weight', '--egf_rps_bg_surface_tangent_weight'),
        ('bonn_rps_bg_surface_tangent_floor', '--egf_rps_bg_surface_tangent_floor'),
        ('bonn_rps_bg_bridge_min_visible', '--egf_rps_bg_bridge_min_visible'),
        ('bonn_rps_bg_bridge_min_obstruction', '--egf_rps_bg_bridge_min_obstruction'),
        ('bonn_rps_bg_bridge_min_step', '--egf_rps_bg_bridge_min_step'),
        ('bonn_rps_bg_bridge_max_step', '--egf_rps_bg_bridge_max_step'),
        ('bonn_rps_bg_bridge_gain', '--egf_rps_bg_bridge_gain'),
        ('bonn_rps_bg_bridge_phi_blend', '--egf_rps_bg_bridge_phi_blend'),
        ('bonn_rps_bg_bridge_target_dyn_max', '--egf_rps_bg_bridge_target_dyn_max'),
        ('bonn_rps_bg_bridge_target_surface_max', '--egf_rps_bg_bridge_target_surface_max'),
        ('bonn_rps_bg_bridge_ghost_suppress_weight', '--egf_rps_bg_bridge_ghost_suppress_weight'),
        ('bonn_rps_bg_bridge_relaxed_dyn_max', '--egf_rps_bg_bridge_relaxed_dyn_max'),
        ('bonn_rps_bg_bridge_max_hits_per_source', '--egf_rps_bg_bridge_max_hits_per_source'),
        ('bonn_rps_bg_bridge_cone_radius_cells', '--egf_rps_bg_bridge_cone_radius_cells'),
        ('bonn_rps_bg_bridge_cone_gain_scale', '--egf_rps_bg_bridge_cone_gain_scale'),
        ('bonn_rps_bg_bridge_patch_radius_cells', '--egf_rps_bg_bridge_patch_radius_cells'),
        ('bonn_rps_bg_bridge_patch_gain_scale', '--egf_rps_bg_bridge_patch_gain_scale'),
        ('bonn_rps_bg_bridge_depth_hypothesis_count', '--egf_rps_bg_bridge_depth_hypothesis_count'),
        ('bonn_rps_bg_bridge_depth_step_scale', '--egf_rps_bg_bridge_depth_step_scale'),
        ('bonn_rps_bg_bridge_rear_support_gain', '--egf_rps_bg_bridge_rear_support_gain'),
        ('bonn_rps_bg_bridge_rear_rho_gain', '--egf_rps_bg_bridge_rear_rho_gain'),
        ('bonn_rps_bg_bridge_rear_phi_blend', '--egf_rps_bg_bridge_rear_phi_blend'),
        ('bonn_rps_bg_bridge_rear_score_floor', '--egf_rps_bg_bridge_rear_score_floor'),
        ('bonn_rps_bg_bridge_rear_active_floor', '--egf_rps_bg_bridge_rear_active_floor'),
        ('bonn_rps_bg_bridge_rear_age_floor', '--egf_rps_bg_bridge_rear_age_floor'),
        ('bonn_rps_rear_hybrid_bridge_support_min', '--egf_rps_rear_hybrid_bridge_support_min'),
        ('bonn_rps_rear_hybrid_dyn_max', '--egf_rps_rear_hybrid_dyn_max'),
        ('bonn_rps_rear_hybrid_manifold_min', '--egf_rps_rear_hybrid_manifold_min'),
        ('bonn_rps_rear_density_radius_cells', '--egf_rps_rear_density_radius_cells'),
        ('bonn_rps_rear_density_min_neighbors', '--egf_rps_rear_density_min_neighbors'),
        ('bonn_rps_rear_density_support_min', '--egf_rps_rear_density_support_min'),
        ('bonn_rps_rear_selectivity_support_weight', '--egf_rps_rear_selectivity_support_weight'),
        ('bonn_rps_rear_selectivity_history_weight', '--egf_rps_rear_selectivity_history_weight'),
        ('bonn_rps_rear_selectivity_static_weight', '--egf_rps_rear_selectivity_static_weight'),
        ('bonn_rps_rear_selectivity_geom_weight', '--egf_rps_rear_selectivity_geom_weight'),
        ('bonn_rps_rear_selectivity_bridge_weight', '--egf_rps_rear_selectivity_bridge_weight'),
        ('bonn_rps_rear_selectivity_density_weight', '--egf_rps_rear_selectivity_density_weight'),
        ('bonn_rps_rear_selectivity_rear_score_weight', '--egf_rps_rear_selectivity_rear_score_weight'),
        ('bonn_rps_rear_selectivity_front_score_weight', '--egf_rps_rear_selectivity_front_score_weight'),
        ('bonn_rps_rear_selectivity_competition_weight', '--egf_rps_rear_selectivity_competition_weight'),
        ('bonn_rps_rear_selectivity_competition_alpha', '--egf_rps_rear_selectivity_competition_alpha'),
        ('bonn_rps_rear_selectivity_gap_weight', '--egf_rps_rear_selectivity_gap_weight'),
        ('bonn_rps_rear_selectivity_sep_weight', '--egf_rps_rear_selectivity_sep_weight'),
        ('bonn_rps_rear_selectivity_dyn_weight', '--egf_rps_rear_selectivity_dyn_weight'),
        ('bonn_rps_rear_selectivity_ghost_weight', '--egf_rps_rear_selectivity_ghost_weight'),
        ('bonn_rps_rear_selectivity_front_weight', '--egf_rps_rear_selectivity_front_weight'),
        ('bonn_rps_rear_selectivity_geom_risk_weight', '--egf_rps_rear_selectivity_geom_risk_weight'),
        ('bonn_rps_rear_selectivity_history_risk_weight', '--egf_rps_rear_selectivity_history_risk_weight'),
        ('bonn_rps_rear_selectivity_density_risk_weight', '--egf_rps_rear_selectivity_density_risk_weight'),
        ('bonn_rps_rear_selectivity_bridge_relief_weight', '--egf_rps_rear_selectivity_bridge_relief_weight'),
        ('bonn_rps_rear_selectivity_static_relief_weight', '--egf_rps_rear_selectivity_static_relief_weight'),
        ('bonn_rps_rear_selectivity_gap_risk_weight', '--egf_rps_rear_selectivity_gap_risk_weight'),
        ('bonn_rps_rear_selectivity_score_min', '--egf_rps_rear_selectivity_score_min'),
        ('bonn_rps_rear_selectivity_risk_max', '--egf_rps_rear_selectivity_risk_max'),
        ('bonn_rps_rear_selectivity_geom_floor', '--egf_rps_rear_selectivity_geom_floor'),
        ('bonn_rps_rear_selectivity_history_floor', '--egf_rps_rear_selectivity_history_floor'),
        ('bonn_rps_rear_selectivity_bridge_floor', '--egf_rps_rear_selectivity_bridge_floor'),
        ('bonn_rps_rear_selectivity_competition_floor', '--egf_rps_rear_selectivity_competition_floor'),
        ('bonn_rps_rear_selectivity_front_score_max', '--egf_rps_rear_selectivity_front_score_max'),
        ('bonn_rps_rear_selectivity_gap_min', '--egf_rps_rear_selectivity_gap_min'),
        ('bonn_rps_rear_selectivity_gap_max', '--egf_rps_rear_selectivity_gap_max'),
        ('bonn_rps_rear_selectivity_gap_valid_min', '--egf_rps_rear_selectivity_gap_valid_min'),
        ('bonn_rps_rear_selectivity_occlusion_order_weight', '--egf_rps_rear_selectivity_occlusion_order_weight'),
        ('bonn_rps_rear_selectivity_occlusion_order_floor', '--egf_rps_rear_selectivity_occlusion_order_floor'),
        ('bonn_rps_rear_selectivity_occlusion_order_risk_weight', '--egf_rps_rear_selectivity_occlusion_order_risk_weight'),
        ('bonn_rps_rear_selectivity_local_conflict_weight', '--egf_rps_rear_selectivity_local_conflict_weight'),
        ('bonn_rps_rear_selectivity_local_conflict_max', '--egf_rps_rear_selectivity_local_conflict_max'),
        ('bonn_rps_rear_selectivity_front_residual_weight', '--egf_rps_rear_selectivity_front_residual_weight'),
        ('bonn_rps_rear_selectivity_front_residual_max', '--egf_rps_rear_selectivity_front_residual_max'),
        ('bonn_rps_rear_selectivity_occluder_protect_weight', '--egf_rps_rear_selectivity_occluder_protect_weight'),
        ('bonn_rps_rear_selectivity_occluder_protect_floor', '--egf_rps_rear_selectivity_occluder_protect_floor'),
        ('bonn_rps_rear_selectivity_occluder_relief_weight', '--egf_rps_rear_selectivity_occluder_relief_weight'),
        ('bonn_rps_rear_selectivity_dynamic_trail_weight', '--egf_rps_rear_selectivity_dynamic_trail_weight'),
        ('bonn_rps_rear_selectivity_dynamic_trail_max', '--egf_rps_rear_selectivity_dynamic_trail_max'),
        ('bonn_rps_rear_selectivity_dynamic_trail_relief_weight', '--egf_rps_rear_selectivity_dynamic_trail_relief_weight'),
        ('bonn_rps_rear_selectivity_history_anchor_weight', '--egf_rps_rear_selectivity_history_anchor_weight'),
        ('bonn_rps_rear_selectivity_history_anchor_floor', '--egf_rps_rear_selectivity_history_anchor_floor'),
        ('bonn_rps_rear_selectivity_history_anchor_relief_weight', '--egf_rps_rear_selectivity_history_anchor_relief_weight'),
        ('bonn_rps_rear_selectivity_surface_anchor_weight', '--egf_rps_rear_selectivity_surface_anchor_weight'),
        ('bonn_rps_rear_selectivity_surface_anchor_floor', '--egf_rps_rear_selectivity_surface_anchor_floor'),
        ('bonn_rps_rear_selectivity_surface_anchor_risk_weight', '--egf_rps_rear_selectivity_surface_anchor_risk_weight'),
        ('bonn_rps_rear_selectivity_surface_distance_ref', '--egf_rps_rear_selectivity_surface_distance_ref'),
        ('bonn_rps_rear_selectivity_dynamic_shell_weight', '--egf_rps_rear_selectivity_dynamic_shell_weight'),
        ('bonn_rps_rear_selectivity_dynamic_shell_max', '--egf_rps_rear_selectivity_dynamic_shell_max'),
        ('bonn_rps_rear_selectivity_dynamic_shell_gap_ref', '--egf_rps_rear_selectivity_dynamic_shell_gap_ref'),
        ('bonn_rps_rear_selectivity_conflict_radius_cells', '--egf_rps_rear_selectivity_conflict_radius_cells'),
        ('bonn_rps_rear_selectivity_conflict_front_score_min', '--egf_rps_rear_selectivity_conflict_front_score_min'),
        ('bonn_rps_rear_selectivity_conflict_static_score_min', '--egf_rps_rear_selectivity_conflict_static_score_min'),
        ('bonn_rps_rear_selectivity_conflict_dist_scale', '--egf_rps_rear_selectivity_conflict_dist_scale'),
        ('bonn_rps_rear_selectivity_conflict_gap_ref', '--egf_rps_rear_selectivity_conflict_gap_ref'),
        ('bonn_rps_rear_selectivity_conflict_ref', '--egf_rps_rear_selectivity_conflict_ref'),
        ('bonn_rps_rear_selectivity_trail_radius_cells', '--egf_rps_rear_selectivity_trail_radius_cells'),
        ('bonn_rps_rear_selectivity_trail_ref', '--egf_rps_rear_selectivity_trail_ref'),
        ('bonn_rps_rear_selectivity_density_radius_cells', '--egf_rps_rear_selectivity_density_radius_cells'),
        ('bonn_rps_rear_selectivity_density_ref', '--egf_rps_rear_selectivity_density_ref'),
        ('bonn_rps_rear_selectivity_topk', '--egf_rps_rear_selectivity_topk'),
        ('bonn_rps_rear_selectivity_rank_risk_weight', '--egf_rps_rear_selectivity_rank_risk_weight'),
        ('bonn_rps_rear_state_decay_relax', '--egf_rps_rear_state_decay_relax'),
        ('bonn_rps_rear_state_active_floor', '--egf_rps_rear_state_active_floor'),
    ]
    for cmd_name in ('tum', 'bonn'):
        cmd = tum_cmd if cmd_name == 'tum' else bonn_cmd
        if bool(spec.get('rps_commit_activation_enable', False)):
            cmd = ensure_flag(cmd, '--egf_rps_commit_activation_enable')
        if bool(spec.get('rps_surface_bank_enable', False)):
            cmd = ensure_flag(cmd, '--egf_rps_surface_bank_enable')
        if cmd_name == 'bonn':
            for key, flag in [
                ('bonn_rps_admission_support_enable', '--egf_rps_admission_support_enable'),
                ('bonn_rps_admission_geometry_enable', '--egf_rps_admission_geometry_enable'),
                ('bonn_rps_admission_occlusion_enable', '--egf_rps_admission_occlusion_enable'),
                ('bonn_rps_space_redirect_history_enable', '--egf_rps_space_redirect_history_enable'),
                ('bonn_rps_space_redirect_ghost_suppress_enable', '--egf_rps_space_redirect_ghost_suppress_enable'),
                ('bonn_rps_space_redirect_visual_anchor_enable', '--egf_rps_space_redirect_visual_anchor_enable'),
                ('bonn_rps_history_obstructed_gate_enable', '--egf_rps_history_obstructed_gate_enable'),
                ('bonn_rps_history_manifold_enable', '--egf_rps_history_manifold_enable'),
                ('bonn_rps_bg_manifold_state_enable', '--egf_rps_bg_manifold_state_enable'),
                ('bonn_rps_bg_dense_state_enable', '--egf_rps_bg_dense_state_enable'),
                ('bonn_rps_bg_surface_constrained_enable', '--egf_rps_bg_surface_constrained_enable'),
                ('bonn_rps_bg_surface_tangent_enable', '--egf_rps_bg_surface_tangent_enable'),
                ('bonn_rps_bg_bridge_enable', '--egf_rps_bg_bridge_enable'),
                ('bonn_rps_bg_bridge_ghost_suppress_enable', '--egf_rps_bg_bridge_ghost_suppress_enable'),
                ('bonn_rps_bg_bridge_keep_multi_hits', '--egf_rps_bg_bridge_keep_multi_hits'),
                ('bonn_rps_bg_bridge_cone_enable', '--egf_rps_bg_bridge_cone_enable'),
                ('bonn_rps_bg_bridge_rear_synth_enable', '--egf_rps_bg_bridge_rear_synth_enable'),
                ('bonn_rps_rear_hybrid_filter_enable', '--egf_rps_rear_hybrid_filter_enable'),
                ('bonn_rps_rear_density_gate_enable', '--egf_rps_rear_density_gate_enable'),
                ('bonn_rps_rear_selectivity_enable', '--egf_rps_rear_selectivity_enable'),
                ('bonn_rps_rear_state_protect_enable', '--egf_rps_rear_state_protect_enable'),
            ]:
                if bool(spec.get(key, False)):
                    cmd = ensure_flag(cmd, flag)
        if cmd_name == 'tum':
            tum_cmd = cmd
        else:
            bonn_cmd = cmd
    for key, flag in common_pairs:
        if key in spec:
            tum_cmd = set_arg(tum_cmd, flag, str(spec[key]))
            bonn_cmd = set_arg(bonn_cmd, flag, str(spec[key]))
    for key, flag in bonn_only_pairs:
        if key in spec:
            bonn_cmd = set_arg(bonn_cmd, flag, str(spec[key]))
    return tum_cmd, bonn_cmd, root


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        better = row['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6
        quality_ok = row['bonn_rear_ghost_sum'] <= control['bonn_rear_ghost_sum'] + 10.0
        no_regress = row['tum_comp_r_5cm'] >= control['tum_comp_r_5cm'] - 0.50 and row['bonn_acc_cm'] <= control['bonn_acc_cm'] * 1.10
        row['decision'] = 'iterate' if better and quality_ok and no_regress else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(iters, key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum']))
        for row in iters:
            if row is not best:
                row['decision'] = 'abandon'


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# S2 rear geometry quality compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution_report(control_payload: Dict[str, dict], path_md: Path) -> None:
    lines = [
        '# S2 rear-point spatial distribution diagnosis',
        '',
        '日期：`2026-03-09`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '控制组：`38_bonn_state_protect`',
        '',
        '## 1. 诊断目的',
        '- 分析 `38` 下真正导出的 rear points 究竟落在 Ghost Region、True Background 还是 Hole/Noise 区域。',
        '- 判断“admission 已增加，但几何质量转化不足”是否来自 rear point 空间分布无效。',
        '',
        '## 2. 38 配置下的 Bonn rear-point 分布',
    ]
    family = control_payload['rear_dist']
    for seq in BONN_ALL3:
        d = family[seq]
        lines.extend([
            f"### `{seq}`",
            f"- rear points: `{d['rear_points']:.0f}`",
            f"- true background: `{d['true_background_points']:.0f}` ({100.0 * d['true_background_ratio']:.2f}%)",
            f"- ghost region: `{d['ghost_region_points']:.0f}` ({100.0 * d['ghost_ratio']:.2f}%)",
            f"- holes / noise: `{d['hole_or_noise_points']:.0f}` ({100.0 * d['hole_or_noise_ratio']:.2f}%)",
            '',
        ])
    total_rear = sum(v['rear_points'] for v in family.values())
    total_bg = sum(v['true_background_points'] for v in family.values())
    total_ghost = sum(v['ghost_region_points'] for v in family.values())
    total_hole = sum(v['hole_or_noise_points'] for v in family.values())
    lines.extend([
        '## 3. family-mean 结论',
        f"- total rear points: `{total_rear:.0f}`",
        f"- total true background: `{total_bg:.0f}`",
        f"- total ghost region: `{total_ghost:.0f}`",
        f"- total holes / noise: `{total_hole:.0f}`",
        '- 若 `hole_or_noise` 占比高，说明 admission 增加后的 rear points 仍然缺乏稳定背景锚定，更多是在填无效区域。',
        '- 若 `ghost_region` 占比高，说明 rear points 仍落在动态物体历史区域，几何一致性不足。',
        '- 若 `true_background` 增长但 `Comp-R` 仍不动，则说明点的空间覆盖或连通性仍不足。',
    ])
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], control_payload: Dict[str, dict], best_payload: Dict[str, dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    best = max(rows[1:], key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum']))
    lines = [
        '# S2 rear geometry quality analysis',
        '',
        '日期：`2026-03-09`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`processes/s2/S2_RPS_REAR_GEOMETRY_QUALITY_COMPARE.csv`',
        '',
        '## 1. 结果概览',
    ]
    for row in rows:
        lines.append(
            f"- `{row['variant']}`: Bonn `ghost_reduction_vs_tsdf = {row['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {row['bonn_comp_r_5cm']:.2f}%`, `rear_true_background_sum = {row['bonn_rear_true_background_sum']:.0f}`, `rear_ghost_sum = {row['bonn_rear_ghost_sum']:.0f}`, decision=`{row['decision']}`"
        )
    lines.extend([
        '',
        '## 2. 几何质量策略是否有效',
        f"- 控制组 `38`：Bonn `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
        f"- 本轮最佳候选 `{best['variant']}`：Bonn `ghost_reduction_vs_tsdf = {best['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
    ])
    if best['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6:
        lines.append(f"- 结论：几何质量改进有效，ghost 提升了 `{best['bonn_ghost_reduction_vs_tsdf'] - control['bonn_ghost_reduction_vs_tsdf']:.2f}` 个百分点。")
    else:
        lines.append('- 结论：几何质量改进未能继续提升 ghost。')
    lines.extend([
        '',
        '## 3. Ghost 与 Comp-R 是否同步提升',
        f"- 控制组 Bonn `Comp-R = {control['bonn_comp_r_5cm']:.2f}%`，最佳候选为 `{best['bonn_comp_r_5cm']:.2f}%`。",
    ])
    if best['bonn_comp_r_5cm'] > control['bonn_comp_r_5cm'] + 1e-6 and best['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6:
        lines.append('- 结论：Ghost 与 Comp-R 同步提升。')
    else:
        lines.append('- 结论：本轮仍未打破 `Comp-R` 僵局，说明几何质量转化仍不充分。')
    lines.extend([
        '',
        '## 4. 若未达标，主要原因是什么',
    ])
    if best['bonn_rear_ghost_sum'] > control['bonn_rear_ghost_sum']:
        lines.append('- 新增 rear points 仍有相当部分落在 ghost region，说明 admission 质量仍不足。')
    elif best['bonn_rear_hole_or_noise_sum'] > control['bonn_rear_hole_or_noise_sum']:
        lines.append('- 新增 rear points 更多落在 hole/noise 区域，说明几何一致性虽增强，但空间覆盖仍不稳定。')
    else:
        lines.append('- Rear point 的空间质量已有改善，但 Export 后的表面连通性/密度仍不足，Comp-R 仍被限制。')
    lines.extend([
        '',
        '## 5. 当前阶段判断',
        '- 本轮若未同时达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过。',
        '- 当前不能进入 `S3`。',
    ])
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 rear geometry quality runner.')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=600)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    base_tum, base_bonn = dryrun_base_cmds(args.frames, args.stride, args.max_points_per_frame)
    base_spec = dict(
        frames=args.frames,
        stride=args.stride,
        max_points=args.max_points_per_frame,
        rps_candidate_rescue_enable=True,
        rps_candidate_support_gain=0.36,
        rps_candidate_bg_gain=0.30,
        rps_candidate_rho_gain=0.22,
        rps_candidate_front_relax=0.38,
        joint_bg_state_enable=True,
        joint_bg_state_on=0.15,
        joint_bg_state_gain=0.76,
        joint_bg_state_rho_gain=0.30,
        joint_bg_state_front_penalty=0.10,
        rps_commit_activation_enable=True,
        rps_commit_threshold=0.34,
        rps_commit_release=0.22,
        rps_commit_age_threshold=1.0,
        rps_commit_rho_ref=0.035,
        rps_commit_weight_ref=0.35,
        rps_commit_min_cand_rho=0.006,
        rps_commit_min_cand_w=0.025,
        rps_commit_evidence_weight=0.38,
        rps_commit_geometry_weight=0.32,
        rps_commit_bg_weight=0.20,
        rps_commit_static_weight=0.10,
        rps_commit_front_penalty=0.12,
        rps_surface_bank_enable=True,
        rps_bank_margin=0.02,
        rps_bank_separation_ref=0.02,
        rps_bank_rear_min_score=0.18,
        bonn_rps_admission_support_enable=True,
        bonn_rps_admission_support_on=0.36,
        bonn_rps_admission_support_gain=0.60,
        bonn_rps_admission_score_relax=0.06,
        bonn_rps_admission_active_floor=0.28,
        bonn_rps_admission_rho_ref=0.07,
        bonn_rps_admission_weight_ref=0.30,
        bonn_rps_rear_state_protect_enable=True,
        bonn_rps_rear_state_decay_relax=0.65,
        bonn_rps_rear_state_active_floor=0.34,
    )
    variants = [
        dict(base_spec, name='38_bonn_state_protect', output_name='38_bonn_state_protect_geom_quality_control'),
        dict(
            base_spec,
            name='40_bonn_geometry_aligned_admission',
            bonn_rps_admission_geometry_enable=True,
            bonn_rps_admission_geometry_weight=0.30,
            bonn_rps_admission_geometry_floor=0.45,
        ),
        dict(
            base_spec,
            name='41_bonn_geometry_occlusion_admission',
            bonn_rps_admission_geometry_enable=True,
            bonn_rps_admission_geometry_weight=0.28,
            bonn_rps_admission_geometry_floor=0.42,
            bonn_rps_admission_occlusion_enable=True,
            bonn_rps_admission_occlusion_weight=0.16,
            bonn_rps_admission_support_on=0.34,
        ),
        dict(
            base_spec,
            name='42_bonn_geometry_density_gate',
            bonn_rps_admission_geometry_enable=True,
            bonn_rps_admission_geometry_weight=0.28,
            bonn_rps_admission_geometry_floor=0.44,
            bonn_rps_admission_occlusion_enable=True,
            bonn_rps_admission_occlusion_weight=0.14,
            bonn_rps_rear_density_gate_enable=True,
            bonn_rps_rear_density_radius_cells=1,
            bonn_rps_rear_density_min_neighbors=1,
            bonn_rps_rear_density_support_min=0.52,
        ),
    ]

    rows: List[dict] = []
    payloads: Dict[str, dict] = {}
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec)
        run(tum_cmd)
        run(bonn_cmd)
        row, tum_details, bonn_payload = summarize_variant(root, str(spec['name']), frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame, seed=args.seed)
        rows.append(row)
        payloads[str(spec['name'])] = {'tum': tum_details, **bonn_payload}

    decide(rows)
    out_dir = PROJECT_ROOT / 'processes' / 's2'
    compare_csv = out_dir / 'S2_RPS_REAR_GEOMETRY_QUALITY_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_REAR_GEOMETRY_QUALITY_COMPARE.md'
    dist_md = out_dir / 'S2_RPS_REAR_GEOMETRY_QUALITY_DISTRIBUTION.md'
    analysis_md = out_dir / 'S2_RPS_REAR_GEOMETRY_QUALITY_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_distribution_report(payloads['38_bonn_state_protect'], dist_md)
    best_name = max(rows[1:], key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum']))['variant']
    write_analysis(rows, payloads['38_bonn_state_protect'], payloads[best_name], analysis_md)
    print(f'[done] {compare_csv} {compare_md} {dist_md} {analysis_md}')


if __name__ == '__main__':
    main()
