#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


COMMON_NUMERIC_FIELDS = [
    "points",
    "accuracy",
    "completeness",
    "chamfer",
    "hausdorff",
    "precision",
    "recall",
    "fscore",
    "normal_consistency",
    "precision_2cm",
    "recall_2cm",
    "fscore_2cm",
    "precision_5cm",
    "recall_5cm",
    "fscore_5cm",
    "precision_10cm",
    "recall_10cm",
    "fscore_10cm",
    "ghost_count",
    "ghost_ratio",
    "roi_ghost_count",
    "roi_ghost_ratio",
    "ghost_tail_count",
    "ghost_tail_ratio",
    "background_recovery",
    "roi_background_recovery",
    "ate_rmse",
    "ate_mean",
    "ate_median",
    "ate_max",
    "rpe_trans_rmse",
    "rpe_trans_mean",
    "rpe_rot_deg_rmse",
    "rpe_rot_deg_mean",
    "traj_frame_count",
    "traj_valid_pair_count",
    "traj_finite_ratio",
    "odom_valid_ratio",
    "odom_fitness_mean",
    "odom_rmse_mean",
]

SIG_METRICS: Sequence[Tuple[str, str]] = [
    ("fscore", "higher"),
    ("recall_5cm", "higher"),
    ("chamfer", "lower"),
    ("ghost_ratio", "lower"),
    ("ghost_tail_ratio", "lower"),
    ("background_recovery", "higher"),
]


def _to_float(v: object) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def _to_int(v: object, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def _read_rows(path: Path, protocol_fallback: str, source_tag: str) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows: List[Dict[str, object]] = []
        for r in reader:
            row: Dict[str, object] = dict(r)
            row["protocol"] = str(row.get("protocol", "")).strip().lower() or protocol_fallback
            row["scene_type"] = str(row.get("scene_type", "")).strip().lower()
            row["sequence"] = str(row.get("sequence", "")).strip()
            row["method"] = str(row.get("method", "")).strip().lower()
            row["seed"] = _to_int(row.get("seed", 0), default=0)
            row["source_tag"] = source_tag
            for k in headers:
                fv = _to_float(row.get(k))
                if fv is not None:
                    row[k] = fv
            rows.append(row)
    return rows


def _dedupe(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    table: Dict[Tuple[str, str, str, str, int], Dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("protocol", "")),
            str(row.get("scene_type", "")),
            str(row.get("sequence", "")),
            str(row.get("method", "")),
            int(row.get("seed", 0)),
        )
        table[key] = row
    out = list(table.values())
    out.sort(key=lambda r: (str(r.get("protocol", "")), str(r.get("scene_type", "")), str(r.get("sequence", "")), str(r.get("method", "")), int(r.get("seed", 0))))
    return out


def _collect_numeric_fields(rows: Iterable[Dict[str, object]]) -> List[str]:
    avail: List[str] = []
    for k in COMMON_NUMERIC_FIELDS:
        if any(isinstance(r.get(k), (int, float)) for r in rows):
            avail.append(k)
    return avail


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for row in rows:
            w.writerow({h: row.get(h, "") for h in headers})


def _aggregate(rows: List[Dict[str, object]]) -> Tuple[List[str], List[Dict[str, object]]]:
    numeric = _collect_numeric_fields(rows)
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("protocol", "")),
            str(row.get("scene_type", "")),
            str(row.get("sequence", "")),
            str(row.get("method", "")),
        )
        groups[key].append(row)

    out_rows: List[Dict[str, object]] = []
    for (protocol, scene_type, sequence, method), g in sorted(groups.items()):
        out: Dict[str, object] = {
            "protocol": protocol,
            "scene_type": scene_type,
            "sequence": sequence,
            "method": method,
            "n": len(g),
        }
        for k in numeric:
            vals = [float(r[k]) for r in g if isinstance(r.get(k), (int, float))]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out[k] = float(np.mean(arr))
            out[f"{k}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        out_rows.append(out)

    headers = ["protocol", "scene_type", "sequence", "method", "n"]
    for k in numeric:
        headers.extend([k, f"{k}_std"])
    return headers, out_rows


def _paired_metric(
    rows: List[Dict[str, object]],
    protocol: str,
    scene_type: str,
    baseline: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    egf: Dict[Tuple[str, int], float] = {}
    base: Dict[Tuple[str, int], float] = {}
    for r in rows:
        if str(r.get("protocol", "")) != protocol:
            continue
        if str(r.get("scene_type", "")) != scene_type:
            continue
        method = str(r.get("method", ""))
        seq = str(r.get("sequence", ""))
        seed = int(r.get("seed", 0))
        val = r.get(metric)
        if not isinstance(val, (int, float)):
            continue
        if method == "egf":
            egf[(seq, seed)] = float(val)
        elif method == baseline:
            base[(seq, seed)] = float(val)
    keys = sorted(set(egf.keys()).intersection(base.keys()))
    if not keys:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    return (
        np.asarray([egf[k] for k in keys], dtype=float),
        np.asarray([base[k] for k in keys], dtype=float),
    )


def _build_significance(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    methods = sorted({str(r.get("method", "")) for r in rows if str(r.get("method", ""))})
    baselines = [m for m in methods if m != "egf"]
    protocols = sorted({str(r.get("protocol", "")) for r in rows if str(r.get("protocol", ""))})
    scenes = sorted({str(r.get("scene_type", "")) for r in rows if str(r.get("scene_type", ""))})

    out: List[Dict[str, object]] = []
    for protocol in protocols:
        for scene in scenes:
            for baseline in baselines:
                for metric, direction in SIG_METRICS:
                    a, b = _paired_metric(rows, protocol, scene, baseline, metric)
                    if a.size == 0:
                        continue
                    raw = a - b
                    improve = raw if direction == "higher" else -raw
                    if improve.size > 1:
                        t_stat, t_p = stats.ttest_rel(a, b)
                        if direction == "lower":
                            t_stat = -t_stat
                        try:
                            w_stat, w_p = stats.wilcoxon(improve)
                            w_stat = float(w_stat)
                            w_p = float(w_p)
                        except Exception:
                            w_stat, w_p = float("nan"), float("nan")
                    else:
                        t_stat, t_p, w_stat, w_p = float("nan"), float("nan"), float("nan"), float("nan")
                    out.append(
                        {
                            "protocol": protocol,
                            "scene_type": scene,
                            "metric": metric,
                            "direction": direction,
                            "method_a": "egf",
                            "method_b": baseline,
                            "n_pairs": int(improve.size),
                            "mean_delta_egf_minus_baseline": float(np.mean(raw)),
                            "mean_improvement": float(np.mean(improve)),
                            "t_stat": float(t_stat),
                            "t_pvalue": float(t_p),
                            "wilcoxon_stat": w_stat,
                            "wilcoxon_pvalue": w_p,
                        }
                    )
    out.sort(key=lambda r: (str(r["protocol"]), str(r["scene_type"]), str(r["metric"]), str(r["method_b"])))
    return out


def _copy_with_protocol(src: Path, protocol: str, source_tag: str) -> List[Dict[str, object]]:
    if not src.exists():
        return []
    return _read_rows(src, protocol_fallback=protocol, source_tag=source_tag)


def _as_float(row: Dict[str, object], key: str) -> float | None:
    v = row.get(key)
    if isinstance(v, (int, float)):
        x = float(v)
        if math.isfinite(x):
            return x
    return None


def _build_local_mapping_tables(
    recon_agg_rows: Sequence[Dict[str, object]],
    dyn_agg_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    recon_idx: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    dyn_idx: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for r in recon_agg_rows:
        key = (
            str(r.get("protocol", "")),
            str(r.get("scene_type", "")),
            str(r.get("sequence", "")),
            str(r.get("method", "")),
        )
        recon_idx[key] = r
    for r in dyn_agg_rows:
        key = (
            str(r.get("protocol", "")),
            str(r.get("scene_type", "")),
            str(r.get("sequence", "")),
            str(r.get("method", "")),
        )
        dyn_idx[key] = r

    keys = sorted(set(recon_idx.keys()).union(dyn_idx.keys()))
    mean_std_rows: List[Dict[str, object]] = []
    main_rows: List[Dict[str, object]] = []

    for key in keys:
        protocol, scene_type, sequence, method = key
        rr = recon_idx.get(key, {})
        dr = dyn_idx.get(key, {})
        mean_std_row: Dict[str, object] = {
            "protocol": protocol,
            "scene_type": scene_type,
            "sequence": sequence,
            "method": method,
            "n_recon": int(rr.get("n", 0) or 0),
            "n_dynamic": int(dr.get("n", 0) or 0),
            "fscore_mean": _as_float(rr, "fscore"),
            "fscore_std": _as_float(rr, "fscore_std"),
            "recall_5cm_mean": _as_float(rr, "recall_5cm"),
            "recall_5cm_std": _as_float(rr, "recall_5cm_std"),
            "precision_5cm_mean": _as_float(rr, "precision_5cm"),
            "precision_5cm_std": _as_float(rr, "precision_5cm_std"),
            "chamfer_mean": _as_float(rr, "chamfer"),
            "chamfer_std": _as_float(rr, "chamfer_std"),
            "ghost_ratio_mean": _as_float(dr, "ghost_ratio"),
            "ghost_ratio_std": _as_float(dr, "ghost_ratio_std"),
            "ghost_tail_ratio_mean": _as_float(dr, "ghost_tail_ratio"),
            "ghost_tail_ratio_std": _as_float(dr, "ghost_tail_ratio_std"),
            "background_recovery_mean": _as_float(dr, "background_recovery"),
            "background_recovery_std": _as_float(dr, "background_recovery_std"),
        }
        mean_std_rows.append(mean_std_row)

        main_rows.append(
            {
                "protocol": protocol,
                "scene_type": scene_type,
                "sequence": sequence,
                "method": method,
                "n": max(int(rr.get("n", 0) or 0), int(dr.get("n", 0) or 0)),
                "fscore": _as_float(rr, "fscore"),
                "recall_5cm": _as_float(rr, "recall_5cm"),
                "precision_5cm": _as_float(rr, "precision_5cm"),
                "chamfer": _as_float(rr, "chamfer"),
                "ghost_ratio": _as_float(dr, "ghost_ratio"),
                "ghost_tail_ratio": _as_float(dr, "ghost_tail_ratio"),
                "background_recovery": _as_float(dr, "background_recovery"),
            }
        )

    return main_rows, mean_std_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified oracle/slam multi-seed summary tables for reconstruction/dynamic metrics."
    )
    parser.add_argument("--summary_root", type=str, default="output/summary_tables")
    parser.add_argument(
        "--oracle_tables_root",
        type=str,
        default="output/tmp/p4_multiseed_tum_final_v2/oracle/tables",
    )
    parser.add_argument(
        "--slam_tables_root",
        type=str,
        default="output/tmp/p5_multiseed_bonn_all3/slam/tables",
    )
    parser.add_argument("--oracle_tag", type=str, default="tum_oracle_multiseed")
    parser.add_argument("--slam_tag", type=str, default="bonn_slam_multiseed")
    args = parser.parse_args()

    summary_root = Path(args.summary_root)
    oracle_root = Path(args.oracle_tables_root)
    slam_root = Path(args.slam_tables_root)

    oracle_recon = oracle_root / "reconstruction_metrics.csv"
    oracle_dyn = oracle_root / "dynamic_metrics.csv"
    slam_recon = slam_root / "reconstruction_metrics.csv"
    slam_dyn = slam_root / "dynamic_metrics.csv"

    recon_rows = _dedupe(
        _copy_with_protocol(oracle_recon, protocol="oracle", source_tag=args.oracle_tag)
        + _copy_with_protocol(slam_recon, protocol="slam", source_tag=args.slam_tag)
    )
    dyn_rows = _dedupe(
        _copy_with_protocol(oracle_dyn, protocol="oracle", source_tag=args.oracle_tag)
        + _copy_with_protocol(slam_dyn, protocol="slam", source_tag=args.slam_tag)
    )

    if not recon_rows and not dyn_rows:
        raise FileNotFoundError(
            f"No reconstruction/dynamic csv found in oracle={oracle_root} or slam={slam_root}"
        )

    recon_agg_rows: List[Dict[str, object]] = []
    dyn_agg_rows: List[Dict[str, object]] = []
    sig_rows: List[Dict[str, object]] = []

    if recon_rows:
        recon_headers = sorted({k for r in recon_rows for k in r.keys()})
        # Keep stable leading columns.
        lead = ["protocol", "scene_type", "sequence", "method", "seed", "source_tag"]
        recon_headers = lead + [h for h in recon_headers if h not in lead]
        _write_csv(summary_root / "dual_protocol_multiseed_reconstruction.csv", recon_headers, recon_rows)
        agg_headers, recon_agg_rows = _aggregate(recon_rows)
        _write_csv(summary_root / "dual_protocol_multiseed_reconstruction_agg.csv", agg_headers, recon_agg_rows)
        sig_rows = _build_significance(recon_rows)
        if sig_rows:
            sig_headers = [
                "protocol",
                "scene_type",
                "metric",
                "direction",
                "method_a",
                "method_b",
                "n_pairs",
                "mean_delta_egf_minus_baseline",
                "mean_improvement",
                "t_stat",
                "t_pvalue",
                "wilcoxon_stat",
                "wilcoxon_pvalue",
            ]
            _write_csv(summary_root / "dual_protocol_multiseed_significance.csv", sig_headers, sig_rows)
            _write_csv(summary_root / "local_mapping_significance_dual_protocol.csv", sig_headers, sig_rows)

    if dyn_rows:
        dyn_headers = sorted({k for r in dyn_rows for k in r.keys()})
        lead = ["protocol", "scene_type", "sequence", "method", "seed", "source_tag"]
        dyn_headers = lead + [h for h in dyn_headers if h not in lead]
        _write_csv(summary_root / "dual_protocol_multiseed_dynamic.csv", dyn_headers, dyn_rows)
        agg_headers, dyn_agg_rows = _aggregate(dyn_rows)
        _write_csv(summary_root / "dual_protocol_multiseed_dynamic_agg.csv", agg_headers, dyn_agg_rows)

    if recon_agg_rows or dyn_agg_rows:
        main_rows, mean_std_rows = _build_local_mapping_tables(recon_agg_rows, dyn_agg_rows)
        if main_rows:
            main_headers = [
                "protocol",
                "scene_type",
                "sequence",
                "method",
                "n",
                "fscore",
                "recall_5cm",
                "precision_5cm",
                "chamfer",
                "ghost_ratio",
                "ghost_tail_ratio",
                "background_recovery",
            ]
            _write_csv(summary_root / "local_mapping_main_table_dual_protocol.csv", main_headers, main_rows)
        if mean_std_rows:
            mean_std_headers = [
                "protocol",
                "scene_type",
                "sequence",
                "method",
                "n_recon",
                "n_dynamic",
                "fscore_mean",
                "fscore_std",
                "recall_5cm_mean",
                "recall_5cm_std",
                "precision_5cm_mean",
                "precision_5cm_std",
                "chamfer_mean",
                "chamfer_std",
                "ghost_ratio_mean",
                "ghost_ratio_std",
                "ghost_tail_ratio_mean",
                "ghost_tail_ratio_std",
                "background_recovery_mean",
                "background_recovery_std",
            ]
            _write_csv(summary_root / "local_mapping_mean_std_dual_protocol.csv", mean_std_headers, mean_std_rows)

    print("[done] dual protocol multi-seed summary generated under", summary_root)


if __name__ == "__main__":
    main()
