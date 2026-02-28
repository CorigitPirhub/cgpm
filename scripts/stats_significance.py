from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats


NUMERIC_FIELDS = {
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
}

METRIC_DIRECTIONS: List[Tuple[str, str]] = [
    ("fscore", "higher"),
    ("precision", "higher"),
    ("recall", "higher"),
    ("normal_consistency", "higher"),
    ("fscore_2cm", "higher"),
    ("fscore_5cm", "higher"),
    ("fscore_10cm", "higher"),
    ("accuracy", "lower"),
    ("completeness", "lower"),
    ("chamfer", "lower"),
    ("ghost_ratio", "lower"),
    ("roi_ghost_ratio", "lower"),
    ("ghost_tail_ratio", "lower"),
    ("background_recovery", "higher"),
    ("roi_background_recovery", "higher"),
]


def _to_float(v: str) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        out = float(s)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _load_reconstruction_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: Dict[str, object] = dict(row)
            for k in NUMERIC_FIELDS:
                if k in out:
                    fv = _to_float(str(out[k]))
                    if fv is not None:
                        out[k] = fv
            # Normalize key fields for stable pairing.
            out["sequence"] = str(out.get("sequence", "")).strip()
            out["method"] = str(out.get("method", "")).strip().lower()
            out["scene_type"] = str(out.get("scene_type", "")).strip().lower()
            out["protocol"] = str(out.get("protocol", "slam")).strip().lower() or "slam"
            seed_val = out.get("seed", "0")
            try:
                out["seed"] = int(float(seed_val))
            except Exception:
                out["seed"] = 0
            rows.append(out)
    return rows


def _discover_recon_csvs(root: Path) -> List[Path]:
    direct = root / "tables" / "reconstruction_metrics.csv"
    files = [direct] if direct.exists() else []
    for p in sorted(root.rglob("reconstruction_metrics.csv")):
        if p not in files:
            files.append(p)
    # Keep only files under a tables directory to avoid accidental stale dumps.
    return [p for p in files if p.parent.name == "tables"]


def _dedupe_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    table: Dict[Tuple[str, str, str, str, int], Dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("protocol", "slam")),
            str(row.get("scene_type", "")),
            str(row.get("sequence", "")),
            str(row.get("method", "")),
            int(row.get("seed", 0)),
        )
        table[key] = row
    return list(table.values())


def _paired_vectors(
    rows: List[Dict[str, object]],
    protocol: str,
    scene_type: str,
    baseline: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    egf_map: Dict[Tuple[str, int], float] = {}
    base_map: Dict[Tuple[str, int], float] = {}
    for row in rows:
        if str(row.get("protocol")) != protocol:
            continue
        if str(row.get("scene_type")) != scene_type:
            continue
        seq = str(row.get("sequence"))
        seed = int(row.get("seed", 0))
        method = str(row.get("method", "")).lower()
        val_obj = row.get(metric)
        if not isinstance(val_obj, (float, int)):
            continue
        val = float(val_obj)
        if not math.isfinite(val):
            continue
        if method == "egf":
            egf_map[(seq, seed)] = val
        elif method == baseline:
            base_map[(seq, seed)] = val

    keys = sorted(set(egf_map.keys()).intersection(base_map.keys()))
    if not keys:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    egf = np.asarray([egf_map[k] for k in keys], dtype=float)
    base = np.asarray([base_map[k] for k in keys], dtype=float)
    return egf, base


def _cohens_d(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("nan")
    if samples.size == 1:
        return float("nan")
    std = float(np.std(samples, ddof=1))
    if std <= 1e-12:
        return float("nan")
    return float(np.mean(samples) / std)


def _wilcoxon_safe(deltas: np.ndarray) -> Tuple[float, float]:
    if deltas.size < 2:
        return float("nan"), float("nan")
    if np.allclose(deltas, 0.0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(deltas)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="output/journal_suite")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    root = Path(args.root)
    files = _discover_recon_csvs(root)
    if not files:
        raise FileNotFoundError(f"No reconstruction_metrics.csv found under: {root}")

    all_rows: List[Dict[str, object]] = []
    for p in files:
        all_rows.extend(_load_reconstruction_rows(p))
    rows = _dedupe_rows(all_rows)

    methods = sorted({str(r.get("method", "")).lower() for r in rows if str(r.get("method", "")).strip()})
    baselines = [m for m in methods if m != "egf"]
    protocols = sorted({str(r.get("protocol", "slam")) for r in rows})
    scene_types = sorted({str(r.get("scene_type", "")) for r in rows})

    out_rows: List[Dict[str, object]] = []
    for protocol in protocols:
        for scene_type in scene_types:
            for baseline in baselines:
                for metric, direction in METRIC_DIRECTIONS:
                    egf, base = _paired_vectors(rows, protocol, scene_type, baseline, metric)
                    if egf.size == 0:
                        continue

                    raw_delta = egf - base
                    improve = raw_delta if direction == "higher" else -raw_delta

                    mean_delta = float(np.mean(raw_delta))
                    mean_improve = float(np.mean(improve))
                    std_delta = float(np.std(raw_delta, ddof=1)) if raw_delta.size > 1 else 0.0
                    std_improve = float(np.std(improve, ddof=1)) if improve.size > 1 else 0.0

                    if raw_delta.size > 1:
                        t_stat, t_p = stats.ttest_rel(egf, base)
                        # Align sign: positive t_stat means EGF improvement.
                        if direction == "lower":
                            t_stat = -t_stat
                        t_stat = float(t_stat)
                        t_p = float(t_p)
                    else:
                        t_stat, t_p = float("nan"), float("nan")

                    w_stat, w_p = _wilcoxon_safe(improve)
                    d_eff = _cohens_d(improve)

                    out_rows.append(
                        {
                            "protocol": protocol,
                            "scene_type": scene_type,
                            "metric": metric,
                            "direction": direction,
                            "method_a": "egf",
                            "method_b": baseline,
                            "n_pairs": int(raw_delta.size),
                            "mean_delta_egf_minus_baseline": mean_delta,
                            "std_delta_egf_minus_baseline": std_delta,
                            "mean_improvement": mean_improve,
                            "std_improvement": std_improve,
                            "cohens_d": d_eff,
                            "t_stat": t_stat,
                            "t_pvalue": t_p,
                            "wilcoxon_stat": w_stat,
                            "wilcoxon_pvalue": w_p,
                        }
                    )

    if not out_rows:
        raise RuntimeError("No paired samples found for significance analysis.")

    out_rows.sort(key=lambda r: (str(r["protocol"]), str(r["scene_type"]), str(r["metric"]), str(r["method_b"])))
    out_path = Path(args.out) if args.out else (root / "tables" / "significance.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "protocol",
        "scene_type",
        "metric",
        "direction",
        "method_a",
        "method_b",
        "n_pairs",
        "mean_delta_egf_minus_baseline",
        "std_delta_egf_minus_baseline",
        "mean_improvement",
        "std_improvement",
        "cohens_d",
        "t_stat",
        "t_pvalue",
        "wilcoxon_stat",
        "wilcoxon_pvalue",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"[done] significance table: {out_path}")


if __name__ == "__main__":
    main()
