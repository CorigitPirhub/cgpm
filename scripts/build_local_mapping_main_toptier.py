#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def _write_markdown(path: Path, title: str, headers: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            vals: List[str] = []
            for h in headers:
                v = r.get(h, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def _paired_vectors(rows: List[Dict[str, object]], dataset: str, protocol: str, metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    egf: Dict[Tuple[str, int], float] = {}
    tsdf: Dict[Tuple[str, int], float] = {}
    for r in rows:
        if str(r.get("dataset")) != dataset:
            continue
        if str(r.get("protocol")) != protocol:
            continue
        if str(r.get("scene_type")) != "dynamic":
            continue
        seq = str(r.get("sequence"))
        seed = int(r.get("seed", 0))
        method = str(r.get("method", "")).lower()
        val = _to_float(r.get(metric_key))
        if val is None:
            continue
        if method == "egf":
            egf[(seq, seed)] = val
        elif method == "tsdf":
            tsdf[(seq, seed)] = val
    keys = sorted(set(egf.keys()).intersection(tsdf.keys()))
    if not keys:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    return (
        np.asarray([egf[k] for k in keys], dtype=float),
        np.asarray([tsdf[k] for k in keys], dtype=float),
    )


def _safe_ttest(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        _t, p = stats.ttest_rel(a, b)
        return float(p)
    except Exception:
        return float("nan")


def _safe_wilcoxon(delta: np.ndarray) -> float:
    if delta.size < 2:
        return float("nan")
    if np.allclose(delta, 0.0):
        return 1.0
    try:
        _s, p = stats.wilcoxon(delta)
        return float(p)
    except Exception:
        return float("nan")


def _load_dataset_rows(path: Path, dataset: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in _read_csv(path):
        method = str(r.get("method", "")).lower()
        if method not in {"egf", "tsdf"}:
            continue
        out.append(
            {
                "dataset": dataset,
                "sequence": str(r.get("sequence", "")),
                "scene_type": str(r.get("scene_type", "")).lower(),
                "protocol": str(r.get("protocol", "")).lower(),
                "seed": int(float(r.get("seed", 0) or 0)),
                "method": method,
                # Top-tier style metrics:
                "acc_cm": (_to_float(r.get("accuracy")) or float("nan")) * 100.0,
                "comp_cm": (_to_float(r.get("completeness")) or float("nan")) * 100.0,
                # Comp-R uses recall@5cm
                "comp_r_5cm": (_to_float(r.get("recall_5cm")) or float("nan")) * 100.0,
            }
        )
    return out


def _aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for r in rows:
        if str(r.get("scene_type")) != "dynamic":
            continue
        k = (str(r.get("dataset")), str(r.get("protocol")), str(r.get("sequence")), str(r.get("method")))
        groups.setdefault(k, []).append(r)

    out: List[Dict[str, object]] = []
    for (dataset, protocol, sequence, method), vals in sorted(groups.items()):
        acc = np.asarray([float(v["acc_cm"]) for v in vals], dtype=float)
        comp = np.asarray([float(v["comp_cm"]) for v in vals], dtype=float)
        compr = np.asarray([float(v["comp_r_5cm"]) for v in vals], dtype=float)
        out.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "sequence": sequence,
                "method": method,
                "n_seeds": int(len(vals)),
                "acc_cm": float(np.mean(acc)),
                "acc_cm_std": float(np.std(acc, ddof=1)) if acc.size > 1 else 0.0,
                "comp_cm": float(np.mean(comp)),
                "comp_cm_std": float(np.std(comp, ddof=1)) if comp.size > 1 else 0.0,
                "comp_r_5cm": float(np.mean(compr)),
                "comp_r_5cm_std": float(np.std(compr, ddof=1)) if compr.size > 1 else 0.0,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build top-tier local mapping main table (Acc/Comp/Comp-R).")
    ap.add_argument("--tum_multiseed", type=str, default="output/summary_tables/tum_reconstruction_metrics_multiseed.csv")
    ap.add_argument("--bonn_multiseed", type=str, default="output/summary_tables/bonn_reconstruction_metrics_multiseed.csv")
    ap.add_argument("--out_csv", type=str, default="output/summary_tables/local_mapping_main_metrics_toptier.csv")
    ap.add_argument("--out_md", type=str, default="output/summary_tables/local_mapping_main_metrics_toptier.md")
    args = ap.parse_args()

    tum_path = Path(args.tum_multiseed)
    bonn_path = Path(args.bonn_multiseed)
    if not tum_path.exists():
        raise FileNotFoundError(f"missing: {tum_path}")
    if not bonn_path.exists():
        raise FileNotFoundError(f"missing: {bonn_path}")

    rows = _load_dataset_rows(tum_path, "tum") + _load_dataset_rows(bonn_path, "bonn")
    agg = _aggregate(rows)

    # Dataset-level paired significance (EGF vs TSDF, dynamic only).
    sig: Dict[Tuple[str, str], Dict[str, float]] = {}
    for dataset, protocol in sorted({(str(r["dataset"]), str(r["protocol"])) for r in rows if str(r["scene_type"]) == "dynamic"}):
        egf_acc, tsdf_acc = _paired_vectors(rows, dataset, protocol, "acc_cm")
        egf_comp, tsdf_comp = _paired_vectors(rows, dataset, protocol, "comp_cm")
        egf_compr, tsdf_compr = _paired_vectors(rows, dataset, protocol, "comp_r_5cm")
        sig[(dataset, protocol)] = {
            # lower is better: test on (tsdf - egf) for one-sided interpretation not enforced here; store p(two-sided)
            "p_acc_cm_egf_vs_tsdf_t": _safe_ttest(egf_acc, tsdf_acc),
            "p_acc_cm_egf_vs_tsdf_w": _safe_wilcoxon(tsdf_acc - egf_acc),
            "p_comp_cm_egf_vs_tsdf_t": _safe_ttest(egf_comp, tsdf_comp),
            "p_comp_cm_egf_vs_tsdf_w": _safe_wilcoxon(tsdf_comp - egf_comp),
            # higher is better:
            "p_comp_r_5cm_egf_vs_tsdf_t": _safe_ttest(egf_compr, tsdf_compr),
            "p_comp_r_5cm_egf_vs_tsdf_w": _safe_wilcoxon(egf_compr - tsdf_compr),
            "n_pairs_sig": int(min(egf_compr.size, tsdf_compr.size)),
        }

    out_rows: List[Dict[str, object]] = []
    for r in agg:
        s = sig.get((str(r["dataset"]), str(r["protocol"])), {})
        rr = dict(r)
        rr.update(s)
        rr["source_metrics"] = "tum_reconstruction_metrics_multiseed.csv|bonn_reconstruction_metrics_multiseed.csv"
        out_rows.append(rr)

    headers = [
        "dataset",
        "protocol",
        "sequence",
        "method",
        "n_seeds",
        "acc_cm",
        "acc_cm_std",
        "comp_cm",
        "comp_cm_std",
        "comp_r_5cm",
        "comp_r_5cm_std",
        "p_acc_cm_egf_vs_tsdf_t",
        "p_acc_cm_egf_vs_tsdf_w",
        "p_comp_cm_egf_vs_tsdf_t",
        "p_comp_cm_egf_vs_tsdf_w",
        "p_comp_r_5cm_egf_vs_tsdf_t",
        "p_comp_r_5cm_egf_vs_tsdf_w",
        "n_pairs_sig",
        "source_metrics",
    ]
    _write_csv(Path(args.out_csv), headers, out_rows)
    _write_markdown(Path(args.out_md), "Top-tier Main Metrics (Local Mapping Dynamic)", headers, out_rows)
    print(f"[done] wrote: {args.out_csv}")
    print(f"[done] wrote: {args.out_md}")


if __name__ == "__main__":
    main()

