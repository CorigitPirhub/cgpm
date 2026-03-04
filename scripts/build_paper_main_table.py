#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


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


def _build_sig_lookup(sig_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, object]]:
    # key=(dataset, protocol), with dataset fallback "*" when not provided in source table.
    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    for r in sig_rows:
        dataset = str(r.get("dataset", "")).strip().lower()
        protocol = str(r.get("protocol", "")).strip().lower()
        scene_type = str(r.get("scene_type", "")).strip().lower()
        metric = str(r.get("metric", "")).strip().lower()
        method_a = str(r.get("method_a", "")).strip().lower()
        method_b = str(r.get("method_b", "")).strip().lower()
        if scene_type != "dynamic" or method_a != "egf" or method_b != "tsdf":
            continue
        if metric not in {"fscore", "chamfer", "ghost_ratio"}:
            continue
        key = (dataset if dataset else "*", protocol)
        slot = out.setdefault(
            key,
            {
                "p_fscore_egf_vs_tsdf": "",
                "p_chamfer_egf_vs_tsdf": "",
                "p_ghost_ratio_egf_vs_tsdf": "",
                "n_pairs_sig": "",
            },
        )
        p_t = _to_float(r.get("t_pvalue"))
        p_w = _to_float(r.get("wilcoxon_pvalue"))
        p_best = p_t if p_t is not None else p_w
        if metric == "fscore":
            slot["p_fscore_egf_vs_tsdf"] = p_best if p_best is not None else ""
        elif metric == "chamfer":
            slot["p_chamfer_egf_vs_tsdf"] = p_best if p_best is not None else ""
        elif metric == "ghost_ratio":
            slot["p_ghost_ratio_egf_vs_tsdf"] = p_best if p_best is not None else ""
        if slot.get("n_pairs_sig", "") == "":
            n_pairs = r.get("n_pairs", "")
            slot["n_pairs_sig"] = n_pairs
    return out


def _collect_rows(
    agg_rows: List[Dict[str, str]],
    dataset: str,
    sig_lookup: Dict[Tuple[str, str], Dict[str, object]],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in agg_rows:
        scene_type = str(r.get("scene_type", "")).strip().lower()
        if scene_type != "dynamic":
            continue
        method = str(r.get("method", "")).strip().lower()
        if method not in {"egf", "tsdf"}:
            continue
        protocol = str(r.get("protocol", "")).strip().lower()
        seq = str(r.get("sequence", "")).strip()
        sig = sig_lookup.get((dataset, protocol), sig_lookup.get(("*", protocol), {}))
        out.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "sequence": seq,
                "method": method,
                "n_seeds": r.get("n_seeds", ""),
                "fscore": _to_float(r.get("fscore_mean")),
                "fscore_std": _to_float(r.get("fscore_std")),
                "chamfer": _to_float(r.get("chamfer_mean")),
                "chamfer_std": _to_float(r.get("chamfer_std")),
                "ghost_ratio": _to_float(r.get("ghost_ratio_mean")),
                "ghost_ratio_std": _to_float(r.get("ghost_ratio_std")),
                "p_fscore_egf_vs_tsdf": sig.get("p_fscore_egf_vs_tsdf", ""),
                "p_chamfer_egf_vs_tsdf": sig.get("p_chamfer_egf_vs_tsdf", ""),
                "p_ghost_ratio_egf_vs_tsdf": sig.get("p_ghost_ratio_egf_vs_tsdf", ""),
                "n_pairs_sig": sig.get("n_pairs_sig", ""),
                "significance_source": "dual_protocol_multiseed_significance.csv",
            }
        )
    out.sort(key=lambda x: (str(x.get("dataset", "")), str(x.get("protocol", "")), str(x.get("sequence", "")), str(x.get("method", ""))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paper-facing main local-mapping table (dynamic rows + p-values) from summary tables."
    )
    parser.add_argument(
        "--tum_agg",
        type=str,
        default="output/summary_tables/tum_reconstruction_metrics_multiseed_agg.csv",
    )
    parser.add_argument(
        "--bonn_agg",
        type=str,
        default="output/summary_tables/bonn_reconstruction_metrics_multiseed_agg.csv",
    )
    parser.add_argument(
        "--dual_sig",
        type=str,
        default="output/summary_tables/dual_protocol_multiseed_significance.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="output/summary_tables/paper_main_table_local_mapping.csv",
    )
    parser.add_argument(
        "--out_md",
        type=str,
        default="output/summary_tables/paper_main_table_local_mapping.md",
    )
    args = parser.parse_args()

    tum_agg = Path(args.tum_agg)
    bonn_agg = Path(args.bonn_agg)
    dual_sig = Path(args.dual_sig)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    if not tum_agg.exists():
        raise FileNotFoundError(f"missing tum agg table: {tum_agg}")
    if not bonn_agg.exists():
        raise FileNotFoundError(f"missing bonn agg table: {bonn_agg}")
    if not dual_sig.exists():
        raise FileNotFoundError(f"missing dual significance table: {dual_sig}")

    sig_lookup = _build_sig_lookup(_read_csv(dual_sig))
    rows = _collect_rows(_read_csv(tum_agg), dataset="tum", sig_lookup=sig_lookup)
    rows += _collect_rows(_read_csv(bonn_agg), dataset="bonn", sig_lookup=sig_lookup)

    headers = [
        "dataset",
        "protocol",
        "sequence",
        "method",
        "n_seeds",
        "fscore",
        "fscore_std",
        "chamfer",
        "chamfer_std",
        "ghost_ratio",
        "ghost_ratio_std",
        "p_fscore_egf_vs_tsdf",
        "p_chamfer_egf_vs_tsdf",
        "p_ghost_ratio_egf_vs_tsdf",
        "n_pairs_sig",
        "significance_source",
    ]
    _write_csv(out_csv, headers, rows)
    _write_markdown(out_md, "Paper Main Table (Local Mapping, Dynamic)", headers, rows)
    print(f"[done] wrote: {out_csv}")
    print(f"[done] wrote: {out_md}")


if __name__ == "__main__":
    main()
