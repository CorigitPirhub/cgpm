#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


TIME_PAT = re.compile(r"ELAPSED_SEC=([0-9.]+)")
RSS_PAT = re.compile(r"MAX_RSS_KB=([0-9]+)")


def _to_float(v: object, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _run_timed(cmd: Sequence[str], stdout_log: Path, time_log: Path, cwd: Path) -> Dict[str, float]:
    wrapped = ["/usr/bin/time", "-f", "ELAPSED_SEC=%e\nMAX_RSS_KB=%M", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(
        wrapped,
        cwd=str(cwd),
        stdout=stdout_log.open("w", encoding="utf-8"),
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    t1 = time.perf_counter()
    stderr = proc.stderr or ""
    time_log.parent.mkdir(parents=True, exist_ok=True)
    time_log.write_text(stderr, encoding="utf-8")
    m_t = TIME_PAT.search(stderr)
    m_r = RSS_PAT.search(stderr)
    return {
        "returncode": float(proc.returncode),
        "elapsed_sec": float(m_t.group(1)) if m_t else float(t1 - t0),
        "max_rss_kb": float(m_r.group(1)) if m_r else float("nan"),
    }


def _read_single_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty table: {path}")
    return dict(rows[0])


def _write_csv(path: Path, headers: Sequence[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def _write_md(path: Path, rows: List[Dict[str, object]], selected_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "config_id",
        "max_points_per_frame",
        "elapsed_sec",
        "fps",
        "fscore",
        "ghost_ratio",
        "delta_fscore_vs_anchor",
        "delta_ghost_ratio_vs_anchor",
        "pass_speed",
        "pass_quality",
        "pass_ghost",
        "selected",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# P7 Efficiency v2 Sweep\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            rr = dict(r)
            rr["selected"] = "yes" if str(r.get("config_id")) == selected_id else ""
            vals = []
            for h in headers:
                v = rr.get(h, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def _plot(rows: List[Dict[str, object]], out_png: Path, selected_id: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    fps = [_to_float(r.get("fps")) for r in rows]
    fscore = [_to_float(r.get("fscore")) for r in rows]
    ghost = [_to_float(r.get("ghost_ratio")) for r in rows]
    sizes = [40.0 + 0.15 * _to_float(r.get("max_points_per_frame"), 0.0) for r in rows]
    sc = ax.scatter(fps, fscore, c=ghost, s=sizes, cmap="viridis_r", alpha=0.9, edgecolors="black", linewidths=0.4)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("ghost_ratio (lower better)")
    ax.axvline(1.0, linestyle="--", color="#444444", linewidth=1.2, label="1.0 FPS target")
    for r in rows:
        x = _to_float(r.get("fps"))
        y = _to_float(r.get("fscore"))
        cid = str(r.get("config_id"))
        label = cid.replace("mpp", "")
        if cid == selected_id:
            ax.scatter([x], [y], s=220, facecolors="none", edgecolors="#d62728", linewidths=2.0, zorder=5)
            label = f"{label}*"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("FPS (40 frames / elapsed_sec)")
    ax.set_ylabel("F-score")
    ax.set_title("P7 Quality-Speed Tradeoff (walking_xyz, oracle)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run P7 efficiency sweep and produce quality-speed tradeoff.")
    ap.add_argument("--dataset_root", type=str, default="data/tum")
    ap.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg3_walking_xyz")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--voxel_size", type=float, default=0.02)
    ap.add_argument("--eval_thresh", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_points_list", type=str, default="900,500,400,380,300")
    ap.add_argument("--out_root", type=str, default="output/tmp/p7_efficiency_v2")
    ap.add_argument("--out_csv", type=str, default="output/summary_tables/local_mapping_efficiency_v2.csv")
    ap.add_argument("--out_md", type=str, default="output/summary_tables/local_mapping_efficiency_v2.md")
    ap.add_argument("--out_json", type=str, default="output/summary_tables/local_mapping_efficiency_v2.json")
    ap.add_argument("--plot_png", type=str, default="assets/quality_speed_tradeoff.png")
    ap.add_argument("--min_fps", type=float, default=1.0)
    ap.add_argument("--max_fscore_drop", type=float, default=0.02)
    ap.add_argument("--max_ghost_rise", type=float, default=0.03)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    project_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mpps = [int(x.strip()) for x in str(args.max_points_list).split(",") if x.strip()]
    if not mpps:
        raise ValueError("max_points_list is empty")

    rows: List[Dict[str, object]] = []
    for mpp in mpps:
        cid = f"mpp{mpp}"
        run_root = out_root / cid
        bench_out = run_root / "bench"
        time_log = run_root / "time.log"
        stdout_log = run_root / "stdout.log"
        recon_csv = bench_out / "oracle" / "tables" / "reconstruction_metrics.csv"
        dyn_csv = bench_out / "oracle" / "tables" / "dynamic_metrics.csv"

        # Backward-compatible reuse: existing probe may write directly under run_root/oracle/tables
        if not recon_csv.exists():
            recon_alt = run_root / "oracle" / "tables" / "reconstruction_metrics.csv"
            dyn_alt = run_root / "oracle" / "tables" / "dynamic_metrics.csv"
            if recon_alt.exists() and dyn_alt.exists():
                recon_csv = recon_alt
                dyn_csv = dyn_alt
                bench_out = run_root

        need_run = bool(args.force) or (not recon_csv.exists()) or (not dyn_csv.exists()) or (not time_log.exists())
        if need_run:
            bench_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                py,
                "scripts/run_benchmark.py",
                "--dataset_kind",
                "tum",
                "--dataset_root",
                str(args.dataset_root),
                "--protocol",
                "oracle",
                "--static_sequences",
                "",
                "--dynamic_sequences",
                str(args.sequence),
                "--methods",
                "egf",
                "--frames",
                str(int(args.frames)),
                "--stride",
                str(int(args.stride)),
                "--max_points_per_frame",
                str(int(mpp)),
                "--voxel_size",
                str(float(args.voxel_size)),
                "--eval_thresh",
                str(float(args.eval_thresh)),
                "--seed",
                str(int(args.seed)),
                "--out_root",
                str(bench_out),
                "--force",
            ]
            timed = _run_timed(cmd, stdout_log=stdout_log, time_log=time_log, cwd=project_root)
        else:
            txt = time_log.read_text(encoding="utf-8")
            m_t = TIME_PAT.search(txt)
            m_r = RSS_PAT.search(txt)
            timed = {
                "returncode": 0.0,
                "elapsed_sec": float(m_t.group(1)) if m_t else float("nan"),
                "max_rss_kb": float(m_r.group(1)) if m_r else float("nan"),
            }

        recon = _read_single_row(recon_csv)
        dyn = _read_single_row(dyn_csv)
        elapsed = _to_float(timed.get("elapsed_sec"))
        fps = (float(args.frames) / elapsed) if elapsed and elapsed > 0 else float("nan")
        rows.append(
            {
                "config_id": cid,
                "max_points_per_frame": int(mpp),
                "elapsed_sec": elapsed,
                "fps": fps,
                "max_rss_kb": _to_float(timed.get("max_rss_kb")),
                "fscore": _to_float(recon.get("fscore")),
                "chamfer": _to_float(recon.get("chamfer")),
                "ghost_ratio": _to_float(dyn.get("ghost_ratio")),
                "ghost_tail_ratio": _to_float(dyn.get("ghost_tail_ratio")),
                "out_dir": str(bench_out),
            }
        )

    # Anchor: highest quality (max F-score) across probed configs.
    anchor = max(rows, key=lambda r: _to_float(r.get("fscore"), float("-inf")))
    anchor_fscore = _to_float(anchor.get("fscore"))
    anchor_ghost = _to_float(anchor.get("ghost_ratio"))

    selected_id = ""
    selected_score = float("-inf")
    for r in rows:
        delta_f = _to_float(r.get("fscore")) - anchor_fscore
        delta_g = _to_float(r.get("ghost_ratio")) - anchor_ghost
        pass_speed = _to_float(r.get("fps")) >= float(args.min_fps)
        pass_quality = delta_f >= -float(args.max_fscore_drop)
        pass_ghost = delta_g <= float(args.max_ghost_rise)
        r["anchor_config_id"] = str(anchor.get("config_id"))
        r["anchor_fscore"] = anchor_fscore
        r["anchor_ghost_ratio"] = anchor_ghost
        r["delta_fscore_vs_anchor"] = delta_f
        r["delta_ghost_ratio_vs_anchor"] = delta_g
        r["pass_speed"] = bool(pass_speed)
        r["pass_quality"] = bool(pass_quality)
        r["pass_ghost"] = bool(pass_ghost)
        score = _to_float(r.get("fps")) + 0.2 * _to_float(r.get("fscore"))
        if pass_speed and pass_quality and pass_ghost and score > selected_score:
            selected_score = score
            selected_id = str(r.get("config_id"))

    headers = [
        "config_id",
        "max_points_per_frame",
        "elapsed_sec",
        "fps",
        "max_rss_kb",
        "fscore",
        "chamfer",
        "ghost_ratio",
        "ghost_tail_ratio",
        "anchor_config_id",
        "anchor_fscore",
        "anchor_ghost_ratio",
        "delta_fscore_vs_anchor",
        "delta_ghost_ratio_vs_anchor",
        "pass_speed",
        "pass_quality",
        "pass_ghost",
        "out_dir",
    ]
    _write_csv(Path(args.out_csv), headers, rows)
    _write_md(Path(args.out_md), rows, selected_id=selected_id)
    Path(args.out_json).write_text(
        json.dumps(
            {
                "rows": rows,
                "selected_config_id": selected_id,
                "acceptance": {
                    "min_fps": float(args.min_fps),
                    "max_fscore_drop": float(args.max_fscore_drop),
                    "max_ghost_rise": float(args.max_ghost_rise),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot(rows, Path(args.plot_png), selected_id=selected_id)

    print(f"[done] csv: {args.out_csv}")
    print(f"[done] md: {args.out_md}")
    print(f"[done] json: {args.out_json}")
    print(f"[done] plot: {args.plot_png}")
    if selected_id:
        print(f"[done] selected_config_id: {selected_id}")
    else:
        print("[warn] no config satisfies all acceptance gates")


if __name__ == "__main__":
    main()
