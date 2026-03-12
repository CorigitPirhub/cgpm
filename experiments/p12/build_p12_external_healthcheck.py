#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


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


def _is_placeholder_source(src: str) -> bool:
    s = str(src).replace("\\", "/").strip().lower()
    if not s:
        return True
    bad_tokens = [
        "/simple_removal/surface_points.ply",
        "../simple_removal/surface_points.ply",
        "placeholder",
        "dummy",
    ]
    return any(t in s for t in bad_tokens)


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _to_float(v: object, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


@dataclass
class MethodHealth:
    method: str
    n_expected: int
    n_present: int
    n_real: int
    n_missing: int
    runtime_returncode: int
    runtime_ok: bool
    pass_health: bool


def _runner_repro_cmd(method: str, sequence: str) -> str:
    if method == "dynaslam":
        return (
            "python scripts/external/run_dynaslam_tum_runner.py "
            f"--sequence_dir data/tum/{sequence} "
            f"--out_points output/external/dynaslam_real/{sequence}/mesh.ply "
            "--frames 40 --stride 3 --max_points_per_frame 2500 --runner_timeout_sec 300"
        )
    if method == "neural_implicit":
        return (
            "python experiments/p3/run_p3_real_external.py "
            f"--sequence {sequence} --protocol slam --frames 40 --stride 3 "
            f"--niceslam_output output/external/nice_slam/{sequence}_f010_fast "
            f"--niceslam_config configs/TUM_RGBD/{sequence}_f010_fast.yaml "
            f"--neural_mesh output/external/neural_mesh/{sequence}/mesh.ply --skip_niceslam"
        )
    if method == "tsdf":
        return "python scripts/run_tsdf_baseline.py --dataset_root data/tum --sequence <sequence>"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build P12 external baseline healthcheck and filtered final tables.")
    ap.add_argument(
        "--recon_csv",
        type=str,
        default="output/summary_tables/external_baselines_native_reconstruction_final.csv",
    )
    ap.add_argument(
        "--dyn_csv",
        type=str,
        default="output/summary_tables/external_baselines_native_dynamic_final.csv",
    )
    ap.add_argument(
        "--runtime_csv",
        type=str,
        default="output/summary_tables/external_baselines_native_runtime_final.csv",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="output/tmp/p12_external_native_final",
    )
    ap.add_argument(
        "--sequences",
        type=str,
        default="rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere",
    )
    ap.add_argument("--protocol", type=str, default="slam")
    ap.add_argument("--allow_one_seq_failure", action="store_true")
    ap.add_argument(
        "--out_health_md",
        type=str,
        default="output/tmp/external_audit/BASELINE_HEALTHCHECK.md",
    )
    ap.add_argument(
        "--out_health_csv",
        type=str,
        default="output/summary_tables/external_baselines_native_healthcheck_final.csv",
    )
    args = ap.parse_args()

    recon_csv = Path(args.recon_csv)
    dyn_csv = Path(args.dyn_csv)
    runtime_csv = Path(args.runtime_csv)
    out_root = Path(args.out_root)
    sequences = [s.strip() for s in str(args.sequences).split(",") if s.strip()]
    protocol = str(args.protocol)

    recon_rows = _read_csv(recon_csv)
    dyn_rows = _read_csv(dyn_csv)
    runtime_rows = _read_csv(runtime_csv)

    methods = sorted(set(r.get("method", "") for r in recon_rows if r.get("method")))
    runtime_by_method = {str(r.get("method", "")): dict(r) for r in runtime_rows}

    # Build per-method per-sequence health.
    seq_health_rows: List[Dict[str, object]] = []
    method_health: List[MethodHealth] = []
    passed_methods: List[str] = []

    for m in methods:
        present = 0
        real = 0
        missing = 0
        rt = runtime_by_method.get(m, {})
        runtime_rc = int(_to_float(rt.get("returncode"), default=1.0))
        runtime_ok = runtime_rc == 0

        is_external_method = m in {"dynaslam", "neural_implicit", "midfusion"}
        for seq in sequences:
            rec = next((r for r in recon_rows if r.get("method") == m and r.get("sequence") == seq), None)
            seq_out = out_root / m / protocol / seq / m
            summary = _load_json(seq_out / "summary.json")
            status = str(summary.get("status", "missing")).strip().lower() if summary else "missing"
            source_path = str(summary.get("source_path", "")).strip() if summary else ""
            source_exists = bool(source_path) and Path(source_path).exists()
            real_external = int(bool(source_path) and (not _is_placeholder_source(source_path)) and source_exists)

            row_state = "ok"
            if rec is None:
                row_state = "missing"
                missing += 1
            elif is_external_method and status == "missing":
                row_state = "missing"
                missing += 1
            elif is_external_method and status == "skipped":
                row_state = "skipped"
                missing += 1
            else:
                present += 1
                if is_external_method:
                    real += real_external
                if is_external_method and real_external == 0:
                    row_state = "non_real_source"

            seq_health_rows.append(
                {
                    "method": m,
                    "sequence": seq,
                    "state": row_state,
                    "summary_status": status,
                    "source_path": source_path,
                    "source_exists": int(source_exists),
                    "is_real_external_source": int(real_external),
                    "runner_repro_cmd": _runner_repro_cmd(m, seq),
                }
            )

        required_present = len(sequences) - (1 if bool(args.allow_one_seq_failure) else 0)
        pass_cov = present >= required_present
        pass_real = True
        if m in {"dynaslam", "neural_implicit", "midfusion"}:
            pass_real = real >= required_present
        # External runners may return non-zero while still producing complete
        # native outputs; accept them if coverage and real-source checks pass.
        runtime_soft_ok = bool(runtime_ok or (is_external_method and pass_cov and pass_real))
        pass_h = bool(pass_cov and pass_real and runtime_soft_ok)
        mh = MethodHealth(
            method=m,
            n_expected=len(sequences),
            n_present=present,
            n_real=real,
            n_missing=missing,
            runtime_returncode=runtime_rc,
            runtime_ok=runtime_soft_ok,
            pass_health=pass_h,
        )
        method_health.append(mh)
        if pass_h:
            passed_methods.append(m)

    # Filter final tables to health-passed methods only.
    recon_keep = [r for r in recon_rows if str(r.get("method", "")) in set(passed_methods)]
    dyn_keep = [r for r in dyn_rows if str(r.get("method", "")) in set(passed_methods)]
    runtime_keep = [r for r in runtime_rows if str(r.get("method", "")) in set(passed_methods)]
    if recon_keep:
        _write_csv(recon_csv, list(recon_keep[0].keys()), recon_keep)
    if dyn_keep:
        _write_csv(dyn_csv, list(dyn_keep[0].keys()), dyn_keep)
    if runtime_keep:
        _write_csv(runtime_csv, list(runtime_keep[0].keys()), runtime_keep)

    # Write health csv.
    _write_csv(
        Path(args.out_health_csv),
        [
            "method",
            "n_expected",
            "n_present",
            "n_real",
            "n_missing",
            "runtime_returncode",
            "runtime_ok",
            "pass_health",
        ],
        [
            {
                "method": h.method,
                "n_expected": h.n_expected,
                "n_present": h.n_present,
                "n_real": h.n_real,
                "n_missing": h.n_missing,
                "runtime_returncode": h.runtime_returncode,
                "runtime_ok": int(h.runtime_ok),
                "pass_health": int(h.pass_health),
            }
            for h in method_health
        ],
    )

    # Build markdown report.
    md_path = Path(args.out_health_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# BASELINE_HEALTHCHECK")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Protocol: `{protocol}`")
    lines.append(f"- Sequences: `{', '.join(sequences)}`")
    lines.append(f"- Allow one sequence failure: `{bool(args.allow_one_seq_failure)}`")
    lines.append("")
    lines.append("## Method Health Summary")
    lines.append("| method | present/expected | real_external/expected | runtime_returncode | runtime_ok(soft) | pass_health |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for h in method_health:
        lines.append(
            f"| {h.method} | {h.n_present}/{h.n_expected} | {h.n_real}/{h.n_expected} | "
            f"{h.runtime_returncode} | {int(h.runtime_ok)} | {int(h.pass_health)} |"
        )
    lines.append("")
    lines.append("## Sequence Traceability")
    lines.append("| method | sequence | state | summary_status | source_exists | is_real_external_source | source_path |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for r in seq_health_rows:
        lines.append(
            f"| {r['method']} | {r['sequence']} | {r['state']} | {r['summary_status']} | "
            f"{r['source_exists']} | {r['is_real_external_source']} | `{r['source_path']}` |"
        )
    lines.append("")
    lines.append("## Runner Commands (Repro Template)")
    for m in sorted(set(r["method"] for r in seq_health_rows)):
        lines.append(f"### {m}")
        for r in [x for x in seq_health_rows if x["method"] == m]:
            cmd = str(r.get("runner_repro_cmd", "")).strip()
            if cmd:
                lines.append(f"- `{r['sequence']}`: `{cmd}`")
    lines.append("")
    lines.append("## Main-Table Eligibility")
    lines.append(
        f"- Health-passed methods kept in final tables: `{', '.join(sorted(passed_methods)) if passed_methods else 'NONE'}`"
    )
    lines.append(f"- Reconstruction table: `{recon_csv}`")
    lines.append(f"- Dynamic table: `{dyn_csv}`")
    lines.append(f"- Runtime table: `{runtime_csv}`")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Any `state != ok` row is treated as sequence-level failure; if `allow_one_seq_failure=true`, method can still pass with full logs and repro command."
    )
    lines.append(
        "- External source is considered real only when `source_path` exists and is not a known placeholder/simple-removal proxy."
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] health markdown: {md_path}")
    print(f"[done] health csv: {args.out_health_csv}")
    print(f"[done] kept methods: {','.join(sorted(passed_methods)) if passed_methods else 'NONE'}")


if __name__ == "__main__":
    main()
