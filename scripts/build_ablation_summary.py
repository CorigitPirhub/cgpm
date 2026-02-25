from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _read_single_row_csv(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"missing csv: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty csv: {path}")
    return rows[0]


def _float(d: Dict[str, str], k: str, default: float = 0.0) -> float:
    v = d.get(k, "")
    if v is None or v == "":
        return default
    return float(v)


def load_variant(out_root: Path) -> Dict[str, float | str]:
    rec = _read_single_row_csv(out_root / "tables" / "reconstruction_metrics.csv")
    dyn = _read_single_row_csv(out_root / "tables" / "dynamic_metrics.csv")
    return {
        "sequence": rec.get("sequence", ""),
        "fscore": _float(rec, "fscore"),
        "precision": _float(rec, "precision"),
        "recall": _float(rec, "recall"),
        "accuracy": _float(rec, "accuracy"),
        "completeness": _float(rec, "completeness"),
        "chamfer": _float(rec, "chamfer"),
        "ghost_ratio": _float(dyn, "ghost_ratio"),
        "ghost_count": _float(dyn, "ghost_count"),
        "background_recovery": _float(dyn, "background_recovery"),
        "ghost_tail_ratio": _float(dyn, "ghost_tail_ratio"),
    }


def default_variants(base: Path) -> List[Tuple[str, Path]]:
    return [
        ("EGF-Full-v6", base / "r40_full"),
        ("EGF-No-Evidence", base / "r40_no_evidence"),
        ("EGF-No-Gradient", base / "r40_no_gradient"),
        ("EGF-Classic-SDF", base / "r40_classic_sdf"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="output/ablation_study")
    parser.add_argument("--out_csv", type=str, default="output/ablation_study/summary.csv")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="custom variant as label=path, can be repeated",
    )
    args = parser.parse_args()

    base = Path(args.base)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    variants: List[Tuple[str, Path]]
    if args.variant:
        variants = []
        for v in args.variant:
            if "=" not in v:
                raise ValueError(f"invalid --variant format: {v}")
            label, p = v.split("=", 1)
            variants.append((label.strip(), Path(p.strip())))
    else:
        variants = default_variants(base)

    rows: List[Dict[str, float | str]] = []
    for label, root in variants:
        data = load_variant(root)
        row: Dict[str, float | str] = {"variant": label, "out_root": str(root)}
        row.update(data)
        rows.append(row)

    headers = [
        "variant",
        "sequence",
        "fscore",
        "precision",
        "recall",
        "accuracy",
        "completeness",
        "chamfer",
        "ghost_count",
        "ghost_ratio",
        "ghost_tail_ratio",
        "background_recovery",
        "out_root",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[done] wrote {out_csv}")


if __name__ == "__main__":
    main()
