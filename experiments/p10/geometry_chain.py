from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from experiments.p10.rps_plane_attribution import extract_dominant_planes, point_to_plane_distance, snap_points_to_plane
from run_benchmark import load_points_with_normals


def _resolve_donor_dir(donor_root: str | Path, sequence: str) -> Path | None:
    root = Path(donor_root)
    candidates = [
        root / "bonn_slam" / "slam" / sequence / "egf",
        root / "tum_oracle" / "oracle" / sequence / "egf",
        root / sequence / "egf",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_donor_rows(path: Path) -> List[dict]:
    import csv

    feat = path / "rear_surface_features.csv"
    if not feat.exists():
        return []
    rows = list(csv.DictReader(feat.open("r", encoding="utf-8")))
    out: List[dict] = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            if value in (None, ""):
                parsed[key] = 0.0
            else:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
        out.append(parsed)
    return out


def apply_geometry_chain_coupling(
    *,
    sequence: str,
    front_points: np.ndarray,
    front_normals: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    donor_root: str | Path,
    mode: str = "direct",
    project_dist: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], Dict[str, float]]:
    donor_dir = _resolve_donor_dir(donor_root, sequence)
    if donor_dir is None:
        return surface_points, surface_normals, np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float), [], {"geometry_chain_enabled": 1.0, "geometry_chain_applied": 0.0, "geometry_chain_donor_found": 0.0}

    donor_rear, donor_normals = load_points_with_normals(donor_dir / "rear_surface_points.ply")
    donor_rows = _load_donor_rows(donor_dir)
    if donor_rear.shape[0] == 0:
        return surface_points, surface_normals, donor_rear, donor_normals, donor_rows, {"geometry_chain_enabled": 1.0, "geometry_chain_applied": 0.0, "geometry_chain_donor_found": 1.0}

    rear = np.asarray(donor_rear, dtype=float).copy()
    normals = np.asarray(donor_normals, dtype=float).copy()
    rows_out = [dict(r) for r in donor_rows] if donor_rows else [dict() for _ in range(rear.shape[0])]

    projected = 0
    if mode == "projected":
        planes = extract_dominant_planes(surface_points, distance_threshold=0.03, min_plane_points=40, max_planes=6, min_extent_xy=0.25)
        if rear.shape[0] > 0 and planes:
            d = np.stack([point_to_plane_distance(rear, plane) for plane in planes], axis=1)
            assign = np.argmin(d, axis=1)
            mind = d[np.arange(rear.shape[0]), assign]
            mask = mind <= float(project_dist)
            for i in np.where(mask)[0]:
                plane = planes[int(assign[i])]
                rear[i : i + 1] = snap_points_to_plane(rear[i : i + 1], plane)
                if normals.shape == donor_rear.shape:
                    nn = np.asarray(plane.normal, dtype=float).copy()
                    if float(np.dot(nn, normals[i])) < 0.0:
                        nn = -nn
                    normals[i] = nn
                if i < len(rows_out):
                    rows_out[i]["geometry_chain_projected"] = 1.0
                projected += 1
    elif mode != "direct":
        raise ValueError(f"Unknown geometry_chain mode: {mode}")

    if front_points.shape[0] > 0:
        pred_points = np.vstack([front_points, rear])
        pred_normals = np.vstack([front_normals, normals]) if front_normals.shape == front_points.shape else normals
    else:
        pred_points = rear
        pred_normals = normals
    stats = {
        "geometry_chain_enabled": 1.0,
        "geometry_chain_applied": 1.0,
        "geometry_chain_donor_found": 1.0,
        "geometry_chain_mode_projected": 1.0 if mode == "projected" else 0.0,
        "geometry_chain_rear_points": float(rear.shape[0]),
        "geometry_chain_projected_points": float(projected),
    }
    return pred_points, pred_normals, rear, normals, rows_out, stats
