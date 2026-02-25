from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_points(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    pcd = o3d.io.read_point_cloud(str(p))
    pts = np.asarray(pcd.points, dtype=float)
    if pts.shape[0] > 0:
        return pts
    mesh = o3d.io.read_triangle_mesh(str(p))
    return np.asarray(mesh.vertices, dtype=float)


def sample_points(points: np.ndarray, max_n: int, seed: int = 7) -> np.ndarray:
    if points.shape[0] <= max_n:
        return points
    rng = np.random.default_rng(seed)
    keep = rng.choice(points.shape[0], size=max_n, replace=False)
    return points[keep]


def make_final_compare(
    tsdf_points: np.ndarray,
    egf_points: np.ndarray,
    stable_bg_points: np.ndarray,
    out_png: Path,
    ghost_thr: float = 0.08,
    recover_bg_thr: float = 0.10,
    recover_tsdf_gap: float = 0.04,
) -> Tuple[int, int]:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    tsdf = np.asarray(tsdf_points, dtype=float)
    egf = np.asarray(egf_points, dtype=float)
    bg = np.asarray(stable_bg_points, dtype=float)
    if tsdf.shape[0] == 0 or egf.shape[0] == 0 or bg.shape[0] == 0:
        raise RuntimeError("tsdf/egf/stable_bg points must all be non-empty")

    bg_tree = cKDTree(bg)
    egf_tree = cKDTree(egf)
    tsdf_tree = cKDTree(tsdf)

    d_tsdf_bg, _ = bg_tree.query(tsdf, k=1)
    d_tsdf_egf, _ = egf_tree.query(tsdf, k=1)
    ghost_mask = (d_tsdf_bg > float(ghost_thr)) & (d_tsdf_egf > float(ghost_thr))
    ghost_pts = tsdf[ghost_mask]

    d_egf_bg, _ = bg_tree.query(egf, k=1)
    d_egf_tsdf, _ = tsdf_tree.query(egf, k=1)
    # "Recovered background details": EGF points close to stable background reference
    # while TSDF has no nearby support at the same location.
    recover_mask = (d_egf_bg < float(recover_bg_thr)) & (d_egf_tsdf > float(recover_tsdf_gap))
    recover_pts = egf[recover_mask]

    tsdf_draw = sample_points(tsdf, 140000, seed=13)
    egf_draw = sample_points(egf, 140000, seed=17)
    ghost_draw = sample_points(ghost_pts, 22000, seed=19)
    recover_draw = sample_points(recover_pts, 16000, seed=23)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    ax0, ax1 = axes

    ax0.scatter(tsdf_draw[:, 0], tsdf_draw[:, 1], s=0.28, c="#8e8e8e", alpha=0.55, label="TSDF Surface")
    if ghost_draw.shape[0] > 0:
        ax0.scatter(ghost_draw[:, 0], ghost_draw[:, 1], s=0.55, c="#d62728", alpha=0.8, label="Ghost (highlight)")
    ax0.set_title("TSDF (Ghost Dominant)")
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")
    ax0.grid(alpha=0.2)
    ax0.set_aspect("equal", adjustable="box")
    ax0.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax1.scatter(egf_draw[:, 0], egf_draw[:, 1], s=0.28, c="#6e6e6e", alpha=0.55, label="EGF-v6 Surface")
    if recover_draw.shape[0] > 0:
        ax1.scatter(
            recover_draw[:, 0],
            recover_draw[:, 1],
            s=0.7,
            c="#2ca02c",
            alpha=0.9,
            label="Recovered Background (highlight)",
        )
    ax1.set_title("EGF-v6 (Clean + Complete)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(alpha=0.2)
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Walking_xyz: TSDF vs EGF-v6\n"
        f"Red=ghost candidates ({ghost_pts.shape[0]} pts), Green=recovered stable background ({recover_pts.shape[0]} pts)"
    )
    fig.savefig(out_png, dpi=260)
    plt.close(fig)
    return int(ghost_pts.shape[0]), int(recover_pts.shape[0])


def make_rho_mechanism_figure(npz_path: Path, out_png: Path) -> None:
    d = np.load(npz_path)
    centers = np.asarray(d["centers"], dtype=float)
    rho = np.asarray(d["rho"], dtype=float)
    d_score = np.asarray(d["d_score"], dtype=float)
    if centers.shape[0] == 0:
        raise RuntimeError(f"empty centers in {npz_path}")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    keep = np.ones((centers.shape[0],), dtype=bool)
    if centers.shape[0] > 220000:
        rng = np.random.default_rng(31)
        idx = rng.choice(centers.shape[0], size=220000, replace=False)
        keep = np.zeros((centers.shape[0],), dtype=bool)
        keep[idx] = True
    c = centers[keep]
    r = rho[keep]
    s = d_score[keep]
    low_q = np.quantile(r, 0.20)
    low_mask = r <= low_q

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)

    sc = axes[0].scatter(c[:, 0], c[:, 1], c=np.clip(r, 0.0, np.quantile(r, 0.95)), s=0.35, cmap="viridis", alpha=0.8)
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.04, pad=0.02)
    cb.set_label("rho (evidence confidence)")
    axes[0].set_title("Evidence Field rho (top view)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].grid(alpha=0.2)
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(c[:, 0], c[:, 1], s=0.25, c="#9a9a9a", alpha=0.45, label="all voxels")
    axes[1].scatter(c[low_mask, 0], c[low_mask, 1], s=0.7, c="#d62728", alpha=0.85, label="low-rho trajectory-like voxels")
    axes[1].scatter(c[s > 0.45, 0], c[s > 0.45, 1], s=0.6, c="#1f77b4", alpha=0.6, label="high d_score")
    axes[1].set_title("Mechanism View: low rho aligns with dynamic path")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].grid(alpha=0.2)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsdf_points", type=str, required=True)
    parser.add_argument("--egf_points", type=str, required=True)
    parser.add_argument("--stable_bg_points", type=str, required=True)
    parser.add_argument("--out_compare", type=str, default="assets/final_comparison_paper_ready.png")
    parser.add_argument("--rho_npz", type=str, default="")
    parser.add_argument("--out_rho", type=str, default="assets/evidence_rho_mechanism.png")
    args = parser.parse_args()

    tsdf = load_points(args.tsdf_points)
    egf = load_points(args.egf_points)
    bg = load_points(args.stable_bg_points)
    g_cnt, r_cnt = make_final_compare(tsdf, egf, bg, Path(args.out_compare))
    print(f"[done] compare figure: {args.out_compare} (ghost={g_cnt}, recovered={r_cnt})")

    if args.rho_npz:
        make_rho_mechanism_figure(Path(args.rho_npz), Path(args.out_rho))
        print(f"[done] rho mechanism figure: {args.out_rho}")


if __name__ == "__main__":
    main()
