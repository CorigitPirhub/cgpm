from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.data.tum_rgbd import TUMRGBDStream

# Official sequence names listed on Bonn RGB-D Dynamic Dataset page.
BONN_SEQUENCE_NAMES = [
    "rgbd_bonn_balloon",
    "rgbd_bonn_balloon2",
    "rgbd_bonn_balloon_tracking",
    "rgbd_bonn_balloon_tracking2",
    "rgbd_bonn_crowd",
    "rgbd_bonn_crowd2",
    "rgbd_bonn_crowd3",
    "rgbd_bonn_kidnapping_box",
    "rgbd_bonn_kidnapping_box2",
    "rgbd_bonn_moving_nonobstructing_box",
    "rgbd_bonn_moving_nonobstructing_box2",
    "rgbd_bonn_moving_obstructing_box",
    "rgbd_bonn_moving_obstructing_box2",
    "rgbd_bonn_person_tracking",
    "rgbd_bonn_person_tracking2",
    "rgbd_bonn_placing_nonobstructing_box",
    "rgbd_bonn_placing_nonobstructing_box2",
    "rgbd_bonn_placing_nonobstructing_box3",
    "rgbd_bonn_placing_obstructing_box",
    "rgbd_bonn_removing_nonobstructing_box",
    "rgbd_bonn_removing_nonobstructing_box2",
    "rgbd_bonn_removing_obstructing_box",
    "rgbd_bonn_static",
    "rgbd_bonn_static_close_far",
    "rgbd_bonn_synchronous",
    "rgbd_bonn_synchronous2",
]


def bonn_sequence_url(sequence: str) -> str:
    seq = sequence.strip()
    if not seq:
        raise ValueError("empty Bonn sequence name")
    return f"https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/{seq}.zip"


def _is_valid_sequence_dir(path: Path) -> bool:
    return (path / "rgb.txt").exists() and (path / "depth.txt").exists() and (path / "groundtruth.txt").exists()


def download_bonn_sequence(dataset_root: str | Path, sequence: str) -> Path:
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    seq_dir = root / sequence
    if _is_valid_sequence_dir(seq_dir):
        return seq_dir

    url = bonn_sequence_url(sequence)
    zip_path = root / f"{sequence}.zip"
    if not zip_path.exists():
        print(f"[download] {url}")
        with requests.get(url, stream=True, timeout=90) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    if _is_valid_sequence_dir(seq_dir):
        return seq_dir

    # Fallback: if archive root folder does not strictly match the expected name.
    for cand in sorted(root.glob("rgbd_bonn_*")):
        if cand.is_dir() and _is_valid_sequence_dir(cand):
            if cand.name == sequence:
                return cand
    raise FileNotFoundError(f"Downloaded archive extracted, but valid sequence dir not found for {sequence}")


class BonnRGBDStream(TUMRGBDStream):
    """Bonn dynamic RGB-D stream.

    Bonn uses TUM-compatible text file format and Freiburg1 intrinsics, so we
    can directly reuse the TUM stream implementation.
    """

    def __init__(
        self,
        sequence_dir: str | Path,
        cfg: EGF3DConfig,
        max_frames: int | None = None,
        stride: int = 1,
        max_points: int = 5000,
        assoc_max_diff: float = 0.02,
        normal_radius: float = 0.08,
        normal_max_nn: int = 40,
        seed: int = 42,
    ):
        super().__init__(
            sequence_dir=sequence_dir,
            cfg=cfg,
            max_frames=max_frames,
            stride=stride,
            max_points=max_points,
            assoc_max_diff=assoc_max_diff,
            normal_radius=normal_radius,
            normal_max_nn=normal_max_nn,
            seed=seed,
        )

