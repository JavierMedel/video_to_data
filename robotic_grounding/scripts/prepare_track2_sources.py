#!/usr/bin/env python3
"""Adapt released Track 2 LeRobot episodes to the existing SOMA-to-G1 input contract.

Object poses stay in the released OpenCV world frame. The existing retargeter
applies its documented CV-to-SOMA conversion and first-frame ground alignment.
Missing object observations are interpolated only to create a continuous policy
input; they remain marked in provenance and never replace visible observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def continuous_object_poses(poses: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Preserve visible wxyz poses; linearly interpolate position and slerp rotation."""
    result = np.array(poses, dtype=np.float64, copy=True)
    visible = np.asarray(visible, dtype=bool)
    if result.ndim != 2 or result.shape[1] != 7 or visible.shape != (len(result),):
        raise ValueError("expected poses (T,7) and visibility (T,)")
    good = np.flatnonzero(visible)
    if not len(good):
        raise ValueError("episode has no observed object poses")
    if not np.isfinite(result[good]).all():
        raise ValueError("nonfinite observed pose")
    norms = np.linalg.norm(result[good, 3:], axis=1)
    if not np.allclose(norms, 1, atol=1e-4, rtol=0):
        raise ValueError("observed object quaternion is not unit length")
    for frame in np.flatnonzero(~visible):
        index = np.searchsorted(good, frame)
        if index == 0 or index == len(good):
            result[frame] = result[good[0] if index == 0 else good[-1]]
            continue
        left, right = good[index - 1], good[index]
        fraction = (frame - left) / (right - left)
        result[frame, :3] = (1 - fraction) * result[left, :3] + fraction * result[right, :3]
        a, b = result[left, 3:].copy(), result[right, 3:].copy()
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        dot = float(a @ b)
        if dot < 0:
            b, dot = -b, -dot
        angle = np.arccos(np.clip(dot, -1, 1))
        q = ((1 - fraction) * a + fraction * b) if angle < 1e-6 else (
            np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b
        ) / np.sin(angle)
        result[frame, 3:] = q / np.linalg.norm(q)
    return result


def poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    q = poses[:, 3:] / np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
    w, x, y, z = q.T
    matrix = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    matrix[:, :3, :3] = np.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ], axis=1).reshape(-1, 3, 3)
    matrix[:, :3, 3] = poses[:, :3]
    return matrix


def prepare(dataset: Path, output: Path, episodes: list[int]) -> list[dict]:
    info = json.loads((dataset / "meta/info.json").read_text())
    names = info["features"]["observation.human.pose"]["names"]
    joints = [name.removesuffix(".rx") for name in names[::3]]
    expected = [f"{joint}.{axis}" for joint in joints for axis in ("rx", "ry", "rz")]
    if names != expected or len(joints) != 77:
        raise ValueError("unexpected SOMA joint order")
    metadata = {m["episode_index"]: m for m in map(json.loads, (
        dataset / "meta/episodes_metadata.jsonl").read_text().splitlines())}
    records = []
    for episode in episodes:
        meta = metadata[episode]
        source = dataset / f"data/chunk-000/episode_{episode:06d}.parquet"
        inputs = [source, dataset / meta["mesh"], dataset / meta["ground_plane"],
                  dataset / meta["support_surface"], dataset / "meta/info.json"]
        hashes = {str(path.resolve()): sha256(path) for path in inputs}
        destination = output / meta["sequence_id"]
        if destination.exists():
            previous = json.loads((destination / "source_manifest.json").read_text())
            if previous["input_sha256"] != hashes or any(
                sha256(destination / name) != digest for name, digest in previous["output_sha256"].items()
            ):
                raise ValueError(f"changed inputs or outputs at {destination}; use a new output root")
            records.append(previous)
            continue
        table = pq.read_table(source)
        if (table.schema.metadata or {}).get(b"pose_convention") != b"world_T_object":
            raise ValueError("unsupported object frame convention")
        def values(name):
            return np.asarray(table[name].to_pylist(), dtype=np.float32)
        frame = np.asarray(table["frame_index"].to_pylist())
        if not np.array_equal(frame, np.arange(len(table))):
            raise ValueError("source frame indices must be contiguous from zero")
        visible = np.asarray(table["observation.object.visible"].to_pylist(), dtype=bool)
        objects = continuous_object_poses(values("observation.object.pose"), visible)
        arrays = {
            "poses": values("observation.human.pose").reshape(-1, 77, 3),
            "transl": values("observation.human.translation"),
            "identity_coeffs": values("observation.human.identity_coeffs"),
            "scale_params": values("observation.human.scale_params"),
            "bone_length_flexibles": values("observation.human.bone_length_flexibles"),
        }
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("nonfinite human source parameters")
        destination.mkdir(parents=True)
        np.savez_compressed(destination / "soma_params.npz", **arrays,
            joint_names=np.asarray(joints), identity_model_type="mhr", unit="meters",
            keep_root=False, absolute_pose=False, rotation_repr="rotvec")
        np.save(destination / "poses.npy", poses_to_matrices(objects))
        (destination / "reconstructed_mesh").mkdir()
        shutil.copy2(inputs[1], destination / "reconstructed_mesh/output_aligned.glb")
        shutil.copy2(inputs[2], destination / "ground_plane.json")
        shutil.copy2(inputs[3], destination / "support.usda")
        record = {"episode_index": episode, "sequence_id": meta["sequence_id"],
            "object": meta["object"], "frames": len(table), "fps": info["fps"],
            "coordinate_frame": "released_track2_opencv_world", "input_sha256": hashes,
            "interpolated_object_frames": np.flatnonzero(~visible).tolist(),
            "adapter_sha256": sha256(Path(__file__)), "output_sha256": {
                str(path.relative_to(destination)): sha256(path)
                for path in destination.rglob("*") if path.is_file()}}
        (destination / "source_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(f"prepared episode {episode}: {len(table)} frames", flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(30)))
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.episodes)
