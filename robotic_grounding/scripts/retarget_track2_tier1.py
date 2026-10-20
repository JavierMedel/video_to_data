#!/usr/bin/env python3
"""Retarget remaining Tier 1 episodes sequentially, preserving the validated episode-16 pilot.

Uses at most one GPU container at a time. Resume only verified, hash-matching
outputs. A failed or ambiguous existing output is preserved for investigation.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare_track2_sources import poses_to_matrices, sha256


def verify_motion(parquet: Path, source: Path) -> dict:
    cells = pq.read_table(parquet).to_pylist()[0]
    raw = np.load(source / "poses.npy")
    position = np.asarray(cells["object_body_position"]).reshape(-1, 3)
    quaternion = np.asarray(cells["object_body_wxyz"]).reshape(-1, 4)
    joints = np.asarray(cells["robot_joint_positions"])
    if len(joints) != len(raw) or not np.isfinite(joints).all():
        raise ValueError("retargeted robot frame count or finite-value check failed")
    if quaternion.shape != (len(raw), 4) or not np.allclose(
        np.linalg.norm(quaternion, axis=1), 1, atol=1e-5, rtol=0
    ):
        raise ValueError("retargeted object quaternion check failed")
    achieved = poses_to_matrices(np.c_[position, quaternion])
    transform = achieved[0] @ np.linalg.inv(raw[0])
    expected = transform[None] @ raw
    position_error = float(np.linalg.norm(expected[:, :3, 3] - position, axis=1).max())
    rotation_error = float(np.linalg.norm(
        expected[:, :3, :3] - achieved[:, :3, :3], axis=(1, 2)).max())
    if position_error > 1e-6 or rotation_error > 1e-6:
        raise ValueError(f"object motion changed beyond one rigid transform: {position_error}, {rotation_error}")
    return {"frames": len(raw), "robot_joint_count": joints.shape[1],
        "object_translation_se3_max_error_m": position_error,
        "object_rotation_se3_max_frobenius_error": rotation_error,
        "released_to_retarget_world": transform.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=[i for i in range(30) if i != 16])
    parser.add_argument("--image", default="v2d_track2_retarget:baseline")
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--reports", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = (args.data_root or root / f"data/track2/tier{args.tier}").resolve()
    data.relative_to(root)
    reports = args.reports or root / ("incident_reports/track2_baseline/retarget" if args.tier == 1 else f"incident_reports/track2_tier{args.tier}/retarget")
    reports.mkdir(parents=True, exist_ok=True)
    lock = (data / "retarget.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifests = {r["episode_index"]: p.parent for p in (data / "source_episodes").glob("*/source_manifest.json")
                 for r in [json.loads(p.read_text())]}
    image_id = subprocess.check_output(["docker", "image", "inspect", args.image,
        "--format", "{{.Id}}"], text=True).strip()
    source_files = [root / "scripts/retarget/soma_to_g1.py", Path(__file__)]
    for directory in [root / "source/robotic_grounding/robotic_grounding/retarget",
                      root / "source/robotic_grounding/robotic_grounding/motion_schema"]:
        source_files.extend(p for p in directory.rglob("*") if p.suffix in {".py", ".json"})
    code_sha = {str(p.relative_to(root)): sha256(p) for p in source_files}
    asset_files = [p for p in (root / "data/track2/soma_assets").rglob("*")
                   if p.is_file() and ".cache" not in p.parts]
    for directory in [root / "source/robotic_grounding/robotic_grounding/assets/urdfs/g1",
                      root / "source/robotic_grounding/robotic_grounding/assets/meshes/g1"]:
        asset_files.extend(p for p in directory.rglob("*") if p.is_file())
    asset_sha = {str(p.relative_to(root)): sha256(p) for p in asset_files}
    for episode in args.episodes:
        source = manifests[episode]
        manifest = json.loads((source / "source_manifest.json").read_text())
        if manifest.get('tier', 1) != args.tier:
            raise ValueError('Source manifest belongs to another Track 2 tier')
        for name, digest in manifest["output_sha256"].items():
            if sha256(source / name) != digest:
                raise ValueError(f"changed source artifact: {source / name}")
        target = data / f"assets/human_motion_data/whole_body/challenge_tier{args.tier}/sequence_id={source.name}/robot_name=g1/data.parquet"
        report_path = reports / f"episode_{episode:06d}.json"
        inputs = {"source_manifest_sha256": sha256(source / "source_manifest.json"),
                  "image_id": image_id, "code_sha256": code_sha, "asset_sha256": asset_sha}
        if target.exists():
            previous = json.loads(report_path.read_text()) if report_path.exists() else {}
            if previous.get("inputs") != inputs or previous.get("output_sha256") != sha256(target):
                raise ValueError(f"existing output lacks matching provenance; preserve and review {target}")
            verify_motion(target, source)
            print(f"episode {episode}: verified prior result", flush=True)
            continue
        command = ["docker", "run", "--rm", "--gpus", "all", "--user", "1000:1000",
            "-e", "HOME=/tmp", "-e", "LOGNAME=alien51", "-e", "PYTHONUNBUFFERED=1",
            "-e", "OMP_NUM_THREADS=4", "-v", f"{root}:/repo", args.image,
            "scripts/retarget/soma_to_g1.py", f"/repo/{source.relative_to(root)}", "--save",
            "--soma-data-root", "/repo/data/track2/soma_assets", "--motion-root",
            f"/repo/{data.relative_to(root)}/assets/human_motion_data", "--soma-subdir", f"challenge_tier{args.tier}"]
        started = time.time()
        log_path = reports / f"episode_{episode:06d}_attempt_{time.time_ns()}.log"
        with log_path.open("x") as log:
            print(f"episode {episode}: retargeting {manifest['frames']} frames", flush=True)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        review = verify_motion(target, source)
        record = {"status": "RETARGET_VERIFIED", "episode_index": episode,
            "inputs": inputs, "output_sha256": sha256(target), "command": command,
            "elapsed_seconds": time.time() - started, "log": str(log_path), **review}
        report_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"episode {episode}: verified {review['frames']} frames", flush=True)


if __name__ == "__main__":
    main()
