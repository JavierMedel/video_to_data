#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(pwd -P)"
cd "$ROOT"

DATA="$ROOT/data/v2d_challenge/track_1"
VIDEO_DIR="$DATA/videos/chunk-000/observation.images.exo_camera"
META="$DATA/meta/episodes_metadata.jsonl"
WEIGHTS="$ROOT/data/weights"
OUT="$DATA/submission_episodes"
LOG="$DATA/logs/full_pipeline"
KIT="$ROOT/v2d_submission_kit"

mkdir -p "$LOG" "$OUT"

MODE="${1:-check}"

# Required inputs. Do not silently download or replace weights.
for f in \
  "$META" \
  "$WEIGHTS/sam2/sam2.1_hiera_large.pt" \
  "$WEIGHTS/cari4d/2026-08-25-09-35-57/step200000.pth"
do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file missing: $f" >&2
    exit 1
  fi
done

for f in \
  modules/v2d_sam2/docker/run_video_to_masks.py \
  modules/v2d_sam3d/docker/run_image_to_mesh.py \
  modules/v2d_cari4d/lib/pack_masks.py \
  modules/v2d_cari4d/lib/run_inference.py
do
  if [[ ! -f "$ROOT/$f" ]]; then
    echo "ERROR: expected pipeline entry point missing: $f" >&2
    exit 1
  fi
done

echo "Repository: $ROOT"
echo "Mode: $MODE"
echo "Logs: $LOG"
echo
echo "=== Episode inventory ==="

python3 - "$DATA" "$VIDEO_DIR" "$OUT" <<'PY'
import json, pathlib, sys

data, videos, out = map(pathlib.Path, sys.argv[1:])
rows = [
    json.loads(line)
    for line in (data / "meta/episodes_metadata.jsonl").read_text().splitlines()
    if line.strip()
]
rows.sort(key=lambda r: int(r["episode_index"]))

if len(rows) != 30:
    raise SystemExit(f"Expected 30 metadata records; found {len(rows)}")

missing = []
for row in rows:
    i = int(row["episode_index"])
    stem = f"episode_{i:06d}"
    video = videos / f"{stem}.mp4"
    work = data / stem
    submission = out / f"{stem}.npz"
    meshes = [
        out / f"{stem}_object.glb",
        out / f"{stem}_object.obj",
        out / f"{stem}_object.ply",
        out / f"{stem}_object.stl",
        out / f"{stem}_object.off",
    ]
    mesh_exists = any(p.is_file() for p in meshes)
    print(
        f"{i:02d} {row['object']:<20} "
        f"video={'OK' if video.is_file() else 'MISSING':7} "
        f"bboxes={'OK' if (work/'object_bboxes.json').is_file() else 'MISSING':7} "
        f"tracks={'OK' if (work/'person_tracks.pt').is_file() else 'MISSING':7} "
        f"masks={'OK' if (work/'cari4d_masks.h5').is_file() else 'PENDING':7} "
        f"npz={'OK' if submission.is_file() else 'PENDING':7} "
        f"mesh={'OK' if mesh_exists else 'PENDING'}"
    )
    if not video.is_file():
        missing.append(str(video))
if missing:
    raise SystemExit("Missing videos:\n" + "\n".join(missing))
PY

if [[ "$MODE" == "check" ]]; then
  echo
  echo "Preflight complete. No models were run and no submission files changed."
  echo "Next: use the run mode only after the per-episode stages are configured."
  exit 0
fi

echo
echo "ERROR: full inference is deliberately not guessed by this script."
echo "The prompt schema, CARI4D output-to-Track-1 conversion, and frame counts"
echo "must be verified before enabling destructive or expensive model stages."
exit 2
