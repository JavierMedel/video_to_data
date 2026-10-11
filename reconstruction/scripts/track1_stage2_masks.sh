#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(pwd -P)"
DATA="$ROOT/data/v2d_challenge/track_1"
WEIGHTS="$ROOT/data/weights/sam2"
SAM2="$ROOT/.venv/bin/python"
PACKER="$ROOT/modules/v2d_cari4d/lib/pack_masks.py"
LOG="$DATA/logs/full_pipeline"
EP="${1:-0}"

printf -v ID '%06d' "$EP"
EPDIR="$DATA/episode_$ID"
VIDEO="$DATA/videos/chunk-000/observation.images.exo_camera/episode_$ID.mp4"
BBOX="$EPDIR/object_bboxes.json"
TRACK="$EPDIR/person_tracks.pt"

mkdir -p "$LOG"

[[ "$EP" =~ ^[0-9]+$ ]] || { echo "Episode must be numeric"; exit 2; }
[[ -x "$SAM2" ]] || { echo "Missing Python environment: $SAM2"; exit 1; }
[[ -f "$VIDEO" ]] || { echo "Missing video: $VIDEO"; exit 1; }
[[ -f "$BBOX" && -f "$TRACK" ]] || {
  echo "Missing detections or person tracks for episode $ID"
  exit 1
}
[[ -f "$WEIGHTS/sam2.1_hiera_large.pt" ]] || {
  echo "Missing SAM2 weights: $WEIGHTS/sam2.1_hiera_large.pt"
  exit 1
}

H5="$EPDIR/cari4d_masks.h5"
if [[ -s "$H5" ]]; then
  echo "Masks already packed; refusing to overwrite: $H5"
  exit 0
fi

HUMAN_PROMPT="$EPDIR/sam2_human_prompt.json"
OBJECT_PROMPT="$EPDIR/sam2_prompts.json"
HUMAN_ROOT="$EPDIR/sam2_human_masks"
OBJECT_ROOT="$EPDIR/sam2_object_masks"

echo "Episode $ID: generating prompts"
docker run --rm \
  -v "$ROOT/data:/workspace/data" \
  -v "$ROOT/scripts:/workspace/scripts:ro" \
  --entrypoint python \
  v2d_cari4d:blackwell \
  /workspace/scripts/generate_track1_prompts.py \
  "/workspace/data/v2d_challenge/track_1/episode_${ID}/object_bboxes.json" \
  "/workspace/data/v2d_challenge/track_1/episode_${ID}/person_tracks.pt" \
  "/workspace/data/v2d_challenge/track_1/episode_${ID}/sam2_human_prompt.json" \
  "/workspace/data/v2d_challenge/track_1/episode_${ID}/sam2_prompts.json"

test -s "$HUMAN_PROMPT" || { echo "Missing human prompt: $HUMAN_PROMPT"; exit 1; }
test -s "$OBJECT_PROMPT" || { echo "Missing object prompt: $OBJECT_PROMPT"; exit 1; }

echo "Verified both prompt files."
echo "Running SAM2 for the human..."
"$SAM2" modules/v2d_sam2/docker/run_video_to_masks.py \
  --video_path "$VIDEO" \
  --prompts_path "$HUMAN_PROMPT" \
  --masks_dir "$HUMAN_ROOT" \
  --weights_dir "$WEIGHTS" \
  --dev 2>&1 | tee "$LOG/episode_${ID}_sam2_human.log"

echo "Running SAM2 for the object..."
"$SAM2" modules/v2d_sam2/docker/run_video_to_masks.py \
  --video_path "$VIDEO" \
  --prompts_path "$OBJECT_PROMPT" \
  --masks_dir "$OBJECT_ROOT" \
  --weights_dir "$WEIGHTS" \
  --dev 2>&1 | tee "$LOG/episode_${ID}_sam2_object.log"

for DIR in "$HUMAN_ROOT/0" "$OBJECT_ROOT/1"; do
  [[ -d "$DIR" ]] || {
    echo "Expected SAM2 mask directory missing: $DIR"
    exit 1
  }
  [[ -n "$(find "$DIR" -maxdepth 1 -type f -name '*.png' -print -quit)" ]] || {
    echo "No PNG masks found in $DIR"
    exit 1
  }
done

# A hard link gives the packer the required CARI4D filename convention
# without duplicating the video data.
PACK_VIDEO="$EPDIR/episode_${ID}.0.color.mp4"
if [[ ! -e "$PACK_VIDEO" ]]; then
  ln "$VIDEO" "$PACK_VIDEO"
fi

cleanup() {
  # Only remove the hard link created here; never remove the source video.
  if [[ -e "$PACK_VIDEO" && "$PACK_VIDEO" -ef "$VIDEO" ]]; then
    rm -f "$PACK_VIDEO"
  fi
}
trap cleanup EXIT

echo "Packing human/object masks into CARI4D H5..."
docker run --rm \
  -v "$ROOT/data:/workspace/data" \
  -v "$ROOT/modules:/workspace/modules:ro" \
  --entrypoint python \
  v2d_cari4d:blackwell \
  /workspace/modules/v2d_cari4d/lib/pack_masks.py \
  --video_path "/workspace/data/v2d_challenge/track_1/episode_${ID}/episode_${ID}.0.color.mp4" \
  --human_masks_path "/workspace/data/v2d_challenge/track_1/episode_${ID}/sam2_human_masks/0" \
  --object_masks_path "/workspace/data/v2d_challenge/track_1/episode_${ID}/sam2_object_masks/1" \
  --output_path "/workspace/data/v2d_challenge/track_1/episode_${ID}/cari4d_masks.h5" \
  2>&1 | tee "$LOG/episode_${ID}_pack_masks.log"

[[ -s "$H5" ]] || { echo "Packing did not produce $H5"; exit 1; }
echo "SUCCESS: $H5"
