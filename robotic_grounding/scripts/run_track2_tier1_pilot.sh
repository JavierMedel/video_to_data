#!/usr/bin/env bash
# A bounded hardware smoke or full native-budget training for Tier 1 episode 16.
# Run from any directory. This script never uploads challenge submissions.
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:-smoke}
IMAGE=${TRACK2_IMAGE:-flash_chord:track2-baseline}
SEQUENCE=2026-03-12_15-06-36_spray_bottle_left_desk_ground_02
PARQUET="/repo/data/track2/tier1/assets/human_motion_data/whole_body/challenge_tier1/sequence_id=${SEQUENCE}/robot_name=g1/data.parquet"
mkdir -p "${TASK_ROOT}/data/track2/runtime_cache"
exec 9>"${TASK_ROOT}/data/track2/tier1/pilot.lock"
flock -n 9 || { echo 'Another Tier 1 pilot command is active.' >&2; exit 1; }
COMMON=(--rm --gpus all --user "$(id -u):$(id -g)"
    -e HOME=/tmp -e LOGNAME="$(id -un)" -e PYTHONUNBUFFERED=1
    -e XDG_CACHE_HOME=/cache -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e OMP_NUM_THREADS=8 -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl
    -v "${TASK_ROOT}/flash_chord:/workspace"
    -v "${TASK_ROOT}:/repo"
    -v "${TASK_ROOT}/data/track2/runtime_cache:/cache"
    -w /workspace --entrypoint python "${IMAGE}")
case "${MODE}" in
  smoke)
    if compgen -G "${TASK_ROOT}/data/track2/tier1/pilot_smoke/episode_000016/*.safetensors" >/dev/null; then
      echo 'Smoke checkpoints already exist; use evaluate-smoke.' >&2; exit 1
    fi
    docker run "${COMMON[@]}" scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \
      "task.parquet='${PARQUET}'" scene.world_count=128 \
      training.total_environment_steps=65536 training.replay.capacity=131072 \
      training.replay.minimum_size=8192 training.replay.batch_size=256 \
      training.updates_per_collection=1 training.checkpoint.save_interval_environment_steps=65536 \
      reset.seed=42 logging.mode=disabled logging.log_interval=100 \
      output_dir=/repo/data/track2/tier1/pilot_smoke/episode_000016
    ;;
  evaluate-smoke)
    docker run "${COMMON[@]}" scripts/evaluate_policy.py \
      evaluation.checkpoint=/repo/data/track2/tier1/pilot_smoke/episode_000016/policy_65536.safetensors \
      evaluation.world_count=1 evaluation.reset_mode=explicit \
      evaluation.metrics_output=/repo/data/track2/tier1/pilot_smoke/episode_000016/evaluation.json \
      evaluation.object_trajectories_output=/repo/data/track2/tier1/pilot_smoke/episode_000016/episode_000016.parquet \
      logging.mode=disabled
    ;;
  capacity-smoke)
    if compgen -G "${TASK_ROOT}/data/track2/tier1/capacity_smoke/episode_000016/*.safetensors" >/dev/null; then
      echo 'Capacity smoke checkpoints already exist; preserve them.' >&2; exit 1
    fi
    docker run "${COMMON[@]}" scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \
      "task.parquet='${PARQUET}'" scene.world_count=1024 \
      training.total_environment_steps=524288 training.replay.capacity=1000000 \
      training.replay.minimum_size=102400 training.replay.batch_size=1024 \
      training.checkpoint.save_interval_environment_steps=524288 \
      reset.seed=42 logging.mode=disabled logging.log_interval=100 \
      output_dir=/repo/data/track2/tier1/capacity_smoke/episode_000016
    ;;
  train|train-and-evaluate)
    if compgen -G "${TASK_ROOT}/data/track2/tier1/baseline/episode_000016/*.safetensors" >/dev/null; then
      echo 'Training checkpoints already exist; resume explicitly instead of overwriting.' >&2; exit 1
    fi
    # Keep the released 350M-step curriculum. Reduce worlds and replay capacity
    # for this 24 GB GPU; these are a local configuration, not release reproduction.
    docker run "${COMMON[@]}" scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \
      "task.parquet='${PARQUET}'" scene.world_count=1024 \
      training.replay.capacity=1000000 training.replay.minimum_size=102400 \
      training.replay.batch_size=1024 reset.seed=42 logging.mode=disabled \
      output_dir=/repo/data/track2/tier1/baseline/episode_000016
    if [[ "${MODE}" == train-and-evaluate ]]; then
      mkdir -p "${TASK_ROOT}/data/track2/tier1/results"
      docker run "${COMMON[@]}" scripts/evaluate_policy.py \
        evaluation.checkpoint=/repo/data/track2/tier1/baseline/episode_000016/policy_349999104.safetensors \
        evaluation.world_count=1 evaluation.reset_mode=explicit \
        evaluation.metrics_output=/repo/data/track2/tier1/results/episode_000016.json \
        evaluation.object_trajectories_output=/repo/data/track2/tier1/results/episode_000016.parquet \
        logging.mode=disabled
    fi
    ;;
  *) echo 'Usage: run_track2_tier1_pilot.sh {smoke|evaluate-smoke|capacity-smoke|train|train-and-evaluate}' >&2; exit 2 ;;
esac
