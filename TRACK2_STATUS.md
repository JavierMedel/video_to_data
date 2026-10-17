# Track 2 local execution status

This document records the Track 2 campaign on the execution laptop. Track 1
artifacts and its prepared v10 submission are separate and remain preserved.

## Objective and initial scope

The user requested starting the Track 2 challenge. Begin with a real Tier 1
G1 + Dex3 policy baseline, validate the complete train/save/evaluate/export path
on one episode, then expand to the 30 scored episodes. Do not label diagnostic
checkpoints as trained submission candidates.

The official challenge requires a separate submission for each of three input
tiers. All are evaluated against clean Track 2 references. Tier 2 training must
use its released noisy input, and Tier 3 training must use reconstruction input;
clean Tier 1 trajectories must not be used to repair their training inputs.
Track 2 episode numbers describe different recordings from Track 1. Never use
the Track 1 v8/v10 episode artifacts as if their IDs matched Track 2.

Official sources:

- https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf
- https://huggingface.co/datasets/nvidia/video_to_data_challenge
- Local submission contract: `reconstruction/v2d_submission_kit/v2d_submission_kit/tracks/track_2/README.md`.

The five-submissions-per-week limit applies independently per track and tier
(official FAQ). Track 1's current hold does not consume Track 2's allowance.
Do not interpret backend daily quota as the complete challenge policy.

## Hardware and runtimes

- Execution host: `/home/alien51/video_to_data`, WSL2/Linux, 24 visible CPU cores,
  approximately 31 GiB host RAM and 8 GiB swap.
- GPU: NVIDIA RTX 5090 **Laptop** GPU, 24,463 MiB, Blackwell `sm_120`.
- About 628 GiB filesystem space was available during initial setup.
- Training runtime: repository-native FlashCHORD 0.4.0,
  `robotic_grounding/flash_chord/`, its frozen `uv.lock`, and
  `docker/Dockerfile`. Image tag: `flash_chord:track2-baseline`.
- The published native full training recipe was validated on a 48 GiB GPU.
  The local configuration reduces parallel worlds and replay storage. It is
  not a reproduction of that hardware/configuration validation.
- Retarget runtime: `v2d_track2_retarget:baseline`, built from
  `robotic_grounding/workflow/Dockerfile.track2-retarget`. This is a separate
  derived image; it does not modify `v2d_cari4d:blackwell`.
- The reconstruction image's NumPy 1.26 constraint conflicts with Pinocchio's
  pinned Boost closure, which requires NumPy 2.2. The retarget image clears that
  inherited pip constraint and installs NumPy 2.2.6 explicitly. This runtime is
  for SOMA/Pink retargeting, not reconstruction.
- Git LFS assets for FlashCHORD and the G1 retargeting robot were fetched from
  the NVIDIA upstream repository. In particular the two SONIC dynamic-batch
  ONNX files are real weights, not 133-byte pointer stubs.

## Inputs and source preparation

Tier 1's non-video release was downloaded to:

`reconstruction/data/v2d_challenge/track_2/tier_1_multiview_caption/`

It includes 30 episode Parquets, 29 object meshes, 30 fitted ground planes,
30 support stages and the metadata. There are **22,990 frames** in total.
Videos are not required for the Tier 1 retarget/training baseline and were
excluded from this initial download.

`robotic_grounding/scripts/prepare_track2_sources.py` adapts the released
LeRobot columns to the existing `soma_to_g1.py` input format. All 30 episodes
have been prepared under:

`robotic_grounding/data/track2/tier1/source_episodes/<sequence_id>/`

Each directory contains `soma_params.npz`, `poses.npy`, the unchanged GLB,
`ground_plane.json`, `support.usda`, and `source_manifest.json` with input/output
SHA-256 hashes. The adapter preserves all human poses, translations, identity,
scales and time-varying flexible-bone parameters. Object quaternion order is
explicitly **wxyz** in the released input. `poses.npy` stores homogeneous
`world_T_object` matrices in the released OpenCV world frame. The adapter does
not preapply the retargeter's camera/source/robot frame transforms.

Visible object observations are preserved. Episode 11 has one invisible frame;
the missing policy input pose is interpolated with linear translation and
shortest-arc quaternion SLERP, with endpoint holding if needed. Its frame index
is recorded in the manifest. The original released Parquet remains untouched.

SOMA assets are pinned to revision
`466879a83d57eabf3d875ded2d869f2075f90348` of `nvidia/SOMA-X`, staged under
`robotic_grounding/data/track2/soma_assets/`. The bundle has the legacy embedded
rig required by `read_soma.py`; do not replace it with mutable latest assets.
Asset hashes and all source manifests are recorded in:

`robotic_grounding/incident_reports/track2_baseline/source_preparation.json`

All generated datasets and checkpoints under `robotic_grounding/data/` are
locally ignored. Preserve artifacts there; commit code and reports, not the
large regenerated data tree.

## Retargeter correction and completed pilot preparation

The existing SOMA reader discarded optional `bone_length_flexibles` even
though every released episode contains nonzero, time-varying values. The reader
now forwards finite `(T,6)` arrays through `SOMALayer(...,
kwargs={"bone_length_flexibles": ...})`, matching the reconstruction exporter.
Inputs without this optional field retain the existing behavior.

Pilot: **Episode 16**, spray bottle, sequence
`2026-03-12_15-06-36_spray_bottle_left_desk_ground_02`.

- Complete retarget: 477/477 source frames.
- G1 has 43 saved joint positions per frame, including hands.
- Fitted ground-plane leveling used, with 2.948-degree initial tilt corrected.
- No constant object ground lift was required; zero robot-only penetration
  lifts were applied.
- Left-hand contact active in 184/477 frames at the native 2 cm threshold;
  right-hand contact active in zero frames.
- Saved sole Z range: approximately 0.0024–0.1137 m. This is preparation
  evidence, not proof of successful dynamic policy execution.
- Independent SE(3) check: a single constant transform maps all original
  object poses to the retargeted object poses, maximum translation residual
  `1.028e-7 m`, maximum rotation-matrix Frobenius residual `1.195e-7`.
- Retarget Parquet SHA-256:
  `1e297bca97786d88f9f4178c04273c06f6f190ab90e6c45d9965659aa8c5fb63`.

Retarget path:

`robotic_grounding/data/track2/tier1/assets/human_motion_data/whole_body/challenge_tier1/sequence_id=2026-03-12_15-06-36_spray_bottle_left_desk_ground_02/robot_name=g1/data.parquet`

The released support files were copied unchanged alongside the motion tree at
`.../human_motion_data/whole_body/reconstructed_stage/`. Do not silently disable
support collisions to get a training command to run.

Verification so far: eleven tests pass, covering source-frame quaternion
conversion, interpolation (including antipodal quaternions), invalid visible
observations, pinned asset requirements, and forwarding of the time-varying
bone parameters, and rejection of per-frame object position/rotation drift.
The full pilot retarget completed in the actual GPU runtime.

The remaining 29 retargets run sequentially using
`robotic_grounding/scripts/retarget_track2_tier1.py`. It pins source-code,
source-artifact, SOMA/robot asset and image hashes; verifies frame counts and
one constant object-frame transform; and resumes only matching completed
outputs. Per-episode reports are under
`robotic_grounding/incident_reports/track2_baseline/retarget/`. Episode 16 is
excluded by default because its separately validated pilot output is preserved.
The batch itself is preparation, not policy training or a submission.

## Training and export commands

The executable workflow is
`robotic_grounding/scripts/run_track2_tier1_pilot.sh`. Invoke it with Bash:

```bash
bash robotic_grounding/scripts/run_track2_tier1_pilot.sh smoke
bash robotic_grounding/scripts/run_track2_tier1_pilot.sh evaluate-smoke
bash robotic_grounding/scripts/run_track2_tier1_pilot.sh capacity-smoke
# Start only after successful hardware smoke and resource review:
bash robotic_grounding/scripts/run_track2_tier1_pilot.sh train-and-evaluate
```

The **diagnostic smoke** uses 128 worlds, 65,536 environment steps, replay
capacity 131,072, minimum replay 8,192, batch size 256, and seed 42. It retains
the native curriculum for compatibility, so it will not progress through its
350M-step assistance schedule. It must not be submitted or reported as a
converged unassisted policy.

Smoke outputs: `robotic_grounding/data/track2/tier1/pilot_smoke/episode_000016/`.
Evaluation records actual achieved object poses, with one world and explicit
reset, using the native evaluator. The evaluator forces zero object assistance
and preserves the entire sequence after failure, rather than truncating or
freezing bad rollouts. Do not replace achieved poses with reference poses.

The **capacity smoke** tests 1,024 worlds, the one-million-transition replay
buffer, batch size 1,024, four learner updates per collection, and 524,288 steps.
This checks the resources required for longer training. It is also diagnostic,
not a converged policy. Sampled combined GPU use with a concurrent retarget
was approximately 13.5 GB during initialization. The run completed all 524,288
steps and saved both actor and resumable training state, with wrapper exit 0.
The training loop, including learner compilation, took about 52 seconds.
This short timing is not a reliable full-run ETA because compilation and
curriculum/contact changes affect throughput. Evidence: `capacity_smoke.json`
under the campaign report directory.

The **full local baseline** keeps the native 349,999,104-step G1
FlashSAC curriculum, while reducing worlds to 1,024, replay capacity to 1,000,000,
minimum replay to 102,400, and batch size to 1,024. Checkpoints go to
`robotic_grounding/data/track2/tier1/baseline/episode_000016/`.
Do not overwrite existing checkpoints; resume explicitly from the saved state.
Runtime, memory feasibility and policy quality must be measured, not inferred
from successful retargeting.

`train-and-evaluate` automatically evaluates the final actor without assistance
after training finishes, writing `episode_000016.json` and
`episode_000016.parquet` under `robotic_grounding/data/track2/tier1/results/`.
It does not package or upload an incomplete 30-episode submission.

The runner uses persistent runtime caches and a host lock. No external W&B
logging or challenge uploads are performed by it. Do not edit a Bash launcher
while an active Bash process is reading it; use a separate command or wait
until it exits.

Completed first diagnostic result:

- 65,536 environment steps, saved actor and resumable state.
- Actor SHA-256:
  `8560bf5507f66b7d74a3e7562dca6da0102b7045d9afc697eaac0304291c2113`.
- One unassisted evaluation world, all 477 frames exported, no nonfinite worlds,
  contiguous source-frame indices and all `environment_valid=true`.
- Local native MPPE: **57.6851 cm**; ADD AUC: **0.263234**; task success zero.
  These are one-episode native metrics, not Kaggle scores or 30-episode means.
  They confirm plumbing, not adequate task performance.
- The evaluator finished successfully, but editing the Bash launcher during
  execution caused the outer wrapper to report an error after export. All
  saved outputs were independently checked and preserved. The launcher is
  frozen during subsequent runs; no repeated evaluation overwrote this result.
- Full diagnostic evidence and hashes: `robotic_grounding/incident_reports/track2_baseline/pilot_smoke.json`.

## Registration and submission state

The user confirmed both organizer registration for Track 2 and acceptance of
all four Tier 1 rules, and then confirmed using account `iorideepp`, which is
the account configured on this machine. Subsequent API entry status must be
checked before upload; an earlier API check still reported all four unjoined.
That discrepancy is not a reason to repeat user permission requests or stop
local computation. Do not automatically accept legal terms for the user.

No Track 2 package has been uploaded. No policy-derived leaderboard scores
exist for this campaign. Retarget success, a saved diagnostic checkpoint, or
reference tracking metrics during assisted training are not submission scores.

Before any submission:

1. Finish and review unassisted policy evaluation for all 30 Tier 1 episodes.
2. Use the official packer with `--track track_2_tier1`, the actual challenge
   dataset root, and a public producing Git commit URL. Preserve reference-frame
   provenance and checkpoint/evaluator hashes.
3. Validate the exact required rows against the supplied sample submission.
4. Verify four-competition entry status, policy allowance and uploader dry run.
5. Send the same CSV to all four metric leaderboards, save an exclusive receipt,
   and poll that receipt instead of blindly retrying an ambiguous upload.
6. Keep Tier 2 and Tier 3 training artifacts and input references separate.

This document describes the initial setup. Consult the latest execution report
under `robotic_grounding/incident_reports/track2_baseline/` for training and
evaluation outcomes before resuming work.
