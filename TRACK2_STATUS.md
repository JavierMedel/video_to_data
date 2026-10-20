# Track 2 local execution status

This document records the Track 2 campaign on the execution laptop. Track 1
artifacts and its prepared v10 submission are separate and remain preserved.

## Current execution snapshot — October 19, 2026 Toronto

The participant requested a Track 2 submission after the completed Track 3 round.
The existing Track 2 campaign is **Tier 1**, using clean multi-view input and
G1 + Dex3 / SONIC. It is not ready for upload:

- All **30/30** retargeted inputs exist and their hashes were reverified against
  the existing per-episode receipts (Episode 16 uses its separate pilot receipt).
- Episode 16's existing **349,999,104-step** full-budget policy and genuine
  rollout were verified and reused. Its local task success rates remain zero.
- The first new short-budget job, **Episode 17**, completed **5,000,192 steps**
  and a valid **440-row** one-world rollout. Its checkpoint, original reference,
  raw trajectory, metrics, and name-only submission export hashes are recorded in
  `short_campaign/first_short_episode_verified.json`. It took about four minutes
  of reported training-loop time including learner compilation, plus initialization
  and evaluation. Native CUDA memory readings were about 9,306 MiB under training.
- At this snapshot **5/30 episodes have evaluated policies** (16, 17, 29, 6, 20),
  and Episode 27 is active. The remaining 25 episodes require training and
  evaluation; read `short_campaign/execution.json` for current progress.
- The remaining episodes need their own policies and evaluated rollouts.
  Do not reuse Episode 16's policy or submit reference poses for missing episodes.
- The latest official entry check confirms **all four Tier 1 competitions entered**
  for `iorideepp`. The account/rules blocker is cleared. Evidence is recorded in
  `short_campaign/entry_check.json`; older unjoined checks below are historical.
- The participant selected **5,000,192 steps per remaining episode**, matching
  the short-budget Track 3 approach. The controller reuses Episode 16's genuine
  full-budget rollout and trains 29 independent policies. Native nine-stage
  curriculum thresholds are compressed proportionally, retaining zero-assistance
  phases. Simulation timing, SONIC/G1 embodiment, and physical support are preserved.
- Run `reconstruction/.venv/bin/python robotic_grounding/scripts/run_track2_tier1_campaign.py --upload`.
  The new authoritative state is
  `robotic_grounding/incident_reports/track2_baseline/short_campaign/execution.json`.
  It has its own campaign lock, policy/results/export directories and receipt;
  training shares the existing Track 2/3 GPU lock and 21,000 MiB guard.
- New jobs use 1,024 simulation worlds, replay capacity 1M, batch 1,024, and
  minimum replay 102,400. They save checkpoints about every 1M steps. Monitoring-only
  interruptions may resume up to three times per episode; actual memory-cap,
  physics-buffer and invalid-rollout failures stop for review. Never label these
  short policies as native 350M-step training.
- Raw rollout files remain in `results_short`; submission exports in
  `results_short_submission` change only the body-name label from native `object`
  to the verified source-manifest object name. All pose arrays are preserved.
  Exact one-world/source-frame coverage, finite poses and checkpoint hashes are checked.
- The controller packs only after all 30 evaluated rollouts are verified. The
  official packer converts back to the released frame, checks coverage/provenance,
  computes clean-reference local scores, and cross-checks evaluator metrics.
  Upload then requires the official dry-run and all four entered competitions.
  Four accepted metric IDs are polled from a hash-linked exclusive receipt;
  existing/ambiguous receipts are not blindly reuploaded.
- After the participant completed the remaining MPPE entry, the official API
  confirmed all four competitions entered. The controller remains authorized to
  upload after all 30 rollouts pass packing and the official dry-run.
- No Track 2 submission has been uploaded. The valid 20-episode Track 3 submission
  and its six scored receipts remain separate and must not be reused as Track 2 data.

Fresh evidence:

- `robotic_grounding/incident_reports/track2_baseline/submission_readiness_20261019.json`
- `robotic_grounding/incident_reports/track2_baseline/entry_check_20261019.json`
- Historical pilot state: `robotic_grounding/incident_reports/track2_baseline/execution.json`
- Native pilot metrics: `robotic_grounding/data/track2/tier1/results/episode_000016.json`

The original setup/producing commit was
`996cb7d327992c4122255f84a971147f62bcd780`. Future training/export changes require
an updated pushed producing commit, exact checkpoint/evaluator provenance,
complete 30-episode coverage, packer cross-checks against the clean Tier 1 reference,
and the official dry-run before sending one CSV to all four Tier 1 metrics.
Choose a fresh exclusive Kaggle receipt and poll its exact IDs after upload.

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

## First short-budget pipeline validation

The initial Episode 17 launch stopped before training because an unquoted
`sequence_id=...` path was parsed as Hydra override syntax. The launcher now
JSON-quotes the task path, with a regression test. The complete host suite passed
**55 tests**, including preserved Track 3 upload behavior and four-metric Track 2
receipt completion. The first new episode subsequently completed training,
checkpointing, unassisted evaluation, exact-frame validation, and semantic-name
export. Code was pushed before launch; the first new episode used implementation commit
 `d6801a1220d8ddf80b0fa79c41a60e0a8e54a121`.

A clearly marked pilot-only diagnostic CSV under `short_campaign/diagnostics`
was used to test frame conversion. It is **incomplete and must never be uploaded**.
The real campaign pack always uses the default official 30-episode configuration.
The pilot's maximum rigid-frame fit residual was 1.03e-7 m and 8.94e-8 radians.
Official local recomputation: AUC 0.255503, SP-SR 0, MP-SR 0, MPPE 63.1675 cm.
Its AUC differed from the native evaluator's 0.255765 by about 0.000262; the other
three metrics agreed. Both computations refer to the same hash-verified raw
trajectory. Preserve the discrepancy and the diagnostic log; do not silently
suppress cross-check warnings or claim local agreement beyond this evidence.

The current queue is designed to keep running without chat interaction. The
laptop and Docker must remain on. It will pack only verified complete coverage
and upload only after the official dry-run passes. Entry remains an external
blocker: after the participant's renewed confirmation, a direct authenticated
API comparison showed `user_has_entered=true` for Track 3 AUC and `false` for
Track 2 Tier 1 AUC. The participant was asked what their Track 2 rules page shows
(Join Competition versus Submit Prediction). Continue training independently
and recheck entry rather than automatically accepting rules or bypassing checks.

## Released ground-only support stages

After Episode 17 passed, Episode 29 stopped before training because its released
support USDA contains only an empty default Xform named `support_surfaces`.
A USD-runtime audit found **16/30** Tier 1 support files intentionally have this
structure. The support loader now optionally returns no support binding for
that exact structure, and the scene builder permits it only when ground
collision is enabled. It does not disable ground contact, modify released
support files, ignore unsupported geometry, or permit ground-free empty scenes.
The default loader behavior still rejects empty files. Missing files, unknown
roots and unsupported child geometry remain errors.

Eight focused native support/builder tests passed (eight unrelated data tests
deselected). Evidence: `short_campaign/support_contract_audit.json`. The queue
must be restarted after this runtime fix; Episode 17 is retained via its verified
cached receipt. Older error traces remain in the append-only logs. Use the
execution JSON and newest stage log entries to determine current state.


The ground-only correction subsequently passed an actual complete episode:
Episode 29 trained 5,000,192 steps, saved its final checkpoint, evaluated all
467 source frames in one valid world, and exported the same achieved poses.
The saved policy's `user.resolved_config_json` confirms `collision.ground=true`,
`robot_object_collision=true`, `robot_support_collision=true`, and `scene.support=true`.
Evidence: `short_campaign/ground_only_episode_verified.json`. The controller then
advanced to Episode 6 automatically. At this snapshot the queue has 3/30 evaluated
policies; remaining training/evaluation will take several hours. The laptop and
Docker must stay running. No Track 2 upload has occurred, and the official entry
check still needs confirmation. There is no fresh receipt to poll until packing
and the official dry-run have passed.


## Account entry resolved

On 2026-10-20T01:57:45.472715+00:00, the official `--check-entry` returned exit code zero and
confirmed all four Track 2 Tier 1 competitions entered. No more browser rule
acceptance is currently required. Training is still incomplete: 5/30 rollouts
are evaluated, Episode 27 is active, and there is no Track 2 upload receipt yet.
The running controller already has `--upload` and will validate and submit after
all 30 episodes are complete. Older notes about unjoined competitions describe
the earlier checks and must not be treated as current account state.

## Constraint-capacity recovery — Episode 12

The queue completed 8/30 rollouts, then stopped Episode 12 during its initial
training collection: MuJoCo-Warp requested up to 1,119 constraints while the
native G1 preset reserved 1,024. The physics guard correctly stopped the job;
Episode 12 had not produced a policy checkpoint. No incomplete physics rollout
was included in the submission outputs.

Remaining training jobs now explicitly use `sim.njmax=4096` and
`sim.nconmax=1024`, retaining 1,024 worlds and the 21,000 MiB total-device guard.
This increases buffer capacity rather than discarding contacts or reducing
collision fidelity. Existing completed policies/results remain unchanged.
All 55 host tests passed, including the new capacity override assertion.
Evidence: `short_campaign/physics_buffer_recovery.json`; the original stop
receipts are archived under Episode 12's `training/previous_attempts` directory.
The restarted controller also prints start/completion milestones to its main log.
The all-four-competition entry check is already confirmed; no further rule
acceptance is required. Read the execution JSON for progress after this recovery.

### Persistent compilation cache

During the resumed queue, the installed JAX runtime reported compilation caching
enabled but no cache directory configured. Future policy-stage containers now
set `JAX_COMPILATION_CACHE_DIR=/cache/jax` inside the existing persistent runtime
cache mount. A native configuration probe accepted the path. This permits reuse
of matching compiled code; actual cache hits and timing must be measured before
claiming speed gains. It does not increase the training budget. The current
in-flight container keeps its existing configuration; subsequent stages pick up
the new wrapper setting. All 55 host tests passed after the change.
