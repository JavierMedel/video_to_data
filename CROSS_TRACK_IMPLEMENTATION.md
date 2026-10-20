# Cross-track improvement implementation

Implementation of the [original cross-track proposal](https://github.com/JavierMedel/video_to_data/blob/cff4d10d3267cdaf1b2f6a6c98be1500bdeb304e/V2D_CROSS_TRACK_IMPROVEMENT_PLAN.md), October 20, 2026.
The superseded proposal was removed from the current tree; Git history preserves it.
Current planning documents: [next execution plan](V2D_NEXT_EXECUTION_AND_VALIDATION_PLAN.md)
and its [review and required corrections](V2D_NEXT_EXECUTION_AND_VALIDATION_PLAN_REVIEW.md).
This change adds implementation and tests. It does not execute the plan's training
schedule, download datasets, create a new reconstruction candidate, or upload a
submission. Existing submitted packets and checkpoint directories are preserved.

## Entry points

Use the execution laptop at `/home/alien51/video_to_data`:

```bash
reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py --help
```

All subcommands have their own `--help`. `jobs` validates its manifest by default;
GPU training requires `--execute`. `submission submit` checks its inputs by default;
upload requires `--execute`. `checkpoint-select` requires `--run-evaluations` to
create missing native rollouts. Fitting commands explicitly process supplied inputs;
`human-fit` defaults to CPU. Its CUDA option shares the existing GPU lock, writes
native CUDA heartbeats, and enforces the 21,000 MiB global-memory guard.

The planning laptop reads the plan and writes Markdown. Dataset preparation,
Docker commands, reconstruction, fitting, training, evaluation and submissions run
on this laptop. Cloning the repository does not copy local inputs or weights.

## Changes to assumptions in the proposed plan

- Track 2 disables initial-pose alignment in the configured host scorer. Its
  trajectories need the official rigid conversion back to the released world;
  that conversion does not erase initial placement error. The local selector uses
  the configured switch, canonical mesh sample, float64 pose arrays and the same
  six-decimal CSV serialization and quaternion normalization as submission.
- Track 2's G1 curriculum has **nine** stages; Track 3's Sharpa curriculum has
  **eleven**. Their thresholds are read from their actual YAML files. Fresh runs
  scale all stages; continuing a 5M learner retains its original thresholds so
  extra steps remain unassisted. Resume retains the saved world count.
- Track 2 Tier 3 supplies **no trajectory** (`input_path: null` in the official
  configuration). The Tier 3 preparer requires a real reconstruction manifest,
  source-video hashes and output hashes. It cannot synthesize a Tier 3 baseline by
  copying the clean Tier 1 trajectories.
- PEN is not a distance to the reference. A reconstruction scored against itself
  can have nonzero PEN. The harness tests unchanged surfaces and containment;
  it never requires self-PEN to be zero.
- For three independent Gaussian position components, mean acceleration norm is
  `sqrt(6)*sigma*2*sqrt(2/pi)`, not scalar RMS `sqrt(6)*sigma`. The test uses the
  actual three-dimensional norm and the official endpoint weighting.
- Jitter magnitudes and leaderboard ties do not establish the hidden reference's
  acceleration or predict a candidate's score. Own-motion magnitudes are labelled
  proxies throughout. Successful implementation does not establish the plan's
  estimated score gains.
- Weekly windows and cross-episode warm-start permissions remain recorded rule
  decisions, not assumptions about Monday. The new upload gate defaults to
  unresolved and performs no upload until documented rules and fresh histories
  are supplied. No organizer contact or new registration occurs in this change.

## Implemented components

| Plan area | Implementation | Required execution evidence |
| --- | --- | --- |
| C.1 | `submission snapshot`: all 23 configured slugs, paginated histories, own best/latest and rank 1/3/5/10 targets | Authenticated API access when explicitly run |
| C.2 / 7 | Freshness checks, per-competition attempt counts, explicit weekly window, coherent packet, hash-bound exclusive receipt, exact-ID polling | A documented rules decision; unresolved interpretations stay unresolved |
| C.3 | `diagnostics` JSON/CSV for baseline object motion, body root/articulation contributions, policy tracking errors, success margins and budgets | Decoded participant body assets for human diagnostics; full hand/support observations for feasibility |
| 3.1 / 3.2 | Official metric harness; recenter surface-equivalence test; sparse robust QP over contiguous scored runs; exact first-frame anchors | Check packed artifacts after serialization, rather than infer neutrality from the fit alone |
| 3.2 / 3.4 | Contact-window targets; winding/triangle-distance half-spaces; five constraint updates; minimal rigid correction and re-fit; serialized silhouette/contact/PEN gates; strongest passing lambda | Full scored-frame masks, calibrated camera transforms, hand points and normals bound to parent hashes |
| 3.2.3 | Sparse SO(3) angular-acceleration fit | Symmetry changes require observed symmetry/image evidence; no unmeasured flips are enabled |
| 3.2.4 | Largest/supported connected-component mask ablations with >=.05 IoU gain on every supplied episode | All scored masks for the episodes being changed |
| 3.3 | Official-space root decomposition; root-translation and SO(3) root-rotation fitting; differentiable float64 FK/correctives/LBS port; joint-space refit; image/vertex/hand gates | The Torch port must pass on >=100 real frames before a refit; synthetic test parity alone does not promote a human fit |
| 3.6 | `assemble-track1`: 30-episode assembly, default >=25 reviewed promotions, unchanged rejected parents, identity/anchor checks and optional official packing | Complete object reviews; a separate passing human review when the pose changes; producing code must be committed before packing |
| 4.1 / 4.2 | Tier-owned source preparation, noise-limited position and SO(3) conditioning, generalized retargeting, complete verified inventory, conditioned-world packer frame references, isolated tier/version campaign outputs | Actual tier input dataset; Tier 3 reconstruction lineage; retargeted kinematics must validate before training |
| 4.3 / 4.5 | Versioned fresh/resume/warm-start job builders; pilot-arm comparison; threshold-closeness ranking; solved-episode exclusion; collapse/retry decision helper; explicit one-job-at-a-time runner | Three matched pilots, deterministic rollout scores, and measured budget decisions; the helpers do not invent pilot winners |
| 4.4 | Inventory all actor checkpoints; deterministic one-world native evaluation; exact Tier 1 lexicographic selection; complete selection packing with unchanged pose columns | All 30 episodes; native checkpoint digests; complete source frames |
| 5.1 | Four matched O/M/R/A development job arms; official mesh registration, symmetry-aware pose metrics and CD-O functions; bottleneck comparison | Public development references, scans, anchors, symmetry metadata and canonical points; none are taken from evaluation/holdout data |
| 5.2 | Largest-component/hole/Taubin mesh cleanup, rigid candidate-to-parent registration, radius and 5mm median checks, paired development-only method selection requiring >=15% gain | Development scores and body-frame tests; new policy rollouts are required after changing collision geometry |
| 5.3 | RANSAC support plane; per-frame hand-gap, floating-motion and support-height diagnostics; smoothing with observed resting constraints | Actual time-aligned hands, resting labels and support samples; no nearest-hand attachment is guessed |
| 5.4 / 5.5 | Baseline and object-focused native reward profiles; native actor-only warm start; versioned jobs and checkpoint evaluation | Development evidence selects the profile/budget; separate fake orientation terms are not added to native YAML |

The object-focused profile doubles the existing object-keypoint weight and halves
the two hand-imitation weights in every Sharpa curriculum stage. Object keypoints
already depend on both position and orientation. Introducing a separate angular
objective would require a new tested native objective kernel; an unsupported
Hydra key is not emitted by this implementation.

Actor warm start is a new native `training.initial_policy_path` option. It copies
only actor parameters and batch statistics after validating policy I/O semantics.
It does **not** restore critic weights, optimizers, replay, RNG, reward normalizer,
exploration state or counters. It is mutually exclusive with learner resume.
The planning builder also requires a recorded same-tier warm-start rule decision.

## Input contracts

Every version uses an `execution.json` with source digests, configuration, code
commit and state. Changed inputs/configuration require a different version.

### Track 1 observation evidence

An NPZ supplied to `object-fit` contains:

- `parent_npz_sha256`, `parent_mesh_sha256` scalar strings;
- `coordinate_frame = "prediction_world_meters"`;
- `frames`: increasing original scored-frame indices, with no duplicates;
- `masks`: `[S,H,W]` nonempty observed silhouettes;
- `intrinsics`: `[3,3]` or `[S,3,3]` for that exact mask grid;
- `world_to_camera`: `[S,4,4]`, OpenCV positive-z camera convention;
- `hand_points`, `hand_normals`: `[S,P,3]`, all supplied hand vertices in the
  prediction world; normals must be unit length. Original and 2mm-dilated points
  are both checked. A supplied subset cannot prove the host's full-hand PEN.

The CPU silhouette renderer rejects near-plane crossings rather than silently
rasterizing invalid projections. Use the repository's established clipped GPU
renderer to prepare evidence for such cases.

Human evidence additionally includes `keypoint_joint_indices`, `keypoints_xy`,
`keypoint_visible`, and (for refit) `hand_joint_indices`. Missing 2D evidence cannot
pass the human promotion gate. `--body-asset` is an explicitly participant-owned
NPZ decoder export with the official `_MHRBody` field contract; host assets and
hidden role indices are not downloaded. `mhr_model.pt` is not this NPZ format.
If a compatible decoder export is unavailable, body diagnostics/refits remain
blocked on that input; root guesses or raw-Euler smoothing are not substituted.
A passing human fit is marked `READY_FOR_COUPLED_OBJECT_REVIEW`, not final-ready.
Regenerate the hand evidence and run the object fit after changing the human.

`assemble-track1 --reviews` takes a JSON dictionary with keys `"0"` through `"29"`.
An empty object retains the parent episode; `{"object": "/path/to/reviewed_fit"}`
promotes an accepted object result; add `"human": "/path/to/human_review"` if its
human changed. All promoted artifact hashes and input fingerprints are verified.

### Training jobs

`plan-jobs --specification spec.json --output <new-version>` reads:

```json
{
  "track": "track_2_tier1",
  "arm": "fresh",
  "target_steps": 20000768,
  "reward_profile": "baseline",
  "episodes": [
    {
      "episode_index": 3,
      "reference": "/absolute/path/to/verified/reference.parquet",
      "reference_sha256": "the actual SHA-256",
      "output_dir": "/home/alien51/video_to_data/robotic_grounding/data/track2/tier1/new_version/episode_000003"
    }
  ]
}
```

`resume` entries add a `checkpoint` pointing to `state_*.safetensors`. `warm-start`
entries point to an actor-only `policy_*.safetensors` and supply a `rules` JSON path.
Training totals must be divisible by 1,024, and a resume total must exceed its
saved step count. The generated `jobs.json` is consumed by `jobs --manifest ...
--report ...`; add `--execute` only when training is intended. Completed jobs reuse
verified actor checkpoints; interrupted jobs resume their own newest learner state.
A physics/VRAM failure stops for review rather than silently changing buffers.

### Tier preparation and frame conversion

`prepare-tier --tier N --dataset <verified-own-tier-root> --output
<own-data-root>/source_episodes` prepares the source files.

Retarget with the existing script and its new `--tier`, `--data-root`, `--reports`
options. Episode 16 is excluded by its legacy default, so for a new Tier 2/3 cohort
explicitly pass **all 30 episode indices including 16**.

`tier-inventory --data-root ... --tier N --output inventory.json` verifies every
prepared source, all 30 retarget outputs and their single-SE(3) motion relation.
It also writes hash-bound `frame_references/episode_*.npz` from the actual
conditioned/reconstructed source in the released world. The packer uses those
references for world conversion; this avoids comparing a conditioned trajectory
against an unconditioned one as if they differed by one rigid transform.

The generalized campaign supports:

```bash
reconstruction/.venv/bin/python robotic_grounding/scripts/run_track2_tier1_campaign.py \
  --tier 2 --version baseline_v1 --data-root /absolute/tier2/data/root \
  --inventory /absolute/tier2/inventory.json --training-steps 5000192
```

This command actually starts training when invoked. It is listed here for later
execution and has **not** been run as part of implementation. New-version upload
requires additional `--rules`, `--snapshots`, and `--upload` flags. The old default
Tier 1 directories remain for receipt recovery; new experiments must use a new
version. Tier 2/3 checkpoint decisions use their own native target as a proxy,
never clean Tier 1 scores. Official packing still performs its mandatory clean
reference comparison, which must not be fed back into tier-internal selection.

### Checkpoint selection

1. `checkpoint-inventory --policy-roots <policies_short> <baseline>
   --existing-results <results_short> --output candidates.json` lists every saved
   actor checkpoint, reusing raw rollouts only when its digest matches.
2. `checkpoint-select --candidates candidates.json --reference-root
   <released-Tier-1-root> --output <new-report-version>` computes available results;
   it refuses missing trajectories unless `--run-evaluations` is explicit.
3. `pack-selected --selection <selection.json> --output <new-package-version>
   --dataset-root <challenge-root> --commit <full-SHA>` exports complete selected
   rollouts and uses the official packer. No upload occurs.

The selector ranks `(SP-SR, MP-SR, AUC, -MPPE)` and rejects incomplete episode sets.
A native single-world deterministic evaluation is a checkpoint evaluation, not a
search over stochastic rollout seeds. Require a local AUC or SP-SR gain over the
preserved baseline before considering a new upload.

### Public development decomposition

`track3 dev-jobs --manifest dev.json --output <new-dev-version>` requires exactly
four public development episodes and `split: "development"`. Every episode has
`arms` `O`, `M`, `R`, `A`, each with a prepared native `reference` and its
`reference_sha256`. The supplied references must already contain the appropriate
mesh/reference/world combinations; this command cannot substitute a mesh filename
for the actual native collision geometry. The generated jobs use 10,000,384 steps.

`track3.dev_score` accepts matched `[T,1,B,7]` trajectories plus explicit public dev
mesh/scan tuples, anchors, canonical surface points and symmetry axes/orders. It
runs the official registration and symmetry functions. Missing anchors or scan
geometry are not invented. `track3.bottleneck` requires the same four episodes in
each arm. `select_mesh_methods` accepts only paired `split: development` records;
evaluation data cannot be used as its oracle.

### Rules and uploads

A rules JSON starts unresolved and contains:

```json
{
  "weekly_window": "unresolved",
  "same_package_all_metrics": true,
  "freeze_utc": "2026-11-04T22:00:00+00:00",
  "same_tier_cross_episode_warm_start_allowed": false
}
```

Set a weekly window only after its official interpretation is documented. A
calendar window additionally needs `weekly_window_source_quote` and
`week_start_utc`. Unlimited final-days behavior requires an explicit
`unlimited_source_quote` and `unlimited_start_utc`; it is not inferred from a
calendar date. Track 3 checks **all six** boards, including the separate mesh
competition. The daily Kaggle limit is still enforced by the official dry run.
A previous upload attempt without a receipt is treated as ambiguous. Existing
receipts can only be polled against their matching packet hash.

## Verification and remaining execution gates

Tests cover official NumPy/Torch synthetic decoder parity and finite-difference
gradients, constrained smoothing, scored gaps, anchor preservation, serialization,
full-world coverage, tier provenance, budget planning, warm-start semantics,
checkpoint selection and quota/receipt boundaries. Real baseline Tier 1 rollout
scoring is checked separately against the recorded public scores.

The implementation does not supply missing masks/cameras/full-hand observations,
participant decoder exports, Tier 3 reconstruction trajectories, public-development
oracle bundles, organizer rule answers, empirical pilot winners or final checkpoint
selections. Those are data and experiment gates. They must be obtained/measured on
this machine before the corresponding fits, training stages or uploads can pass.
Optional BundleSDF/SAM3D reconstruction ablations use the existing reconstruction
pipelines rather than a new untested reconstruction backend. No gains, final
packages, experiment winners or updated leaderboard positions are claimed here.

Baseline validation detail: the recomputed Tier 1 values were AUC
`0.24466419212519452`, SP-SR `0.1`, MP-SR `0.03333333333333333`, MPPE
`53.848209455446096 cm`. The receipt displays `0.24466`, `0.10000`, `0.03333`,
`53.84820`. Regenerated six-decimal pose coordinates match every coordinate of
all 22,989 submitted CSV rows exactly. MPPE differs from the displayed receipt
by `0.0000094554 cm`; the implementation records this precision limit and uses a
`1e-4` comparison tolerance rather than asserting identical displayed fifth digits.
The discrepancy's host-reference/display cause is not established by this check.

Track 3 also has a versioned native campaign entry point. Use
`run_track3_full_campaign.py --profile improvement --version <new-name>
--training-steps <explicit-budget> --parent-packet <preserved-submitted-packet>`
when a full new cohort is intended. Its data and reports use fresh version
directories. Optional `--support-evidence <JSON>` maps episode numbers to
`points_npz`, `points_sha256` and `development_gate_passed`. The NPZ contains
observed scene `points`, an explicit approximate `up` direction, and
`coordinate_frame = "reconstruction_world_meters"`. A passing plane fit supplies
one common transform to the loader for every object and both hands. It does not
independently align each object, flatten object bottoms to different planes, or
claim that estimated gravity has been validated against the hidden reference.
New-profile uploads use the same explicit rules/snapshot gate as the other tools.
