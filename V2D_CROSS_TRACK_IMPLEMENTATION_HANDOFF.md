# V2D cross-track implementation handoff to the next planning model

**Prepared:** October 20, 2026, America/Toronto.

**Repository:** https://github.com/JavierMedel/video_to_data

**Execution checkout:** `/home/alien51/video_to_data`

**Branch:** `v2d-track1-blackwell`

**Base commit before implementation publication:** `ba334d1bb2b75d4c225658a27a55938be15004a9`

**Purpose:** Give another LLM enough context to review the implemented improvement tooling and return the next concrete implementation and validation plan as a new Markdown file.

## 1. Instructions to the receiving model

You are the **planning model on a different laptop**. The challenge does **not** run on your laptop. The user will bring your resulting Markdown file back to the execution assistant on the original laptop.

Please produce **`V2D_NEXT_EXECUTION_AND_VALIDATION_PLAN.md`**. Write it for an implementation agent that can inspect and change this repository, access its local datasets and Docker images, and run experiments on the original GPU. Include concrete changes, required inputs, commands, validation gates, estimated resources and rollback conditions. Distinguish verified facts, hypotheses, proposed measurements and unresolved questions.

The prior plan, `V2D_CROSS_TRACK_IMPROVEMENT_PLAN.md`, has now received a substantial implementation. **Do not start by proposing to implement all those tools again.** Review what exists, identify defects or missing integration, resolve the data/validation gaps, and propose a measured route to better submissions. Existing code is not proof that an experiment improves a hidden-reference score.

This document is a standalone snapshot. Relative paths point into the execution checkout. If you only receive this Markdown file, you cannot claim to have inspected the code, local reports, data or weights. Mark any code-level conclusion that needs the actual files as a requested audit. The most useful companion is `CROSS_TRACK_IMPLEMENTATION.md` plus the current source changes.

### The two-laptop workflow

1. The user gives this document, and optionally source files, to you on the planning laptop.
2. You analyze the supplied evidence and write the new plan in Markdown.
3. The user copies your Markdown file back to `/home/alien51/video_to_data` on the execution laptop.
4. The execution assistant checks your assumptions against the actual checkout, corrects them where necessary, implements the next changes and runs the authorized validation.
5. Reconstruction, simulation, training, evaluation, packing and Kaggle submission happen on the execution laptop.

A Git clone on the planning laptop does not provide the execution laptop's generated artifacts, checkpoints, datasets, model weights or Docker runtime. Do not ask the user to move those workloads to the other laptop.

## 2. What changed, and what did not run

The most recent completed request was to **implement** the cross-track plan. That work added tested tooling for constrained Track 1 fitting, human refitting, tier-separated Track 2 campaigns, policy checkpoint selection, Track 3 development diagnosis, mesh handling and guarded submission.

**The implementation did not start a new challenge training campaign, reconstruct a new candidate, download new challenge datasets, generate a new final submission packet, refresh all leaderboards or upload anything.** Previously scored submissions remain the baselines. Read-only scoring of existing Track 2 Tier 1 rollouts was performed.

At completion of this implementation, there were no remaining baseline GPU jobs. This is a status snapshot, not a guarantee about processes when a future plan is executed; inspect the machine before scheduling work.

### Important Git state

At the initial drafting snapshot, the new implementation was local and uncommitted. The base commit above is the earlier `docs: refresh cross-track challenge status` commit and does not contain it. **Publication update:** the user has now requested committing and pushing the implementation, this handoff, the runbook and `V2D_CROSS_TRACK_IMPROVEMENT_PLAN.md` together on `v2d-track1-blackwell`. Use the publication commit containing this file, rather than the base commit, for source review. On the execution checkout, `git log -1 --format=%H -- V2D_CROSS_TRACK_IMPLEMENTATION_HANDOFF.md` identifies that commit; compare the remote branch before claiming publication is complete.

New files include `CROSS_TRACK_IMPLEMENTATION.md`, `reconstruction/scripts/cross_track/`, the cross-track CLI, metric harness and tests. Existing campaign and native training files are modified. Other historical generated files are also untracked; do not stage the entire checkout indiscriminately.

The next execution plan should first inventory the current working tree and preserve it. Publication does not include large generated artifacts or unrelated historical logs. Before producing a packet after further changes, commit the actual producing code as required by the packing path. Do not claim a later working-tree change was pushed unless the execution assistant verifies the remote commit.

## 3. Execution laptop and practical resource constraints

| Item | Execution environment |
| --- | --- |
| OS/runtime | Linux under WSL2 on Windows; Bash; Docker Desktop integration |
| Checkout | `/home/alien51/video_to_data` |
| GPU | One NVIDIA RTX 5090 Laptop GPU, Blackwell `sm_120` |
| Device memory | Approximately 24,463 MiB total |
| GPU guard | 21,000 MiB device-wide memory cap; retain headroom |
| Host memory | Approximately 31 GiB; earlier inventory showed approximately 8 GiB swap |
| CPU | Approximately 24 logical CPUs; recheck current allocation |
| Host Python | `reconstruction/.venv/bin/python` |
| Host packages | NumPy, SciPy, pandas, pyarrow, trimesh, pytest, Kaggle; no host Torch |
| Native policy image | `flash_chord:track2-baseline`; JAX/FlashSAC, not Torch |
| Torch image | `v2d_cari4d:blackwell` |
| Other relevant images | `v2d_track2_retarget:baseline`, `v2d_track3_loader:blackwell` |
| Shared heavy-GPU lock | `reconstruction/data/track3/pilot/episode_000038/pilot.lock` |
| Persistent native JAX cache | `/cache/jax` in the policy runtime |

The user wants high GPU utilization, but this laptop has already experienced crashes, stalls and out-of-memory messages. Schedule **one heavy GPU workload at a time** through the existing lock/guard. CPU work can proceed alongside it when memory permits. Do not raise the global cap merely to make the allocation number larger.

Allocated VRAM is not GPU compute utilization. Earlier heartbeats reported roughly 12.3–12.7 GiB for successful policy runs; utilization and power were unavailable from that source. Measure throughput and real utilization with available supported telemetry. Repeated JAX 4 GiB allocation warnings did not always imply a failed job: later compile, checkpoint and rollout evidence determines the actual outcome.

Fresh native policy runs commonly use 1,024 worlds, replay capacity 1,000,000, minimum replay 102,400 and batch size 1,024. Resume must preserve the saved world count, which can be 512 or 1,024. Use the native wrapper's existing timing, source-frame and physics-buffer settings; do not substitute generic Isaac configuration guesses.

`run_track3_policy_stage.py` supports Track 3 and all three Track 2 tiers. Track 2 containers mount `robotic_grounding` as `/repo`; Track 3 mounts the monorepo as `/repo`. A valid host path must be mapped for the actual container mode. Reports and outputs must remain under the corresponding mounted tree.

Old GPU-hour forecasts are not measurements for the new workloads. Re-measure compile cost, steady-state steps/second, evaluation time, peak device memory and host memory on a small pilot before forecasting a full campaign.

## 4. Confirmed submission baselines

These scores were already returned by Kaggle before the new implementation. No improvement is claimed from the new tooling yet. A best-per-metric team row can combine different submissions; it is not necessarily the score vector of one coherent packet.

### 4.1 Track 1: latest confirmed complete submitted packet is v8

Coverage: **30 episodes, 740,780 packed rows**. All five metrics are lower-is-better, in the reported centimeter units.

| Metric | v8 public score | Historical leader score |
| --- | ---: | ---: |
| CD-H | 15.43955 | 13.43306 |
| CD-O | 23.48324 | 17.91571 |
| ACC-H | 1.15894 | 0.13738 |
| ACC-O | 0.71908 | 0.07111 |
| PEN | 0.00322 | 0.00000 |

**Leaderboard timestamp:** `2026-10-16T21:04:17.392752+00:00`. These comparisons are stale. They were not refreshed for this handoff. Historical CARI4D baseline CD-O was 23.16246; our pipeline is CARI4D-based, not a verified reproduction of every organizer production input.

- v8 packet SHA-256: `bd73a931e9f9ec85b920884415c5dc2123d86e2d3d91395c7820ebf564fd3198`.
- Parent episode files: `reconstruction/data/v2d_challenge/track_1/contact_smoothing_v8/submission_episodes/`.
- Detailed history: `CHALLENGE_STATUS_AND_LLM_HANDOFF.md`.
- v6: CD-O 23.48489, ACC-O 0.81013, PEN 0.00325.
- v8 improved ACC-O about 11.24% versus v6, with only about 0.007% CD-O change.
- v7 ray-shift/smoothing worsened CD-O to 23.61430 despite temporal improvement. Do not repeat unsupported MoGe depth-ray shifts.
- v9 raw-Euler human smoothing was rejected: approximately 115–174 cm p99 deformation. Do not repeat it.
- v10 was a prepared Episode 18 rotation-only candidate, held rather than confirmed submitted. Its readiness/quota/packet lineage must be re-audited before any proposed upload. Do not treat it as the new scored parent.
- Historical best displayed team CD-H was 14.99887 from v2; best PEN was 0.00213 from v1. Neither is v8's score for that metric.

The broad Track 1 submission history is retained in the earlier handoff. This document focuses on the current implementation rather than repeating every historical experiment.

### 4.2 Track 2 Tier 1: complete scored short-budget baseline

Coverage: **30/30 episodes**, 22,989 submitted rows, 11,838,359-byte CSV. Episode 16 reused a genuine **349,999,104-step** policy; the other 29 policies each received **5,000,192 steps**. This is not 350M training for every episode.

| Metric | Direction | Public score | Submission ID | State |
| --- | --- | ---: | ---: | --- |
| AUC | Higher | 0.24466 | 57444008 | COMPLETE |
| SP-SR | Higher | 0.10000 | 57444011 | COMPLETE |
| MP-SR | Higher | 0.03333 | 57444013 | COMPLETE |
| MPPE, cm | Lower | 53.84820 | 57444015 | COMPLETE |

- Packet: `robotic_grounding/data/track2/tier1/submission_short/track2_tier1.csv`.
- SHA-256: `73f60d4a35996c7f9d6763fc28a637fec542248f538c98b4929af33bdc372a23`.
- Producing code commit: `fa969859b4e9156321ceee6fc1723ea49ce4f1d1`.
- Raw rollouts: `robotic_grounding/data/track2/tier1/results_short/`.
- Export-only body-name mapping outputs: `robotic_grounding/data/track2/tier1/results_short_submission/`.
- Policies: `robotic_grounding/data/track2/tier1/policies_short/`; the longer pilot has its separate existing directory.
- Campaign records: `robotic_grounding/incident_reports/track2_baseline/short_campaign/`.
- Key records: `execution.json`, `kaggle_receipt.json`, `upload_manifest.json`, per-episode `policy.json` and native logs.
- Broader status: `TRACK2_STATUS.md`.

All four submissions were COMPLETE by approximately 11:34 Toronto time on October 20. A fresh current rank/leader comparison has not been obtained in this implementation. Request a read-only authenticated snapshot rather than inventing it.

**Track 2 Tier 2 and Tier 3 have no completed new campaigns or submissions in this snapshot.** Supporting code exists; that is not a baseline result. Tier 3 still needs actual reconstruction-derived training inputs.

### 4.3 Track 3: complete scored short-budget baseline

Coverage: **20 episodes, 36 object instances, 5,181 source frames and 9,120 pose rows**, one evaluation world. Episode 38 reused **60,002,304 steps**; the other 19 policies received **5,000,192 steps** each. The native full target of 250,003,456 steps per episode was not completed.

| Metric | Direction | Our public score | Historical leader | Display position | Submission ID |
| --- | --- | ---: | ---: | --- | ---: |
| AUC | Higher | 0.16307 | 0.82309 | 17/17 | 57416110 |
| SP-SR | Higher | 0.05000 | 0.85000 | 16/17 | 57416113 |
| MP-SR | Higher | 0.00000 | 0.70000 | 17/17 | 57416115 |
| RPE, cm | Lower | 22.28685 | 3.05138 | 17/18 | 57416118 |
| MPPE, cm | Lower | 20.82175 | 2.62042 | 17/17 | 57416120 |
| CD-O, cm | Lower | 0.57037 | 0.06523 | 17/18 | 57416123 |

All six returned **COMPLETE**. Snapshot time: `2026-10-20T01:14:03.178868+00:00`. Display positions are API row positions, not an assertion about tie handling or combined ranking. They are not live standings.

- Packet: `robotic_grounding/data/track3/submission_today/track3_submission.parquet`, 3,567,453 bytes.
- SHA-256: `19c1de67171660febec90b943f9826da599f6a69125f5869e391d2e4fe8913f8`.
- Producing code commit: `4f25220b0172f830567b908dd52003d5ec516111`.
- Raw rollouts / exports: `robotic_grounding/data/track3/results_today/` and `results_today_submission/`.
- Policies / meshes: `robotic_grounding/data/track3/policies_today/` and `submission_meshes/`.
- Evidence: `reconstruction/incident_reports/track3_baseline/today_campaign/`.
- Records include `execution.json`, `kaggle_receipt.json`, `upload_manifest.json`, `completed_rollout_audit.json`, `submission_name_mapping.json`, `mesh_preflight.json`, `leaderboard_snapshot.json`.
- Broader status: `TRACK3_STATUS.md`.

All 20 local native reference-tracking success rates were zero. That criterion differs from official object-only, symmetry-aware SP-SR, so it does not contradict public SP-SR 0.05. Valid rollouts and successful packing prove neither good manipulation nor collision convergence. Some baseline native logs contained EPA warnings.

Correct Track 3 leaderboard URLs have **no `tier1` segment**, and the shape competition ends in **`cd-o`**:

- https://www.kaggle.com/competitions/v2d-challenge-track3-auc/leaderboard
- https://www.kaggle.com/competitions/v2d-challenge-track3-sp-sr/leaderboard
- https://www.kaggle.com/competitions/v2d-challenge-track3-mp-sr/leaderboard
- https://www.kaggle.com/competitions/v2d-challenge-track3-rpe/leaderboard
- https://www.kaggle.com/competitions/v2d-challenge-track3-mppe/leaderboard
- https://www.kaggle.com/competitions/v2d-challenge-track3-cd-o/leaderboard

## 5. Corrections to the original plan that must carry forward

1. **Track 2 does not enable initial-pose alignment in the configured scorer.** A helper named `_warp_to_reference_start` exists, but its existence does not establish that the scorer uses it. The rigid conversion back to the released world is required and does not erase initial-placement error. Track 3's alignment behavior is separate.
2. **Track 2 G1 has nine curriculum stages; Track 3 Sharpa has eleven.** Read actual YAML. A continuation of a 5M learner retains its original thresholds and saved world count; do not silently restart assistance by rescaling its curriculum.
3. **Track 2 Tier 3 has `input_path: null`.** It needs real reconstructions from its own videos. Do not copy clean Tier 1 inputs or reuse Track 1 outputs just because episode numbers coincide.
4. **A self-reference PEN score need not be zero.** PEN tests penetration of the predicted hand against the predicted object; it is not simply distance from a reference artifact.
5. For independent Gaussian position noise in three axes, expected acceleration **norm** is `sqrt(6) * sigma * 2 * sqrt(2/pi)`, before accounting for the official endpoint weighting. `sqrt(6)*sigma` alone is not that mean norm.
6. Mesh recentering can preserve world surfaces analytically. Verify neutrality before and after serialization; export rounding, triangulation and surface sampling can matter. Never infer packed neutrality from algebra alone.
7. Zero penetration on supplied points proves only what those points cover. A guaranteed superset of the host hand points is needed for a host-subset guarantee. The implementation checks original points plus dilated points, not a replacement set that omits the originals.
8. Leaderboard ties do not reveal hidden-reference acceleration or establish an attainable score floor. Own acceleration magnitudes are proxies, not official acceleration-error scores. Root/articulation norm summaries are not additive energies.
9. Track 2 Tier 1 local recomputation agrees within `1e-4`; the printed MPPE fifth digit is not perfectly reproduced. Do not alter references to force a display match.
10. The weekly quota boundary, final-days exception, final-selection rules and cross-episode warm-start permission require documented official interpretations. Do not assume Monday reset, automatic final-days unlimited uploads or warm-start permission.
11. The new object-focused reward profile changes existing object-keypoint and hand weights. It does not add a separately implemented angular reward kernel. An unsupported Hydra key is not an objective implementation.
12. The original 290-GPU-hour schedule and score-gain estimates are proposals, not demonstrated availability or results. Re-plan from measured bottlenecks and remaining time.

The original plan reports a freeze of **2026-11-04 22:00 UTC** (17:00 Toronto after the DST transition). This has not been freshly verified for the handoff. Verify official dates and limits before scheduling uploads; do not build a quota decision from this document alone.

## 6. Implemented source map

All paths below are relative to the execution repository. Read `CROSS_TRACK_IMPLEMENTATION.md` for the operational runbook.

### 6.1 Entry points and shared state

`reconstruction/scripts/run_cross_track_improvement.py` exposes:

`object-fit`, `human-fit`, `assemble-track1`, `pack-selected`, `diagnostics`, `checkpoint-inventory`, `checkpoint-select`, `prepare-tier`, `tier-inventory`, `plan-jobs`, `track3`, `jobs`, `submission`.

- Every command has `--help`.
- `jobs` validates by default; `--execute` starts training.
- `submission submit` checks gates by default; `--execute` uploads.
- `checkpoint-select` needs `--run-evaluations` to create missing rollouts.
- Fitting commands actually process their supplied inputs. Human fitting defaults to CPU; CUDA is explicit.

The package is `reconstruction/scripts/cross_track/`:

| Module | Implemented responsibility | Important limit |
| --- | --- | --- |
| `common.py` | Strict/atomic JSON, hashes, version contracts, input/config/code fingerprints | Changed inputs or code require a new version; a commit label alone does not capture uncommitted bytes |
| `temporal.py` | Contiguous scored stretches, official-style endpoint acceleration, sparse constrained QP, robust weighting, SO(3) fitting | Does not establish hidden-reference improvement |
| `image.py` | Calibrated CPU triangle silhouettes and mask IoU | Rejects near-plane crossings; use established clipped rendering where needed |
| `track1.py` | Mesh recentering, contact/penetration constraints, lambda selection, serialized review | Candidate output still requires packed validation |
| `mhr_torch.py` | Float64 differentiable official-style FK/correctives/LBS decoder | Synthetic parity is not real-asset parity |
| `human.py` | Root decomposition, constrained root fit, joint-space refit, image/body/hand gates | Needs participant decoder export and observed keypoints; human changes require renewed object review |
| `gpu.py` | Shared GPU lock, memory monitor and heartbeat | New CUDA fitting path has not received a real GPU stress run |
| `policy.py` | Checkpoint inventory, native deterministic evaluation, scoring, ranking, job builders and pilot/budget decisions | Several scheduling helpers need a measured controller decision; no complete autonomous halving campaign has run |
| `tier_inputs.py` | Tier provenance, source preparation, conditioning, complete retarget inventory and frame references | Consumes genuine Tier 3 reconstructions; does not generate them |
| `track3.py` | Support/hand diagnostics, resting constraints, mesh cleanup/frame checks, public-dev scoring and matched jobs | Public-development oracle inputs must be prepared correctly first |
| `jobs.py` | Typed serial native-job execution, version records and interrupted-state resume | No blind multi-day global optimizer or calendar scheduler |
| `packaging.py` | Full-cohort review/lineage checks and official packing | Track 1 assembly is the supported v11-style path, not arbitrary mesh/rotation promotion |
| `submission.py` | All 23 configured competitions, snapshots, quota/packet/receipt gates | Requires fresh histories and documented rules; no new upload occurred |

`reconstruction/scripts/track1_metric_harness.py` imports official metric functions and checks geometry/packing behavior using an explicit participant decoder asset. Official metric source remains under `reconstruction/v2d_submission_kit/v2d_submission_kit/metric_code/`.

### 6.2 Track 1 implementation details

- Object translation uses robust constrained smoothing over contiguous scored runs. The first scored anchor and unscored frames are preserved.
- Candidate lambdas are 3, 10, 30, 100 and 300, selecting the strongest passing result.
- Full supplied hand points, their 2 mm dilation, winding containment and triangle distances generate constraints. Up to five constraint updates are followed by a minimal correction/re-fit where needed.
- Contact targets use nearest surface hand points within contact windows, rather than an average position between both hands.
- Review gates include mean translation change at most 5 mm, p99 at most 3 cm, mean silhouette-IoU drop at most 0.01, worst drop at most 0.05, median contact change at most 1 cm, no penetration on checked hand points and no acceleration increase.
- Serialized GLB/NPZ artifacts are rechecked and hash-bound. State `CANDIDATE_REQUIRES_PACKED_VALIDATION` means exactly that, not submission-ready.
- SO(3) angular-acceleration fitting exists; unsupported symmetry flips are not automatically enabled.
- Component ablation helpers compare full/largest/mask-supported geometry. Their all-episode image gate is not an automatically executed v12 geometry campaign.
- Human fitting uses root translation, SO(3) root rotation and differentiable joint-space refit with identity fixed. It preserves the first scored anchor and unscored parameters.
- Human review constrains body p99 displacement to 3 cm, hand p99 to 1.5 cm and mean 2D error to at most 1.05 times baseline; refit also limits maximum vertex movement to 5 cm and requires nonincreasing joint acceleration.
- Human success state is `READY_FOR_COUPLED_OBJECT_REVIEW`. Regenerate hands and rerun object fitting. Alternating human/object optimization is not automatically completed twice.
- Complete Track 1 assembly expects all 30 episode entries and, by default, at least 25 reviewed promotions; rejected parents are retained byte-for-byte. Verify whether this minimum is appropriate for a particular measured candidate rather than fabricating passes.

### 6.3 Native campaigns and warm start

Modified existing files:

- `reconstruction/scripts/run_track3_policy_stage.py`: all three Track 2 tier modes and correct container mount selection.
- `reconstruction/scripts/run_track3_full_campaign.py`: isolated `--profile improvement --version ...`, explicit budget and parent packet; optional hash-bound observed support evidence; new-profile upload gates.
- `robotic_grounding/scripts/prepare_track2_sources.py`: tier/source lineage and Tier 3 reconstruction manifest checks.
- `robotic_grounding/scripts/retarget_track2_tier1.py`: generalized tier/data/report roots and verified own-source handling.
- `robotic_grounding/scripts/run_track2_tier1_campaign.py`: tier/version/inventory support, isolated outputs and guarded uploads.
- `robotic_grounding/flash_chord/src/flash_chord/training/flash_sac/config.py`, `runner.py`, and `configs/training/reference_flash_sac.yaml`: actor-only warm start.

`training.initial_policy_path` loads actor parameters and batch statistics after I/O schema validation. It creates fresh critics, optimizers, replay, RNG, reward normalization and counters. It is mutually exclusive with full learner resume. Builders require a recorded warm-start rule decision.

Checkpoint selection uses `(SP-SR, MP-SR, AUC, -MPPE)` lexicographic order, deterministic one-world rollouts, checkpoint digests and full 30-episode coverage. Tier 1 uses released clean references and canonical scoring/serialization. Tier 2/3 internal decisions use their own inputs as labelled proxies; clean Tier 1 data must not feed back into their selection.

The legacy retarget script's default excludes Episode 16. **For a new Tier 2 or Tier 3 cohort explicitly include all 30 episodes, including 16.** Only the original Tier 1 path can reuse its existing genuine pilot.

### 6.4 Track 3 implementation details

- RANSAC support-plane fitting and hand-gap/floating/support-height diagnostics consume observed, time-aligned evidence.
- Support correction applies one common transform to all objects and both hands. It does not flatten each object independently.
- Resting constraints need actual resting labels. No guessed nearest-hand attachment is introduced.
- Mesh cleanup includes largest component, hole filling and Taubin smoothing, with candidate-to-parent rigid registration.
- Body-frame checks include radius ratio 0.95–1.05 and bidirectional median geometry distance at most 5 mm.
- Public-development helpers use official mesh registration and symmetry-aware metrics with supplied scans, anchors, canonical samples and symmetry metadata.
- O/M/R/A matched development job generation requires four supplied development episodes and prepared native references; it cannot build correct oracle/collision geometry by renaming a file.
- Mesh-method selection uses paired development records and requires at least 15% gain. Evaluation scans are not permitted substitutes.
- Object-focused Sharpa training doubles the existing object-keypoint weight (20 to 40) and halves both hand-imitation weights (0.25 to 0.125) throughout the actual curriculum.

Public-dev input construction, two-single/two-multi coverage where required by the experimental design, native collision/reference consistency and split provenance still need explicit verification. A manifest field saying `development` alone is not independent proof.

## 7. Inputs and integration gaps that block real experiments

### 7.1 Track 1 observation evidence

Object-fit NPZ input requires:

| Field | Contract |
| --- | --- |
| `parent_npz_sha256`, `parent_mesh_sha256` | Scalar hashes binding the evidence to the actual parent |
| `coordinate_frame` | `prediction_world_meters` |
| `frames` | All original scored-frame indices, increasing and unique |
| `masks` | `[S,H,W]` observed silhouettes, nonempty |
| `intrinsics` | `[3,3]` or `[S,3,3]`, calibrated for the exact mask grid |
| `world_to_camera` | `[S,4,4]`, OpenCV positive-z convention |
| `hand_points`, `hand_normals` | `[S,P,3]`, full supplied hand vertices in the prediction world; finite, unit normals |

Missing frames, guessed camera conventions or partial hand coverage cannot be represented as full observed validation. Human evidence additionally needs `keypoint_joint_indices`, `keypoints_xy`, `keypoint_visible` and, for refitting, `hand_joint_indices`.

The next plan should identify existing local sources for each field and specify a reproducible export/verification adapter. The implementation has not produced complete evidence bundles for all 30 episodes.

### 7.2 Participant MHR decoder export

`--body-asset` expects a **participant-owned NPZ decoder export with the official `_MHRBody` field contract**. An existing `mhr_model.pt` file is not automatically compatible.

The next plan must inspect lawful locally available model assets and propose an exact export adapter or compatible decoder integration. Do not invent host role indices or use inaccessible hidden assets. Before fitting real humans, compare the Torch port to the official NumPy decoder on **at least 100 real frames**, with maximum error at most `1e-5 m`, and verify meaningful gradients. Synthetic success alone is insufficient.

Until a compatible asset and real-frame parity exist, human diagnostics/refits remain blocked by input, not solved by raw-root guesses or Euler smoothing.

### 7.3 Track 2 Tier 2/3 datasets and retargeting

Tier 2 needs its actual released noisy input. Tier 3 needs its own video-derived reconstruction producing compatible human/object trajectories. Reconstruction manifests bind source-video hashes, output hashes and `coordinate_frame = released_track2_opencv_world`.

New preparation code consumes this contract; it does not create a genuine Tier 3 reconstruction. The next plan should specify which existing reconstruction path can generate it, how scale/world coordinates are established, how its human representation is converted, and how native retarget/contact validity will be verified.

Position/SO(3) conditioning is not a guarantee of kinematic or contact feasibility. Require all 30 source and retarget records and a verified single-SE(3) relation. The inventory exports hash-bound frame references from the actual conditioned source for packing.

### 7.4 Public Track 3 development bundles

Need genuine public-development scans, reference trajectories, participant reconstructions, anchors, symmetries, canonical samples and native collision assets. Prepare matched O/M/R/A combinations with the same episode/frame/world conventions and budgets.

Do not substitute evaluation geometry for a development oracle or treat development success as direct hidden-reference evaluation success. Optional BundleSDF/SAM3D method experiments use existing reconstruction backends; no new backend was implemented here.

### 7.5 Scheduling and promotion integration

Pilot-arm comparison, near-success ranking, successive-halving selection and collapse/retry helpers exist. They are not yet a complete autonomous campaign controller with proven pilot winners. Specify the missing orchestration or explicit decision steps rather than claiming it already ran.

Similarly, mesh ablation and SO(3) helpers are not a generic end-to-end Track 1 v12 promotion pipeline. Audit which review/packing contracts accept each proposed change before recommending it for all episodes.

## 8. Verification already completed

**90 tests passed across three distinct scopes**, plus CLI import/help checks, Python compilation and `git diff --check`. These are implementation checks, not 90 successful challenge experiments.

### 8.1 Host checks: 87 passed

```bash
cd /home/alien51/video_to_data
reconstruction/.venv/bin/python -m pytest -q \
  reconstruction/scripts/test_cross_track_improvement.py \
  reconstruction/scripts/test_track3_campaign.py \
  robotic_grounding/tests/test_track2_campaign.py \
  robotic_grounding/tests/test_track2_sources.py \
  robotic_grounding/tests/test_track2_retarget_verification.py \
  reconstruction/scripts/test_track3_policy_guard.py
```

Includes constrained smoothing, scored gaps and anchors, serialization, world/frame handling, tier provenance, checkpoint selection and mocked quota/receipt boundaries. All 13 cross-track CLI help entry points imported successfully.

### 8.2 Torch decoder/gradient checks: 2 passed, CPU Docker

```bash
docker run --rm --name v2d-cross-track-mhr-tests --cpus 4 --memory 8g \
  -e PYTHONPATH=/repo/reconstruction/scripts \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -v /home/alien51/video_to_data:/repo:ro -w /repo \
  --entrypoint python v2d_cari4d:blackwell \
  -m pytest -q -p no:cacheprovider \
  reconstruction/scripts/test_cross_track_mhr_torch.py
```

Synthetic NumPy/Torch maximum error was below `1e-12 m`; finite-difference gradient checks passed. **No real 100-frame body-asset parity run has occurred.**

### 8.3 Native actor warm-start check: 1 passed, 17 deselected

```bash
docker run --rm --name v2d-cross-track-native-tests --cpus 4 --memory 8g \
  -e PYTHONPATH=/repo/robotic_grounding/flash_chord/src \
  -e JAX_PLATFORMS=cpu \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -v /home/alien51/video_to_data:/repo:ro -w /repo \
  --entrypoint python flash_chord:track2-baseline \
  -m pytest -q -p no:cacheprovider \
  robotic_grounding/flash_chord/tests/training/flash_sac/test_flash_sac_runner.py \
  -k actor_warm_start
```

This used a fake native environment for four CPU steps. It checked actor loading, fresh counters, incompatible policy rejection and resume/warm-start mutual exclusion. It was not challenge GPU training.

### 8.4 Existing Track 2 Tier 1 baseline recomputation

Report: `reconstruction/incident_reports/cross_track_implementation/tier1_baseline_validation.json`.

| Metric | Recomputed from existing rollouts | Receipt display |
| --- | ---: | ---: |
| AUC | 0.24466419212519452 | 0.24466 |
| SP-SR | 0.1 | 0.10000 |
| MP-SR | 0.03333333333333333 | 0.03333 |
| MPPE, cm | 53.848209455446096 | 53.84820 |

All were within `1e-4`. Regenerated six-decimal pose coordinates matched every coordinate in **all 22,989 submitted CSV rows exactly**, with maximum coordinate difference zero.

MPPE differs from the displayed receipt by approximately `0.0000094554 cm`. Its host-reference/display cause was not established. Preserve that limitation. Several baseline per-episode evaluator-JSON AUC values differed slightly from canonical trajectory-derived recomputation; official packing scores the trajectory. Do not select checkpoints from inconsistent JSON summaries without checking the actual rollout.

### 8.5 Checks not completed

- Real participant-asset decoder parity on 100 or more frames.
- New CUDA human-fit stress/guard run.
- Full observation evidence export for all Track 1 episodes.
- New coupled human/object candidate or final packed neutrality check.
- New Tier 2/3 retargeted cohort, training or submission.
- Real public-development O/M/R/A construction and measured comparison.
- New native reward-profile pilot or selected improved checkpoint packet.
- New authenticated 23-board snapshot or real upload exercise of the new guards.

Do not infer these from the passing unit tests.

## 9. Commands and execution semantics for the next plan

The safe entry-point discovery command is:

```bash
cd /home/alien51/video_to_data
reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py --help
```

Use subcommand help and inspect the actual parser before giving exact flags. Avoid fabricated filenames or schemas. Useful existing flows are:

1. `checkpoint-inventory` over all preserved actor checkpoints and existing rollouts.
2. `checkpoint-select` with Tier 1 released references; missing native evaluations require the explicit evaluation flag.
3. `pack-selected` after a complete measured improvement, with the actual producing commit.
4. `prepare-tier`, native retarget and `tier-inventory` after obtaining genuine own-tier inputs.
5. `plan-jobs`, then `jobs` validation, then `jobs --execute` only for an intended measured experiment.
6. `track3 dev-jobs` after preparing and validating the four matched development bundles.
7. `submission snapshot` for fresh authenticated leaderboard/history evidence, and guarded submission only after rules and packet gates pass.

For job specifications, target step totals must be divisible by 1,024. Resume totals must exceed the saved step count; full state and actor-only checkpoints serve different purposes. Prefer explicit absolute paths and new version directories. `CROSS_TRACK_IMPLEMENTATION.md` contains a typed specification example and preparation/selection usage.

The generalized Track 2 campaign and Track 3 full campaign are **execution commands** when invoked. Do not run them just to discover their behavior. Read code or use `--help` first. The new Track 3 profile takes `--profile improvement`, a new `--version`, `--training-steps` and `--parent-packet`; optional support evidence has its own hash/frame/development gates.

## 10. Submission and provenance rules for the next executor

The account is `iorideepp`. Competition entry/rule acceptance blockers previously discussed with the user were cleared. Do not ask the user to reaccept the same competitions merely because an old report says unjoined. Distinguish account entry from unresolved interpretations of quota and allowed method choices.

- Preserve submitted bytes, parent episodes, policy checkpoints and exact receipts.
- New work goes in new versioned directories with input, configuration, code and parent hashes.
- Commit producing code before official packing where required; record full commit SHA.
- A changed code/config/input fingerprint requires a new experiment version.
- Keep all episodes and distinct genuine policy provenance; no policy rollout fabrication or label-only claims of training.
- Use coherent packets across a track/tier's metrics unless documented rules explicitly permit another approach.
- Track 3 needs all six competitions, including the separate mesh packet.
- Histories must be fresh (the new gate uses a 24-hour maximum age), paginated and complete enough for the relevant window.
- Count pending and failed upload attempts when the gate requires it, not only scored submissions.
- Weekly boundaries and any unlimited period need documented source text; default is unresolved.
- An attempted upload with no receipt is ambiguous and must not be blindly repeated.
- Existing receipts may be polled only with the matching packet hash and exact returned IDs. Polling is not a reason to upload again.
- A successful official dry run does not settle a separate organizer weekly limit or prove leaderboard improvement.

The new guard covers 23 configured competitions: five Track 1, four per Track 2 tier, and six Track 3. The checked-in/runbook rules example is intentionally unresolved, not upload-ready.

Do not contact organizers or send messages on the user's behalf without authorization. Plans can list a precise unresolved question and the official source needed to settle it.

## 11. What the next returned Markdown should contain

Please make the next plan executable and evidence-driven. Include these sections or an equivalent clear structure:

1. **Current-state audit.** Identify what you could inspect and what comes only from this snapshot. Separate real submissions, passing code tests, unrun experiments and missing artifacts. Call out any defects found in the implementation.
2. **Corrected priorities.** Rank work by measured or plausibly testable benefit per resource cost. Preserve existing baselines. Do not repeat the original predicted score gains as facts.
3. **Input acquisition/export tasks.** Give exact local searches, source files and adapter changes for masks/cameras/hands, participant MHR export, tier datasets, Tier 3 reconstructions and public-development bundles. Include units, frame conventions, shapes, digests and split checks.
4. **Implementation changes.** For each change, name files/functions, the current limitation, proposed behavior, integration points and compatibility risks. State when existing tools can simply be reused.
5. **Small validation experiments.** Start with CPU/read-only checks and the smallest meaningful real-data pilots. Define baseline comparison, primary metric, feasibility checks, required observations and minimum gain before expansion.
6. **Measured scheduling.** Budget the single GPU, host RAM and disk; include compile/evaluation cost, checkpoint intervals and heartbeat review. Re-estimate full runs from actual pilot throughput. Define how interruption/resume works.
7. **Explicit gates.** Give pass/fail criteria, stop conditions, rollback, provenance checks and what remains blocked if an input is absent. Include real MHR parity before human refit and native retarget validity before training.
8. **Selection and packing.** Specify complete episode coverage, deterministic checkpoint evaluation, accepted review inputs and serialized-artifact validation. Keep Tier 2/3 clean-data isolation and Track 3 development/evaluation separation.
9. **Submission readiness.** Require fresh standings/history, documented limits, committed producing code, official dry run and exclusive receipt recovery. Do not schedule an automatic duplicate baseline upload.
10. **Exact first steps on this laptop.** Give a short ordered set of commands the execution assistant can start with, and make clear which inspect, modify, train, evaluate, pack or upload. Use real parsers; label placeholders that must be resolved locally.

If additional information is required, describe a local inspection or extraction task the execution assistant can perform. Prefer a plan that makes useful progress with the available artifacts while explicitly leaving dependent work gated. Do not claim hypothetical measurements, invented organizer answers or access to hidden evaluation references.

## 12. Supporting files to request if needed

The user can provide these to the planning model if code-level review is needed. Large generated datasets are not necessary for an initial source audit.

- This document: `V2D_CROSS_TRACK_IMPLEMENTATION_HANDOFF.md`.
- Original proposal: `V2D_CROSS_TRACK_IMPROVEMENT_PLAN.md`.
- Implementation runbook: `CROSS_TRACK_IMPLEMENTATION.md`.
- Historical context: `CHALLENGE_STATUS_AND_LLM_HANDOFF.md`, `TRACK2_STATUS.md`, `TRACK3_STATUS.md`.
- New source: `reconstruction/scripts/cross_track/`, `run_cross_track_improvement.py`, `track1_metric_harness.py`.
- New tests: `reconstruction/scripts/test_cross_track_improvement.py`, `test_cross_track_mhr_torch.py`.
- Modified campaign/retarget/native training files listed in Section 6.
- Small evidence: `reconstruction/incident_reports/cross_track_implementation/tier1_baseline_validation.json` and the relevant existing campaign receipts/manifests.

Check the actual file availability on the receiving machine. The original base commit lacks the implementation; obtain the publication commit described in Section 2. Even after the source is pushed, GitHub does not provide the execution laptop's large local data, weights or checkpoints. The requested deliverable from the planning model is the new Markdown plan; execution returns to the original laptop.
