# Track 3 execution campaign and planning handoff

This file describes the Track 3 campaign on the execution laptop at
`/home/alien51/video_to_data`. It is intended for the user and a second LLM
planning improvements on a different laptop. **That other laptop does not run
this challenge.** It should produce a Markdown improvement plan for the user
to bring back here. Execution, Docker builds, model provisioning, reconstruction,
simulation, policy training, evaluation and submission happen on this laptop.
Every future plan should distinguish observed artifacts from proposed work and
state which local files and validations its implementation will require.

## Scope and actual progress

Track 3 was started at the user's request on October 17, 2026. The user also
requested using the available GPU capacity. This campaign is independent of
Track 1 and Track 2; matching episode numbers never imply matching recordings.

The initial target is one complete and reviewed reconstruction for **Episode
38**, followed by a simulation/learned-policy pilot, then expansion to every
scored episode and object. Starting all policy training before establishing
correct camera motion, metric scale and contact would consume substantial
compute without validating the input contract.

Current machine-readable evidence is under
`reconstruction/incident_reports/track3_baseline/`. Read the timestamped JSONs
and the ends of the logs before resuming; this document is a snapshot.

- All released Track 3 files have been downloaded at a pinned dataset revision.
- The 20 scored episodes match the kit's public row template: **5,181 source
  frames, 36 episode/object instances, 9,120 required pose rows**, one world.
- All 60 evaluation video headers match the required source-frame counts, and
  their first and last frames decode. This is not a claim of sequentially
  decoding every frame of every video.
- Separate development data covers 20 different episode IDs. Its reference
  trajectories and scans are development assets.
- The host ego wrappers are installed and the consolidated entrypoint's help
  works in `reconstruction/.venv`.
- Episode 38's first-frame dustpan detection has been visually reviewed.
  Grounding DINO found one box with confidence 0.7974 covering the actual pan
  and circular handle, excluding hands/table.
- The pilot runner completed masks and depth for all 263 frames and a
  depth-grounded SAM3D dustpan candidate (134,622 vertices, 269,204 faces).
  Sampled masks track the pan through pickup/rotation, and its triangular pan
  and circular handle are recognizable. Surface cleanup and independent metric
  scale validation remain. Evidence: `episode_000038/execution.json`.
- The dedicated Blackwell hand image has built; WiLoR weights were provisioned
  and its GPU loader completed. Raw hand reconstruction also completed all 263
  frames, with no empty-detection frames. There are 266 left and 261 right
  detections and up to three hands in a frame, so temporal association and
  source-person identity filtering remain necessary. These are virtual-camera
  estimates awaiting depth/world alignment. Evidence: `episode_000038/hands.json`.
- FoundationPose ONNX models were downloaded with hash verification after the
  user explicitly accepted the NVIDIA Open Model License in this session.
- The original TensorRT 10.7 engine build failed explicitly because `sm_120`
  is unsupported. The separate TensorRT 10.9/CUDA 12.8 image and both `sm_120`
  engines are now built. An inherited CARI4D `learning` package collision was
  resolved by isolating FoundationPose's PYTHONPATH in its image.
- After the crash recovery below, camera-frame object tracking completed all
  263 frames with finite translations, unit quaternions and positive depth.
  This is reconstructed reference motion; camera/world alignment and tracking
  quality review still precede learned-policy training. Evidence:
  `episode_000038/tracking.json`.
- HaMeR's 6,037,554,929-byte upstream archive completed its resumed transfer,
  full gzip validation and safe extraction. The checkpoint SHA-256 is
  `e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53`.
  Evidence: `hamer_provisioning.json`.
- The bounded hand continuation completed silhouette-seeded SAM2 masks,
  association, depth alignment and diagnostic overlays. The final HaMeR pass
  has **263 real frames for each hand**, with no gap interpolation needed.
  Evidence: `episode_000038/hamer_pipeline.json`.
- The depth-preparation batch completed all **other 19 scored episodes** at
  `2026-10-17T18:49:31Z`, making depth/intrinsics available for all 20 episodes. It uses the same raw RGB
  and MoGe baseline as the pilot. This prepares depth/intrinsics and does not
  imply complete reconstructions or policies for those episodes. See
  `depth_batch/execution.json` and `depth_batch.log` for live progress.
- No Track 3 policy has been trained. No submission or leaderboard score exists.

## GPU continuation after recovery

Hand masks use rendered WiLoR MANO silhouettes as seed prompts. The initial
box-prompt trial selected a floor seam for the left hand; that attempt is
preserved under the pilot's `attempts/bbox_seed_*` directory. The accepted
silhouette prompts and masks were visually checked against both source hands.
Final WiLoR association yielded 263 left and 248 right real frames. Its missing
right-hand interval exceeded the configured gap bracket, so those records
were preserved as missing. HaMeR then predicted and aligned all 263 frames for
each hand from the propagated masks. No reference object poses were substituted.

The empty-track branch of the native interpolation utility previously emitted
a four-field summary where the summary printer required five fields. This is
corrected and covered by a test that keeps an empty hand track empty while
preserving the other hand's real record.

Both hand runners use a 55% PyTorch VRAM allocation budget, a 19,000 MiB total
GPU watchdog, 16 GiB container RAM, 18 GiB combined RAM/swap and six CPUs.
They share the pilot GPU lock. The HaMeR runner automatically waits for verified
weights and the completed mask/association pipeline before starting.

The remaining-depth runner shares that GPU lock, releasing it between episodes.
It uses batch size eight, a 70% PyTorch allocator budget (increased from 55%, then 65%, on
2026-10-17), a 21,000 MiB total GPU watchdog, the same RAM limits and a named
`v2d-track3-depth-stage` container. Each episode records source-video and model
hashes, image identity, commands, output hashes and exact frame coverage. Its
completed outputs are under `reconstruction/data/track3/episodes/episode_XXXXXX/`;
Episode 38's original pilot outputs remain in their existing directory.
The first batch job reached 100% GPU utilization using roughly 7.6k MiB VRAM.
The timestamped report confirms all 19 completed; no depth GPU job remains.
A failure in any subsequent batch stops that queue and records the error.

Diagnostic hand MP4s currently use the renderer's native 30 fps default, with
all source frame indices preserved. Their playback cadence is not the motion
sampling rate: the original source MP4s are 20 fps. Use the source timing when
preparing simulation data. World/camera alignment, scale/contact validation,
retargeting and learned policy training remain after these reference estimates.

## October 17 crash and verified recovery

The laptop rebooted unexpectedly during full-resolution FoundationPose
registration. Windows logged events 41 and 6008. WSL logged a failed GPU
make-resident allocation (`-12`), and the last telemetry sample showed
**24,001 / 24,463 MiB VRAM** used. These are evidence of severe GPU memory
pressure near the crash; a definitive hardware/driver root cause is not proven.
No pose had been saved by that interrupted attempt.

After reboot, all saved hashes matched for the completed detection (1 file),
masks (263), depth/intrinsics (526), mesh outputs (3) and raw hand outputs (263).
Those stages were preserved. Both compiled TensorRT engines survived.

The interrupted tracking stage was restarted with:

- Correctly scaled **1014×760** tracking input, preserving all 263 source frames.
- A **55% PyTorch VRAM allocator limit**, leaving capacity for TensorRT,
  nvdiffrast, CUDA and Windows allocations outside that allocator.
- A dedicated watchdog that stops only `v2d-track3-pilot-tracking` if total GPU
  use reaches **19,000 MiB**.
- A 20 GiB container RAM limit, 22 GiB combined RAM/swap limit and eight CPUs.
- One GPU stage at a time during recovery.

The restarted stage **completed successfully in 140.3 seconds** at
`2026-10-17T17:42:32Z`. Telemetry still reached 99–100% GPU utilization during
compute; peak sampled VRAM during recovery stayed below the 19,000 MiB guard.
Peak sampled VRAM was **18,464 MiB**. No memory guard was triggered. Keep
these limits for future tracking; do not
rerun the unbounded full-resolution command from the interrupted report.

HaMeR resumes through a separate host downloader that preserves the interrupted
tail, re-downloads its last 8 MiB, requires a valid HTTP byte-range response,
flushes progress, verifies the completed gzip stream, and extracts with the
safe data filter. Its completed archive and checkpoint hashes are recorded
only after success. This download uses no GPU.

Recovery evidence is in `reconstruction/incident_reports/track3_baseline/recovery_20261017/`.

## Official contract and rule interpretation

Read these sources before producing a plan:

- [Challenge overview](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)
- [Official FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)
- [Released dataset](https://huggingface.co/datasets/nvidia/video_to_data_challenge)
- Local authoritative packing instructions:
  `reconstruction/v2d_submission_kit/v2d_submission_kit/tracks/track_3/README.md`
- Local public roster:
  `reconstruction/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json`
- Local public pose template:
  `reconstruction/v2d_submission_kit/v2d_submission_kit/data/track_3_sample_submission.parquet`

The FAQ requires learned-policy object trajectories and the participant's own
reconstructed object meshes for all 20 scored egocentric videos. Although some
objects also have scans in the public development release, those scans must
not replace reconstructed evaluation meshes or supply their evaluation poses.
The native default embodiment is Sharpa Wave; other embodiments, including
floating hands, are allowed. G1 + Dex3 is fixed for Track 2, not Track 3.

The final artifact is one combined Parquet for six metrics: AUC, SP-SR, MP-SR,
RPE, MPPE and CD-O. The packer embeds every reconstructed mesh; the uploader
extracts the CD-O artifact. One public producing-code GitHub commit URL and
real checkpoint/evaluator/packer/uploader hashes are required. Never fill
those provenance columns with sample placeholders.

Mesh vertices and pose translations use metres. Submitted quaternions are
**xyzw**. Mesh body axes/origin must agree with the achieved rollout poses.
Registration can normalize mesh scale for matching, but rollout translations
are not rescaled, so wrong metric motion remains penalized. Every object in a
two-object episode needs a trajectory in the same world frame.

Local confidential evaluation trajectories/scans are unavailable. Image
reprojection, mask overlap, contact plausibility and development-set metrics
are useful diagnostics, not official held-out scores. The empty sample meshes
and zero sample poses are format placeholders, not acceptable predictions.

## Hardware, environment and artifact separation

- Linux/WSL2 checkout: `/home/alien51/video_to_data`.
- Reconstruction root: `/home/alien51/video_to_data/reconstruction`.
- Host Python: `reconstruction/.venv/bin/python` from the repository root;
  `.venv/bin/python` when working inside `reconstruction/`.
- Host environment initially lacked pip; pip was installed with `uv pip`, then
  the scoped ego wrapper install script ran successfully. Heavy ML inference
  stays in Docker. The Windows `python` shim on PATH is broken; use the explicit
  interpreter above or system `python3` for standard-library-only scripts.
- GPU: RTX 5090 **Laptop**, Blackwell `sm_120`, 24,463 MiB visible VRAM.
- Host memory: approximately 31 GiB RAM and 8 GiB swap; 24 visible CPU cores.
- Available filesystem space at campaign start: approximately 620 GiB.
- Existing compiled image: `v2d_cari4d:blackwell`, PyTorch 2.8/CUDA 12.8.
  Base image ID at startup:
  `sha256:e392b3ae8df2519986bbe69d6f33c611bdbe680748ade48339bd772c7d0183f9`.
- Existing segmentation/generation images: `v2d_grounding_dino:blackwell`,
  `v2d_sam2:blackwell`, `v2d_sam3d:blackwell`.
- Dedicated hand image: `v2d_track3_hands:blackwell`.
- Dedicated tracking image: `v2d_track3_foundation_pose:blackwell`, validated
  by successful TensorRT engine builds and the recovered 263-frame tracking run.
- Large data and models under `reconstruction/data/` remain locally ignored.
  Commit code, status and compact provenance reports; preserve generated
  checkpoints, videos and geometry locally.

Do not overwrite or retag the established Track 1/2 runtime images. Dedicated
Track 3 images derive from the already compiled stack. The WiLoR upstream
metadata limits torch to 2.5; its code is installed without that dependency
restriction to retain Blackwell support, and imports/GPU loading are checked.
Pinned WiLoR, HaMeR, manotorch and chumpy commits are in its Dockerfile.
The dedicated hand image enables compatibility loading of upstream serialized
YOLO/HaMeR publisher checkpoints; do not use it for arbitrary untrusted models.

## Inputs and episode mapping

Dataset repository revision:
`5f68335f3acc802033d1e80728c1633197521de8`.

Local source root: `reconstruction/data/v2d_challenge/track_3/`.
The release has `public/` development references/assets and `evaluation/videos/`
for scored input clips. The adapter does not derive scored IDs from a dense
range and does not pair them with Track 1/2 episode numbers.

| Episode | Scored frames | Objects in scored slot order |
|---|---:|---|
| 1 | 335 | white_pot, white_pot_lid |
| 3 | 313 | white_pot, wooden_spoon |
| 5 | 288 | white_pot, wooden_spoon |
| 6 | 243 | white_pot, wooden_spoon |
| 8 | 219 | beige_cup, white_pot |
| 10 | 161 | blue_cup, beige_cup |
| 14 | 158 | blue_cup, plastic_dish_rack |
| 15 | 166 | plastic_dish_rack, wooden_spoon |
| 16 | 173 | plastic_dish_rack, wooden_spoon |
| 19 | 235 | mini_sweeper, mini_dust_pan |
| 20 | 202 | blue_cup, water_pitcher |
| 25 | 356 | wooden_piece_1, wooden_piece_2 |
| 28 | 220 | plastic_dish_rack, white_pot |
| 30 | 263 | mini_sweeper, plastic_dish_rack |
| 32 | 259 | blue_cup, wooden_spoon |
| 33 | 348 | blue_cup, wooden_spoon |
| 35 | 421 | wooden_spoon |
| 36 | 407 | wooden_spoon |
| 37 | 151 | wooden_spoon |
| 38 | 263 | mini_dust_pan |

The RGB stream is `observation.images.ego_cam_a`, 2028×1520; the two mono
streams are `ego_cam_b` and `ego_cam_c`, 1280×800. The evaluation MP4 headers
report 20 fps. Public metadata's nominal acquisition rate is approximately
20.254 fps; preserve the actual input/frame mapping rather than silently
assuming 30 or 50 source fps. Future simulation may run at a different control
rate; export every exact required source frame once.

Factory calibration is at `public/meta/camera_calibration.json`. It describes
the camera rig and stereo baseline, but has no camera-to-mocap-world extrinsic.
Ego rotation metadata is 180 degrees and the stereo left/right order is swapped.
The released videos are already upright, verified visually. Validate the
calibration's pixel-frame convention before rectification; do not rotate these
videos a second time or treat factory rig transforms as world-camera poses.

## Reproducible commands and evidence

From repository root:

```bash
reconstruction/.venv/bin/python reconstruction/scripts/prepare_track3_campaign.py --download
reconstruction/.venv/bin/python reconstruction/scripts/run_track3_object_pilot.py --stage detect
# Inspect data/track3/pilot/episode_000038/detection_review.jpg and record the
# matching detections SHA-256 in detection_review.json before continuing.
reconstruction/.venv/bin/python reconstruction/scripts/run_track3_object_pilot.py --stage reconstruct --depth-batch 4
reconstruction/.venv/bin/python reconstruction/scripts/run_track3_pose_pilot.py
reconstruction/.venv/bin/python reconstruction/scripts/resume_track3_hamer_weights.py
reconstruction/.venv/bin/python reconstruction/scripts/run_track3_hand_pilot.py
reconstruction/.venv/bin/python reconstruction/scripts/run_track3_hamer_pilot.py
reconstruction/.venv/bin/python reconstruction/scripts/run_track3_depth_batch.py
reconstruction/.venv/bin/python -m pytest -q reconstruction/scripts/test_track3_campaign.py
```

The object runner is deliberately a reconstruction-stage pilot, not the
complete hand/object/world/policy pipeline. It uses the original upright RGB
without rectification, runs reviewed text detection, full-video SAM2 and MoGe
depth, then depth-grounded SAM3D with seed 42. The saved transform must be
validated and applied consistently when preparing the simulation body mesh
and its tracked poses. An unbaked SAM3D candidate is not a final rollout asset.

Depth batch 4 and SAM2 can overlap; GPU telemetry is saved every two seconds
in `episode_000038/gpu.csv`. Samples reached 99% GPU utilization during depth
inference. SAM3D follows after both stages finish to give the larger model
available memory. Size batches using observed peak memory and throughput.
Reserve VRAM headroom for native CUDA, TensorRT and Windows. CPU image encoding
and model downloads can temporarily lower utilization.

Each expensive stage records its command, container image ID, code/input
hashes, output hashes and elapsed time. A previously completed stage is reused
only when those hashes still match. Nine CPU test cases cover changed inputs
and outputs, failed stages, each runner's guard target and threshold, and the
empty-hand interpolation case. A nonblocking host file lock prevents two
copies of this pilot from writing its outputs simultaneously.

Build the dedicated images from `reconstruction/`:

```bash
docker build -t v2d_track3_hands:blackwell \
  -f modules/v2d_wilor/docker/Dockerfile.track3-blackwell .
docker build -t v2d_track3_foundation_pose:blackwell \
  -f modules/v2d_foundation_pose/docker/Dockerfile.track3-blackwell .
```

Monitor from repository root:

```bash
tail -n 12 reconstruction/incident_reports/track3_baseline/pilot.log
tail -n 12 reconstruction/incident_reports/track3_baseline/foundation_pose_build.log
tail -n 5 reconstruction/incident_reports/track3_baseline/hamer_weights.log
tail -n 10 reconstruction/incident_reports/track3_baseline/depth_batch.log
docker ps
nvidia-smi
```

The initial FoundationPose engine failure is preserved in
`foundation_pose_engines.log`. The successful ONNX provisioning log is
`foundation_pose_download.log`; model manifests contain verified SHA-256s.
The license acceptance applies to these models and persists for later runs.

## Remaining work before policy training and submission

1. Review the pilot masks and reconstructed surface against several RGB views.
   Confirm real object identity, holes/thickness and metric scale. Keep the
   original SAM3D transform and a transform ledger for the eventual body mesh.
2. Preserve the validated FoundationPose image/engines, HaMeR checkpoint and
   resource limits. Review the completed aligned-hand overlays and source
   frame mapping before accepting the estimates for simulation.
3. Estimate camera motion and gravity/support geometry, then express hands and
   objects in one stable metric world. Compare monocular estimates with the
   released stereo pair where calibration is verified. Do not train on poses
   still moving with the head camera.
4. Retarget hands to the native floating Sharpa embodiment and review contact,
   joint limits, mesh/body alignment and support collisions. The existing
   FlashCHORD checkout includes `sharpa_hands` and a MANO/Sharpa data loader.
   Do not reuse the Track 2 G1 source adapter as though it handled ego MANO.
5. Run a short training/evaluation/save/export smoke, then a full training pilot
   only after input and resource checks pass. Evaluate actual achieved object
   motion without object assistance, reference-pose substitution or freezing
   failures. Preserve the full frame roster, failures and nonfinite diagnostics.
6. Expand reconstruction, shared-world retargeting and learned policies to all
   20 episodes and 36 object instances, explicitly handling two-object scenes.
   The current consolidated ego runner is a single-object pilot; extending it
   requires preserving both object identities and their common camera/world.
7. Pack with `--track track_3 --results ... --meshes ...`, validate all 9,120
   required pose rows and all 36 reconstructed meshes, and inspect provenance.
   Keep every mesh in exactly the same body coordinates as its rollout poses.
8. Check Track 3 organizer registration and entry in all six Kaggle competitions
   using account `iorideepp`. Track 2 rule acceptance does not establish Track 3
   entry. Never accept legal rules automatically. User confirmation/API entry
   checks are still outstanding for Track 3 and do not block local computation.
9. Recheck the official per-track submission allowance, use the official uploader
   and an exclusive receipt, and poll that receipt for all six scores. Avoid
   repeating an ambiguous upload. There is currently no Track 3 leaderboard
   comparison because no Track 3 submission has been made.

For a future planning LLM: return an actionable `.md` plan with prioritized
experiments, expected local artifacts, exact validation/acceptance criteria,
compute/storage estimates tied to this laptop, and explicit prerequisites.
Do not claim access to this laptop's local artifacts from the planning laptop,
invent held-out scores, or label reconstructed reference motion as a learned
policy rollout. The user will bring your Markdown back to this execution agent.

The depth worker was restarted to apply the 70% allocator cap and batch size eight. This cap is an
upper bound, not a reservation or a guarantee of faster inference. Native CUDA
allocations remain additional. Hand and FoundationPose limits remain at 55%.
For this resource-only transition, `run_track3_depth_batch.py --resume-budget-change`
checks completed output hashes, all model/video/module/helper input hashes,
Docker image identity, stage runner identity and the full command apart from the
allocator environment setting and batch size. Previous batch-four outputs remain
valid and keep their original provenance; new batches may have small floating-point
differences. It retains original per-episode producing reports
and lists preserved episodes separately in `execution.json`. Only this queue
script's hash may differ; do not use this option after computational changes.

On 2026-10-17 the user requested a further watchdog increase. The active depth
queue now stops its own container at total GPU usage >=21,000 MiB, leaving
3,463 MiB below the reported 24,463 MiB GPU capacity. This is a stop threshold,
not a memory reservation; polling can overshoot it. Batch size remains eight
and the PyTorch allocator cap remains 70%. Hand/FoundationPose guards and the
historical crash-recovery settings remain 19,000 MiB. The queue restart preserves completed results with their original provenance;
the episode active at restart is rerun. Episodes 32 and 30 completed before
rollout, bringing the preserved non-pilot depth total to twelve.

## Camera-motion continuation

The user requested resuming the pipeline after the depth batch completed.
`reconstruction/scripts/run_track3_camera_batch.py` now queues DROID-SLAM for
all 20 scored episodes, with Episode 38 first. It waits for an isolated
`v2d_track3_droid_slam:blackwell` build, then acquires the shared GPU lock.
Live state: `incident_reports/track3_baseline/camera_batch/execution.json`;
build log: `incident_reports/track3_baseline/droid_build_cuda128.log`;
queue log: `incident_reports/track3_baseline/camera_batch.log`. Read these
actual states before claiming any camera stage completed.

The original CARI image has torch 2.8/cu128 but a CUDA 12.4 compiler, which
rejected compute_120. The dedicated image instead derives from
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, pins DROID to
`2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99` with its submodules, removes
lietorch's hardcoded legacy architecture flags and compiles for sm_120.
The official Google Drive checkpoint downloaded and passed a CPU
`torch.load(weights_only=True)` check with 102 state-dictionary entries.
Its actual hash is recorded in the camera queue state before GPU execution.

Camera stages use 512x384 inputs, stride one, a 384-keyframe buffer, 70%
PyTorch allocation, a 21,000 MiB total-GPU guard, 20 GiB RAM/22 GiB combined
RAM+swap, eight CPUs and one GPU container at a time. They use existing
MoGe depth as a bundle-adjustment sensor prior and median source intrinsics.
Every exact source frame must have a finite camera pose with unit quaternion
and unit scale. Reports preserve commands, image identity, code/model/input
and output hashes, frame coverage, translation ranges and largest pose step.

Outputs are `camera_poses_slam/`, `camera_trajectory_slam.txt`, and
`camera_slam_manifest.json` in each episode's existing work directory.
These poses are world_T_camera with wxyz quaternions in an arbitrary SLAM
world. No factory calibration rotation is applied. Metric scale remains tied
to unverified monocular depth, and gravity/support alignment, visual tracking
review, composing camera/object/hand transforms, retargeting and learned-policy
training remain. Completion of this queue alone does not make a submission.

### Camera queue startup update

The CUDA 12.8 build compiled all three extension packages, then initially
failed its final import smoke because upstream imports matplotlib. Adding
matplotlib after the compiled layers preserved the build cache. The corrected
image completed its import smoke and exported successfully; see
`incident_reports/track3_baseline/droid_build_matplotlib.log`.
The camera queue restarted successfully with Episode 38 active. Consult its
live execution JSON for subsequent completion/failure; startup is not evidence
that all twenty camera trajectories finished.
DROID checkpoint SHA-256:
`46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526`.
All 25 CPU checks passed after the dependency fix.

## Unattended full-pipeline campaign (October 17 continuation)

The user authorized completing Track 3 and leaving this laptop computing while
away for several hours. This starts an actual reconstruction/retarget/training
queue, rather than stopping after each completed batch. It can take substantially
longer than a few hours: the native full training budget is **250,003,456
transitions per episode**, up to roughly five billion transitions across all 20.
No completion time or leaderboard quality is guaranteed by launching it.

Two host processes coordinate through the existing exclusive GPU lock:

- `reconstruction/scripts/run_track3_reconstruction_campaign.py`: preserves the
  Episode 38 reference estimates, then detects each other scored object using
  its semantic prompt, propagates SAM2 masks, reconstructs a new depth-grounded
  SAM3D mesh, scales its baked scene once, and runs FoundationPose. Two-object
  clips keep both slots and distinct detection boxes. Several reference views
  are tried before an undetected object is recorded as a failure.
- The same reconstruction queue runs WiLoR, chooses an observed two-hand seed,
  renders silhouette prompts, propagates separate hand masks, runs HaMeR,
  depth-aligns hands, and fills only bounded interior gaps (15 source frames;
  no extrapolation). Missing hands are not invented. It packages one portable
  result bundle per episode/object using the same camera and hand tracks.
- `reconstruction/scripts/run_track3_full_campaign.py`: watches completed
  reconstruction manifests and prepares simulation references on CPU while
  the GPU reconstructs later clips. Once reconstruction ends, it trains and
  evaluates each prepared episode sequentially, then invokes the official
  Track 3 packer only if every required episode was successfully evaluated.
- `reconstruction/scripts/run_track3_policy_stage.py`: runs native FlashCHORD
  training/evaluation under the shared GPU lock, with a 21,000 MiB total-memory
  watchdog. Training uses JAX/Warp/Newton, so a PyTorch allocator cap does not
  apply to it. XLA preallocation is disabled. Reconstruction uses PyTorch:
  70% normally, 80% for SAM3D generation, 55% for TensorRT tracking; each has
  the same 21,000 MiB total-memory stop guard. One GPU job runs at a time.

Live files (paths relative to `reconstruction/`):

- `incident_reports/track3_baseline/reconstruction_campaign/execution.json`
- `incident_reports/track3_baseline/reconstruction_campaign.log`
- `incident_reports/track3_baseline/full_campaign/execution.json`
- `incident_reports/track3_baseline/full_campaign.log`
- `incident_reports/track3_baseline/pilot_training_smoke.log`
- Per-object producing commands, image IDs, inputs/output hashes and logs live
  under `reconstruction_campaign/episode_XXXXXX/`.
- Per-episode loading, alignment, retargeting, training, evaluation and packing
  evidence lives under `full_campaign/`.

Simulation data and checkpoints remain ignored under
`robotic_grounding/data/track3/`; reconstructed references remain ignored under
`reconstruction/data/track3/`. This is separate from Track 1/2 runtime data.
The native loader image is `v2d_track3_loader:blackwell`; CPU retargeting reuses
`v2d_track2_retarget:baseline` with Track 3 inputs and isolated data roots.
This image reuse does **not** imply reusing Track 2 reference motion.
FlashCHORD uses `flash_chord:track2-baseline` with `experiment=sharpa_flash_sac`,
which selects floating Sharpa hands, rather than Track 2's G1 embodiment.

### Corrections and validated pilot handoff

The portable bundle writer expects OBJ input. Generated GLBs are exported to
metric OBJ before packaging; the same baked body geometry is retained. The
FoundationPose image contains compiled `mycpp` binaries, so its host source
mount is omitted: hiding those image binaries caused a startup failure in the
first expanded tracking attempt. Failed attempts remain recorded locally.
Other source mounts remain available for pure-Python model wrappers.

The native ego loader assumed **30 fps**. A backwards-compatible `--fps`
argument preserves that default for other users but explicitly sets **20 fps**
for this campaign. Both simulation control and task playback are configured
at 20 fps and speed 1.0. `source_frame_playback` stays false because that flag
is supported only for motion_v1, while the ego loader produces ManoSharpaData.
At matching source/control rates, all original source-frame indices remain.

The downloaded HaMeR bundle contains only MANO_RIGHT.pkl. HaMeR predicts left
hands using that right model followed by reflection. To use the native two-side
FK loader consistently, `provision_track3_mirrored_mano.py` derives an explicit
**mirrored-right left asset**, not an official independent MANO_LEFT model.
It reflects template/shape/pose blend shapes, axis-angle statistics and face
winding. A fixed-seed random-pose FK check matches mirrored vertices and the
native left-tip convention exactly (maximum error 0.0 m in the observed check).
Both original and derived hashes and this limitation are recorded in
`data/weights/track3_mano/mirrored_left_manifest.json`. No model binary is
committed. Native world re-pose gates passed for both pilot hands at submicron
numerical error; this is coordinate consistency, not real-world accuracy.

Episode 38 completed native loading, 20-fps Parquet export, Sharpa IK retargeting,
and rigid-object URDF generation. The short training smoke uses 128 worlds and
65,536 transitions to test the simulation/checkpoint handoff. Its checkpoint
must not be confused with a full-budget trained submission candidate. Read its
log and actual artifacts before claiming it completed or passed evaluation.

### Shared worlds, policies, and completion gates

Each episode's object bundles use one camera trajectory. CPU preparation
estimates one global rigid alignment from the primary reconstructed object's
thin PCA axis and sets the union of initial reconstructed meshes on z=0.
This is a support/gravity heuristic, **not independently measured gravity**.
The loader accepts a validated rigid `--scene_alignment_path`, applying the
same SE(3) to every object's motion and both hands. Numeric gates verify the
loader's object motions equal this transform applied to the raw bundle for
every source frame, and that both objects' hand tracks agree in one world.

Object bodies are concatenated in scored slot order. Per-body mesh/URDF/radius
metadata stays separate; active contact IDs are remapped to slot+1, and contact
slots from both objects are retained. One merged episode is retargeted and
trained, so two-object scenes are not replaced by independent single-object
policy rollouts. Each instance gets a unique mesh filename; public development
scans and other episodes' reconstructed meshes are never substituted.

Full policies use 512 worlds, one-million-entry replay, 51,200 warmup entries,
1024 samples per update, seed 42, disabled remote logging, and the native
250M-transition curriculum, including its zero-assistance stages. Real learner
`state_*.safetensors` checkpoints allow resuming training. Training/evaluation
resources are bounded to 20 GiB RAM, 22 GiB combined RAM/swap and eight CPUs.
Native evaluation uses explicit initial resets, one world, actual achieved
object trajectories, and no object assistance. It preserves failures rather
than relabelling reference trajectories as achieved motion.

Before packing, every episode must have the exact required frame/object grid,
finite valid rollout worlds, real checkpoint/evaluator provenance, and all 36
reconstructed meshes in the same body frames as their achieved poses. Geometry
is copied into `robotic_grounding/data/track3/submission_meshes/episode_XXXXXX/`.
A failed reconstruction/preparation/policy is recorded individually; it cannot
be replaced with zeros, a public sample, another episode, or a reference pose.
A GPU guard event halts GPU-dependent work for inspection rather than repeatedly
retrying the allocation. A complete pack is marked `PACKED_FOR_REVIEW`.
This queue does not upload to Kaggle or accept rules automatically. Track 3
competition entry/organizer registration remain separate outstanding checks.

### Pose-registration recovery after the unattended run

The first campaign stopped at Episode 14's FoundationPose registration: the
total-memory watchdog observed 22,435 MiB, above the 21,000 MiB limit. Episode
37 completed reconstruction, but regenerating its mesh/bundle during earlier
runner changes invalidated its previous CPU-preparation receipt. The controller
correctly rejected that stale reference; it did not train on mismatched assets.

Pending FoundationPose jobs now use 760 x 570 processing images and a 45%
PyTorch allocator cap. TensorRT/native allocations remain additional; the total
watchdog stays at 21,000 MiB. Source frame indices, camera trajectories, and
submission frame counts remain unchanged. This resolution trades image detail
for lower registration memory and must be assessed against tracking quality.

Restart now reuses previously completed episodes after checking the original
video hash, manifest frame/object counts, finite camera/object transforms,
object validity, and mesh existence. This avoids regenerating a finished mesh
solely because orchestration code changed. Episode 37's stale prepared outputs
and receipt were moved, preserving relative paths, under
`reconstruction/incident_reports/track3_baseline/recovery_pose_resolution_760/`;
CPU preparation is regenerated from the current reconstruction. Its source
bundles remain intact. The live execution JSON files are authoritative for
whether this recovery completed; this note is not a claim of final training,
evaluation, packing, or submission success.

Reducing the image alone was insufficient: the next Episode 14 attempt still
hit 21,876 MiB. The registration renderer constructed crops for all 252 pose
hypotheses before TensorRT's existing inference batching. `trt_predictors.py`
now also constructs render/crop inputs in groups of at most 42, preserving all
hypotheses and their original order. Refiner inference retains its 42-pose
limit; scorer inputs are concatenated before the single global scorer call so
cross-candidate scoring is preserved. The optional debug scorer visualization
continues to use the original full-crop path; normal campaign tracking disables
that visualization.

The local FoundationPose image was refreshed with the changed predictor file,
retaining its compiled extensions. The regular Blackwell Dockerfile also copies
this source file on a full rebuild. All 10 predictor tests passed inside the
updated image, including 1/43/252-pose crop ordering and batch bounds. The actual
Episode 14 retry passed registration and reached source frame 22/158 with about
4,612 MiB observed GPU usage. That observation establishes progress past the
previous failure, not completion or tracking accuracy. Earlier watchdog reports
are archived under the recovery directory. Check live execution JSON for the
latest campaign result.
