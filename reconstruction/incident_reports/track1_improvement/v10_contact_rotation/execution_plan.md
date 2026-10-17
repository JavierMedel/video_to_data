# Contact-constrained object orientation repair from submitted v8

The user requested running the new improvement through the complete pipeline to a submission. The rejected Whittaker pilot is not promoted or silently expanded. This campaign addresses the remaining measured object-pose failures, using isolated v8-parent object-rotation proposals. All original submitted artifacts remain immutable. An eventual payload must cover all30 episodes and pass the full checks before one new version is uploaded across five metrics.

## Preregistered experiment

- Parent: complete v8 pairs and Parquet SHA256 `bd73a931e9f9ec85b920884415c5dc2123d86e2d3d91395c7820ebf564fd3198`.
- Initial targets: Episode7 (iron),18 and20 (paint rollers), selected from the all30 audit. Other episodes retain v8 unless subsequently fitted and reviewed. This is a pose-repair pipeline, not a rerun of original segmentation/reconstruction checkpoints.
- Variables: only per-frame object rotation. Human pose/shape/scales, object translation/scale and actual GLB remain exactly identical to v8. Thus this experiment preserves human alignment and three of the four non-PEN official metric inputs (CD-H, ACC-H, ACC-O); CD-O and PEN still require actual scoring.
- Fixed proposal set: parent orientation; local-axis offsets±5/10/20 degrees; projected nearby rotation averages with radii2/4. These are hypotheses scored against the actual asymmetric mesh, not declared symmetries.
- At each video frame, render all21 hypotheses at a384-pixel maximum side and retain the best two non-parent alternatives. A non-parent option needs at least0.005 IoU gain and visible observed support. Endpoints (first/last two frames) remain parent.
- Evaluate actual-mesh full-hand and FPS512 mean penetration and nearest-vertex gap for every retained frame/option. A new option must not increase either per-frame penetration, nor increase a parent-contact-like (<=1.5 cm) gap by more than0.5 cm. The parent is always available.
- An exact first-order DP chooses among the feasible options, using observed mask cost, a small orientation-fidelity term and geodesic angular-step penalty. The finite-option contact summaries can be reused after verifying selected arrays; this is explicitly fit evidence, not an independent hand rerender.
- All source/input hashes and intermediate choices/contact costs are checkpointed. A single coordinator lock prevents duplicate runs.

## Full acceptance, before packaging

For each changed episode:

1. Run all-scored-frame image/common-support inferred-depth review at512-pixel maximum side using the unchanged actual meshes.
2. Require mean IoU gain of at least0.02 (the returned plan's geometry/rotation criterion), no gross frame-level alignment loss, and common-depth residual not worse by more than0.5 cm where supported. This is local video evidence, not hidden-reference CD-O.
3. Require sampled-surface acceleration proxies not to regress beyond max(0.1 cm,1.1×parent) on both all-video and consecutive-scored-triple scopes. Object translation is exactly identical, so official ACC-O inputs are preserved.
4. Verify exact per-frame full-hand/FPS512 non-increase and contact-like gap bound on all frames. Run a separate sampled actual-mesh hand recomputation as an additional cross-check, labeled by its scope.
5. Inspect six source mask/mesh comparisons and actual reconstruction video samples; fully decode any full comparison videos claimed as complete. No generic gate is relaxed just to obtain a submission.
6. If no new candidate passes, preserve v8 and report that outcome rather than upload a known rejected or duplicate package.

## Package and submission

Stage a complete30-pair package, retaining the v8 source for every unpromoted episode. Validate finite arrays, rotation matrices, positive scales, meshes, scored-frame coverage and all60 parent hashes. Compare all740,780 packed numeric rows with v8, allowing only promoted object-rotation rows (array3). Publish a real producing-code commit, pack with its exact accessible URL, and run official environment preflight plus authenticated uploader dry-run. Check current quota/rules. Submit only one new version with exclusive upload guard and exact-ID receipt, then collect all five scores. The user's current instruction authorizes this reviewed submission; no redundant upload confirmation is needed if all requirements are satisfied.

## Environment compatibility

A mask-only review initially failed because the current tracker export module imports a symbol absent from the installed FoundationPose image. The mask raster/grid functions were isolated into `track1_mask_render.py`, preserving their equations. A GPU test executes the exact source-extracted original helper and confirms identical grids and batched masks at both test resolutions. Tracking models/pipelines are not initialized merely to render an existing submitted mesh.

## Recorded selector ablation

The first-order angular-step path for Episode7 increased the all-video surface-motion proxy by about13%, failing the preregistered10% motion limit. It also fails the campaign's0.02 mean-IoU criterion on full review. Its artifacts remain preserved. A separate second-order selector was therefore implemented over the exact same physically checked choices, using a fixed0.02 weight on mean-squared surface acceleration in cm². The mesh covariance/centroid identity evaluates each triple exactly; an exhaustive explicit-surface oracle test passes. This is a new isolated selector ablation, not a relaxation of contact or acceptance limits. Its outputs live under `surface_selection/` and retain all original choice/cache/status hashes. No additional pose or physical geometry is invented by reselection.

Visually ineligible alternatives were not subjected to expensive mesh queries; their unused cache slots copy parent values. Selected options must be visually eligible and physically evaluated. Status counters for completed slots include those placeholders; `explicit_mesh_evaluations` distinguishes actual baseline+eligible-option calculations. This distinction matters when interpreting workload totals.

## Completed local outcome

Only Episode18 passes the fixed final gates. Episode7 gains0.01151 IoU and Episode20 gains0.01539, below0.02, so both retain v8. The other27 episodes also retain verified v8. Episode18 changes only319 all-video rotations, including219 scored frames. Its complete535-frame source-RGB actual-mesh projection and object-crop videos decode at30fps; six crops and two full-scene samples were inspected. This is object-only normal-shaded projection over original RGB with observed-human occlusion, not newly rendered human reconstruction. Existing geometry mismatch remains. A separate33-frame actual-mesh hand check also passes. Unknown exact host hand-role IDs/alignment remain limitations of local PEN evidence.

The current official website still states five submissions per week. An authenticated history check on2026-10-16 found seven versions since Monday. Local validation/package preparation continues, but upload is held until the applicable allowance or an organizer-approved exception is established. Kaggle daily quota alone does not establish weekly eligibility; no new upload guard or upload receipt is created while this remains unresolved.
