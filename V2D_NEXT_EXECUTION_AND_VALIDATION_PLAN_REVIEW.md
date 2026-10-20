# Review of the next V2D execution and validation plan

Reviewed October 20, 2026 against execution checkout commit `cff4d10d3267cdaf1b2f6a6c98be1500bdeb304e`.

Reviewed proposal: `V2D_NEXT_EXECUTION_AND_VALIDATION_PLAN.md`.

## Verdict

The plan is a useful next step, but **needs corrections before execution**. Its priorities, preservation of baselines, deterministic checkpoint evaluation, real-asset decoder parity and measured pilots are sound directions. Several asserted facts and proposed gates do not match the execution environment or the code.

This review does not implement the plan, launch training, create a new candidate or upload anything. The original proposal remains unchanged. A read-only CPU Docker inspection of the existing MHR model was performed; no GPU was requested.

## 1. High priority: fix checkpoint ownership beyond the report directory

**Plan:** D1, Sections 1.3 and 4.1.

**Confirmed.** `cross_track/jobs.py:39` searches the requested output directory for learner states and can declare an existing policy `TRAINED`, or resume it, without proving the output belongs to the requested job. A separate hash-bound stage report does not establish ownership of pre-existing checkpoints in that output directory.

Treat this as a provenance correctness issue, not merely a medium scheduling issue. Later schema and coverage checks do not prove which arm produced a checkpoint.

The proposed output identity file is appropriate, with these corrections:

- Resolve paths and reject duplicate output directories within a manifest before starting any job.
- Check ownership before adopting checkpoints, including the completed-stage fast path.
- Write/check identity under an exclusive output-directory lock. Queue locks in different report directories do not prevent competing controllers from claiming one output.
- Protect actual submitted/baseline paths using resolved artifact provenance, in addition to any naming-based checks.
- An explicit `arm: resume` does **not** bypass the plan's own rule rejecting checkpoint-containing directories without an identity. Specify migration explicitly: use a new output directory and a hash-checked parent learner checkpoint, or implement a deliberate validated adoption path.
- Audit the generalized campaign entry points too; changing only the typed job runner does not automatically fix every campaign's checkpoint reuse behavior.

Required tests include foreign checkpoints, matching interrupted runs, warm-start/fresh collisions, duplicate outputs, symlinked protected directories, and explicit resume into a new output directory.

## 2. High priority: the actual MHR parameter transform is 889 by 249

**Plan:** Sections 3.1 and 4.6 describe or require a `889×204` model transform.

Read-only inspection of the local file `reconstruction/data/weights/track1_submission_mhr/mhr_model.pt` found:

```text
character_torch.parameter_transform.parameter_transform  (889, 249)
character_torch.skeleton.joint_translation_offsets        (127, 3)
character_torch.skeleton.joint_prerotations                (127, 4)
character_torch.skeleton.joint_parents                     (127,)
character_torch.linear_blend_skinning.inverse_bind_pose   (127, 8)
character_torch.blend_shape.shape_vectors                  (45, 18439, 3)
character_torch.blend_shape.base_shape                     (18439, 3)
character_torch.mesh.faces                                (36874, 3)
pose_correctives_model.pose_dirs_predictor.0.sparse_indices (2, 53136)
pose_correctives_model.pose_dirs_predictor.0.sparse_weight  (53136,)
pose_correctives_model.pose_dirs_predictor.2.weight        (55317, 3000)
```

The scripted forward explicitly concatenates `model_parameters` with `zeros_like(identity_coeffs)` before applying the parameter transform. The official NPZ decoder takes 204 model parameters directly.

**Correction:** distinguish the source TorchScript representation from the exported NPZ contract. Verify the column semantics, then export the effective first 204 columns corresponding to the scripted zero-padding behavior. Prove equivalence against the actual scripted forward on real frames. Do not reject this valid local model because its stored transform has 249 columns, or map buffers by shape alone.

The skin is stored as flattened vertex indices, joint indices and weights, not as a ready-made `skin_dense` buffer. Reconstruct it with accumulation and test normalization. The second corrective layer is a large dense stored matrix; inspect memory cost and sparsity before converting it to CSR and running double-precision parity in a small container.

Keep the proposed real-frame, three-way vertex/joint parity gate. It has not yet run. The alternate path `checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt` does not exist at the repository root on this laptop; locate any alternate copy rather than requiring that exact path.

### Units correction

`tools/track1/mesh_to_mhr_params.py:59` documents and implements mixed wrapper outputs:

- Raw TorchScript vertex coordinates are centimeters.
- `MHR.run(...)[0]` multiplies by the axis flip and 10, returning **millimeter vertices**.
- `MHR.run(...)[1]` applies the axis flip and divides by 100, returning **meter joints**.
- `track1_contact_rotation.py` divides those wrapper vertices by 1000, yielding **meters**.

The plan's statement that the contact wrapper divides by 1000 and therefore returns millimeters reverses that conversion. Keep vertex and joint conversions explicit to avoid applying a second scale or flip. The evidence adapter should pass batched Torch tensors exactly as the actual wrapper requires.

## 3. High priority: the proposed parent IoU equality gate compares different quantities

**Plan:** Section 3.2 and G3 require the new render's median IoU to equal `image_choices.npz` parent IoU within 0.01.

The existing parent image screen in `track1_contact_rotation.py`:

- rescales the image and intrinsics to a maximum side of 384;
- uses the nvdiffrast renderer;
- calls `measure_visible_silhouettes`, excluding the human mask from both observed and rendered masks.

`cross_track/image.py` instead rasterizes the supplied resolution on CPU and computes ordinary full-mask IoU. It has no human-occlusion input. A difference between these scores does not necessarily indicate a camera or frame error. The proposed equality gate can reject correctly aligned evidence.

**Correction:** first reproduce the historical screen with its exact grid, human mask exclusion and renderer semantics. Separately establish the new evidence-resolution baseline and compare candidate versus parent under identical new semantics. Record both quantities under distinct names. Where archived screen files do not exist for an episode, explicitly generate a baseline review instead of assuming all 30 episodes have them.

Also avoid requiring the centroid of both hands to lie within a human silhouette in 95% of frames. That centroid can lie in background between the arms even when both hands are correct. Use per-hand/per-vertex projection checks with visibility and occlusion handling, and calibrate the acceptance criterion on known-good frames.

## 4. High priority: a Tier 2 expansion gate violates the plan's own isolation default

**Plan:** Section 5.6(a) compares Tier 2 own-reference MPPE against the same episode's Tier 1 own-reference MPPE, requiring it to be within 2×.

The Tier 1 diagnostic is derived from clean Tier 1 reference values. Using it to choose whether Tier 2 training expands is feedback from clean-data-derived measurements. Calling both values “our own diagnostics” does not remove that dependence under Q3's default of no clean-data use for tuning or production decisions.

**Correction:** retain Tier 2-only rollout validity, coverage, finite-value, physical feasibility and measured resource gates. Compare Tier 2 methods on the same Tier 2 inputs with explicitly labelled proxies. Do not use Tier 1 trajectory-derived errors for Tier 2 continuation or selection unless the rules explicitly permit that use.

The proposal is also inconsistent between WP4's surrogate-based promotion language and Section 5.6(c)'s second-upload measurement. Define the exact allowed evidence and promotion policy before that experiment.

## 5. Kaggle D4 is not a confirmed installed-signature failure

The execution laptop has **Kaggle 2.2.4**. Local `inspect.signature` and source inspection show:

```text
competition_submissions(..., page_number=-1, page_token='', page_size=20)
competition_leaderboard_view(..., page_size=20, page_token=None)
```

The currently used `page_number` and `page_size` keyword arguments are supported. Do not block all snapshots on an unproven signature incompatibility.

There is a genuine compatibility/completeness question: the high-level submissions method returns only `response.submissions`, discarding any response-level token, and the leaderboard method prints a continuation token rather than returning a response envelope. A generic “read next_page_token from the returned object” implementation will not work against these list returns.

**Correction:** test the actual installed paging semantics and response structure. If token-based iteration is necessary, use the lower-level response-bearing client or an explicit supported adapter. Verify page boundaries, no missing/duplicate attempts, and termination before asserting `history_complete`. Keep quota checks closed if completeness cannot be established. No authenticated snapshot was requested or run during this review.

## 6. PEN constraint accumulation needs a qualified convergence claim

**Plan:** D2/D3 correctly identify replacement of constraints and loss of constraints during final repair in `track1.py:115` and `track1.py:131`.

Retaining useful constraints is worth testing. However, mesh collision avoidance is nonconvex. Half-spaces generated from particular closest faces are conservative choices of feasible side; retaining every historical face constraint can make the QP infeasible even when another collision-free trajectory exists.

Increasing the outer cap from five to ten and making the constraint set grow does not prove a collision-free solution or QP convergence. Replace that claim with a bounded sequential-convex heuristic: controlled active-set retention, infeasibility handling, fixed-anchor protection, exact penetration rechecks and rejection/fallback. Test concave geometry and changing closest faces, as well as the proposed cube case.

## 7. GCV and surrogate selection remain hypotheses

**Plan:** D6 and Sections 4.5/5.4.

The concern is valid: reducing our own acceleration does not prove reducing acceleration error against the hidden reference. However, ordinary weighted smoothing GCV targets a reconstruction/prediction residual criterion; it is not automatically a selector for acceleration error after robust fitting, contact constraints and projection.

Before changing defaults:

- Define weighted residuals, effective degrees of freedom, anchors, scored gaps and pooling precisely.
- Split surrogate trajectories by source episode into tuning and held-out validation sets. Do not tune and certify `lambda_cap` on the same examples.
- Evaluate correlated noise/drift and plausible real high-frequency motion. A high-pass residual estimated from v8 is not known to contain noise alone.
- Include unsmoothed and current max-passing controls, zero-noise controls and the actual constrained final fit.
- Treat a failed or empty qualifying lambda set as a valid result, preserving the baseline.
- Check whether a single cap transfers across frame rates, mesh-origin conventions, weights and motion types before imposing it globally.

The existing image/displacement gates also do not prove CD-O neutrality. Recentering can preserve surfaces; a subsequent translation change generally changes them. Label CD-O preservation as unmeasured without reference geometry.

## 8. Additional execution-plan corrections

- **Reward profile:** the implemented name is `object-focused`, not `object_focused`. It is currently supported for Track 3 Sharpa only. WP5's object-focused Track 2 proposal requires a separate G1 objective implementation and tests; it cannot simply reuse that profile.
- **Retarget G5:** `verify_motion` currently checks finite joints, frame counts, object quaternion normalization and one-rigid-transform preservation. It does not itself check robot joint limits, support-stage presence or contact feasibility. Add explicit checks before claiming G5 covers them.
- **Pilot ordering:** G5 currently requires all 30 episodes before even a single-episode training pilot. Use an episode-level gate for the pilot, then complete-cohort validation before scale-up/packing, unless there is a concrete dependency requiring the entire cohort first.
- **Throughput:** 13.5 hours divided by 25 episodes is about 32.4 minutes per episode, or 16.2 hours for 30 before new assumptions. More importantly, an average inferred from one historical campaign is not a hard upper bound for a different tier. Measure the pilot as proposed.
- **Checkpoint gains:** re-selection is a useful low-cost search, not a certain improvement. The final checkpoints may already be best. Preserve aggregate no-regression checks and report no gain honestly.
- **Checkpoint pilot path:** the current selector requires all 30 episodes. Add the explicitly diagnostic subset path, or call the native evaluation helper directly with correctly mounted report paths. Do not create a partial packet.
- **Track 3 curriculum smoke:** divisibility by 1024 alone does not demonstrate that all eleven stages were visited. Check resolved thresholds and recorded stages before calling a short run representative.
- **Tier 3 world frame:** distinguish the reconstruction manifest's `released_track2_opencv_world` from the packer NPZ's `released_track2_world`. The former is the current provenance contract, the latter the packer field. Establish a genuine permitted world anchor; do not change labels to hide gauge uncertainty.
- **Approval scope:** G12 adds a new exact-hash user-approval requirement. This review request authorizes review, not submission. For future execution, use the user's actual authorization; the proposal itself is not a source of mandatory new approval for work already explicitly authorized.

## 9. Confirmed useful parts of the proposal

- Preserve all scored packets, receipts and parent checkpoints.
- Fix output ownership and versioned development-job preparation before launching new jobs.
- Inventory existing Tier 1 checkpoints and measure deterministic evaluation before allocating new training budgets.
- Require real model parity before human fitting, with observed evidence and coupled object review afterward.
- Prepare genuine Tier 2 inputs and include Episode 16 explicitly.
- Use one heavy GPU job at a time with the existing 21,000 MiB guard.
- Separate public development evidence from Track 3 evaluation outputs.
- Measure throughput and stop expanding experiments when the remaining schedule cannot support them.

The pinned [Tier 2 README](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/5f68335f3acc802033d1e80728c1633197521de8/track_2/tier_2_synthetic_noise/README.md) was retrieved read-only during this review. It supports the proposal's statements that this release has no camera matrices, uses w-first object quaternions, retains a static exocentric stream, and declares ground-plane/support assets. Tier 2 is not yet downloaded in the expected local dataset directory. This source check does not settle competition permissions or the freeze date.

## 10. Recommended corrected first implementation scope

1. Fix job-output ownership and the version guard for Track 3 development jobs, with meaningful provenance tests.
2. Verify actual installed Kaggle paging and save a read-only authenticated snapshot when that execution work is authorized.
3. Add the diagnostic checkpoint subset/evaluation path; inventory all Tier 1 checkpoints and measure missing-rollout cost.
4. Build the MHR exporter against the inspected local tensor layout and prove real-frame parity, while correcting the evidence renderer comparison.
5. Obtain and validate Tier 2 inputs; use an isolated single-episode retarget/train/evaluate pilot without Tier 1 score feedback.
6. Run the surrogate study before changing the default lambda policy or scaling Track 1 fitting.

Do not launch the full multi-track schedule directly from the current proposal. Correct these contracts and gates first; then expand only from measured passing pilots.
