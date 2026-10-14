# CD-O improvement execution

## Objective and controls

Current scored control: CD-O 34.15571 cm. Official leaderboard baseline reported 23.16246 cm. Reduce spatial error while measuring all other metrics. Preserve submitted v3: all 60 NPZ/GLB artifact hashes rechecked and unchanged. New outputs use `data/v2d_challenge/track_1/cdo_v4_experiments/`.

## Phase 1: configuration audit complete

Verified checkpoint and configuration against the released manifest (both exact SHA256 matches). Standard local inference uses 300 refinement steps and full-clip batches; v3 used 100 steps and 64-frame batches. Geometry input and custom tracking also differ. Rotation is fixed by default in both. The official leaderboard team's complete input artifacts and run recipe are not available here, so this is a comparison to repository defaults, not a claimed exact leaderboard reproduction.

## Phase 2: six-episode depth diagnosis complete

Compared submitted v3, unused rollout candidates, and gated tracking on 33 scored-frame samples per episode. Rendered depth was validated with a synthetic box (known front depth 2.5m). Cached aligned monocular depth is a diagnostic, not reference truth. Thin hoops and boundaries can be contaminated by background. Each report records overlap support; Episode 1's submitted estimate has only 2/33 supported samples and cannot support a robust correction.

Gated tracking has substantially better depth agreement than final predictions in several cases, particularly rollers 19 and 20. Potential errors introduced by learned pose updates deserve isolated testing. This does not prove the tracker has lower hidden-reference CD-O.

## Translation-only probe complete; no promotions

Fit a constant camera-depth offset on two-thirds of samples and evaluate on the disjoint remaining third. Move the object center along its camera ray; keep geometry, rotations, and human predictions unchanged. Larger offsets can improve depth residual while worsening silhouette, so these are diagnostic probes only. Depth comparisons use each render's valid overlap; changing overlap can bias the residual. They require common-support and geometric review before any selection.

| Episode | Parent | Offset m | Evaluation IoU before → after | Depth residual cm before → after |
|---|---|---:|---|---|
| 1 | rollout_candidate | +1.382 | 0.175 → 0.067 | 126.15 → 125.00 |
| 2 | submitted_v3 | +1.135 | 0.012 → 0.007 | 117.82 → 20.84 |
| 2 | rollout_candidate | +0.157 | 0.686 → 0.440 | 14.22 → 6.12 |
| 12 | submitted_v3 | +0.143 | 0.069 → 0.063 | 15.47 → 10.54 |
| 12 | rollout_candidate | +0.038 | 0.725 → 0.720 | 7.57 → 11.38 |
| 14 | submitted_v3 | -0.149 | 0.072 → 0.081 | 16.45 → 2.41 |
| 14 | rollout_candidate | -0.056 | 0.732 → 0.736 | 8.61 → 4.84 |
| 19 | submitted_v3 | -0.063 | 0.452 → 0.446 | 14.60 → 11.76 |
| 19 | rollout_candidate | -0.063 | 0.452 → 0.446 | 14.60 → 11.76 |
| 20 | submitted_v3 | -0.201 | 0.431 → 0.405 | 20.35 → 6.07 |
| 20 | rollout_candidate | -0.201 | 0.431 → 0.405 | 20.35 → 6.07 |

## Phase 3: seeded refinement experiments complete, reviewed, not selected

Episode 12 uses the same CoCoNet bundle and mesh in all experiments, seed 42 and identical settings except the named factor:

1. `control_100_batch64`: reproduce the refinement settings as a seeded experimental control.
2. `rotation_100_batch64`: enable rotation refinement only.
3. `steps300_batch64`: increase steps only; preserve fixed rotation and batch size.

The separate 100-step full-clip experiment also completed. Compared with the seeded 64-frame control, IoU changes from 0.71401 to 0.71302, translation acceleration proxy from 3.70186 to 3.47259 cm, and large rotation jumps remain 72. The 300-step and rotation-enabled experiments also retain the flip failures. None is selected for submission. Full-clip batching is not sufficient to fix tracking ambiguity.

## Phase 4: temporal selection, geometry and contact review complete

The stage audit locates the flip failures upstream in tracking. A dynamic-programming selector evaluates twelve orientations per frame against the actual mesh silhouette; alternative orientations are hypotheses, not assumed symmetries. A separate circular-hoop mode transports an unoriented plane normal and removes irrelevant spin. It is explicitly restricted to the hoop metadata and geometry, so pans retain their asymmetric handles.

Translation smoothing is checked against video alignment. Shared scale/depth factors are fitted on 22 frames and checked on 11 separate frames using identical common depth pixels. Episode 14 accepts a -0.05m correction and Episode 19 accepts -0.10m. Episode 20 accepts a separately fitted -0.08m correction that reduces signed penetration on both fitting and evaluation frames.

All candidates copy the exact scored v3 human pose, scales and shape. Signed body intersection uses the actual submitted human decoder and a validated watertight 4000-vertex collision proxy, with deterministic area-weighted object samples. This is a local physical check, not the official hidden-reference PEN score.

Selected episodes: **2, 12, 14, 19, 20**. Episode 1's metric-ring and body-aware plane fits improve image alignment and motion but fail the signed-penetration gate; retain v3. Episode 2's unsigned hand distance worsens despite improved geometry/alignment and reduced motion; this uncertainty is recorded in its review decision.

Seven synthetic and exhaustive-reference tests pass. Exact Episode 12 before/after videos contain all 405 frames at 30fps and were decoded to validate completion. Review evidence and decisions are in `promotion_decisions.json` and `final_review.md`.

## Phase 5: package, official validation, one submission complete

Build a complete 30-episode package, verify every original v3 artifact against its recorded hash, check exact human-array identity, schema, scored-frame coverage, finite values, positive scales, rotations and meshes. Pack with the official converter and producing Git commit. Run official preflight and upload dry-run, then submit one version across all five metrics with an exclusive receipt. Compare official scores with v3; local diagnostics do not establish a leaderboard improvement.

All five submissions are COMPLETE. CD-O 28.45924 (16.68% better), ACC-O 0.88239 (19.82% better), human scores unchanged, PEN 0.00331 (4.09% worse). No current-batch steps remain. The 23.16246 baseline target remains unmet; next iteration must address physical consistency and the remaining CD-O gap. Exact values and IDs are in `submission_results.md`.
