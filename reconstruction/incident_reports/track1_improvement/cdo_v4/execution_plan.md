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

## Phase 3: seeded refinement experiments running

Episode 12 uses the same CoCoNet bundle and mesh in all experiments, seed 42 and identical settings except the named factor:

1. `control_100_batch64`: reproduce the refinement settings as a seeded experimental control.
2. `rotation_100_batch64`: enable rotation refinement only.
3. `steps300_batch64`: increase steps only; preserve fixed rotation and batch size.

GPU experiments run sequentially. Full-clip batching is a separate planned factor, deferred until these results and memory use are known. Runtime/progress is recorded in `ablation_status.json`; logs and checkpoints reside in the isolated data folders. Completed outputs still require diagnostic/visual review. No fourth submission has been packaged or uploaded.
