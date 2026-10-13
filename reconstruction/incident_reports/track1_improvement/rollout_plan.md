# Track 1 improvement rollout

The objective is to evaluate the remaining 27 episodes using the corrected-mask,
metric-mesh, quality-gated tracking, and 100-step refinement workflow. The scored
second submission is the comparison baseline. Episodes 6, 18, and 28 remain the
validated pilot results. All original predictions, meshes, and checkpoints are
preserved; new work uses `data/v2d_challenge/track_1/improvement_rollout_v1/`.

## Execution plan

1. **Inventory and image review.** Verify video coverage and reusable human/depth
   caches. Inspect object masks at multiple frames, including scored frames and
   occlusion/motion changes. Correct wrong-object prompts and drifting masks.
   A mask review sheet alone is not approval: record an explicit visual review.
2. **Prepare meshes.** Reuse a pilot mesh only for the same physical object and
   validate its metric scale in the new camera observations. For other object
   groups, regenerate a mesh from a reviewed visible frame and aligned depth.
   Reuse within a three-episode object group only after scale/shape review.
3. **Run isolated reconstruction.** Reuse compatible human initialization and
   aligned depth; run gated object tracking, CoCoNet, and 100 refinement steps.
   Add resumable tracking checkpoints before the expanded GPU batch. Do not
   weaken mask/depth gates to reduce runtime. Save stage logs and progress.
4. **Convert and compare every episode.** Use the official converter. Compare
   both old and new geometry against identical reviewed image masks and scored
   frame samples. Check positive depth, silhouette alignment, temporal motion,
   human overlays, and contact. Promote only supported improvements; preserve
   the second-submission prediction where a candidate regresses or is uncertain.
5. **Validate the complete candidate.** Verify 30 NPZ/GLB pairs, finite arrays,
   rotations, frame coverage, positive object scale, unchanged-file hashes,
   exact sample row IDs, public producing-code provenance, official packing,
   preflight, and authenticated uploader dry-run.
6. **Submission and results.** Upload a third version only if authorized for this
   rollout. Save the exact submission IDs, poll without duplicate uploads, and
   compare all five scores with the second submission.

## Batches and priorities

| Batch | Episodes | Purpose |
|---|---|---|
| 1 | 7, 8, 19, 20, 27, 29 | Extend the three pilot object groups; prioritize wrong masks, negative depth, and drift |
| 2 | 0–5, 9–14 | Hoop, bowl, desk, and pan: repair selection/scale and image disagreement |
| 3 | 15–17, 21–26 | Block, foam roll, and stool: improve tracking/refinement and verify actual motion; Episode 16 rerun approved |

## Status and updates

- Multi-frame visual mask review: complete for all 27 remaining episodes; 21
  sampled mask sets can be reused. Episodes 8 and 9 needed wrong-object correction;
  full-resolution review also found omitted handles in Episodes 12, 14, 19, and 20.
- Cache preflight: 27/27 PASS, including legacy Episodes 15 and 16.
- Tracking recovery: implemented and regression-tested; completed frames are
  saved atomically and resume rejects changed inputs or implementation.
- Execution: all 27 isolated inputs are prepared; six corrected mask sets passed review. Seven depth-grounded object-group meshes are ready, with the body-only pan mesh preserved and replaced by a full pan/handle mesh. The full queue is running on Episode 19.
  Corrected segmentation, seven depth-grounded object-group meshes, and the
  remaining sequential episode queue are prepared.
- Full candidate review complete: 2/27 (Episodes 7 and 8), approved for promotion. Episode 7 mean observed-mask IoU improved from 0.22576 to 0.71487, and translation acceleration proxy decreased from 6.87487 to 1.76550 cm.
- Episode 8 now targets the foreground iron rather than the background tripod: object IoU improves from 0 to 0.71756, and sampled surface acceleration decreases from 13.15445 to 3.08162 cm. Human silhouette IoU decreases by 0.00983, within the 0.03 review tolerance; reviewed overlays remain aligned.
- Motion review includes rotation jumps and sampled surface acceleration; remaining flips are recorded rather than hidden by translation-only diagnostics.
- Episode 16 isolated rerun: approved by the user's follow-up “yes continue.” Originals remain preserved.
- One validated third version across five competitions: approved by the same follow-up.
- Current GPU plan: sequential jobs on the 24 GiB GPU; concurrent inference jobs
  would compete for memory. The roller pilot took about 4.4 hours in tracking,
  so the complete rollout may take multiple days. Estimates will be revised
  from actual per-episode timing rather than assuming every episode is equally slow.
- Updates cover current episode/stage, completed/reviewed/retained counts,
  measured improvements or regressions, blockers, and the next batch.
- Read-only status command: `.venv/bin/python scripts/track1_rollout_status.py`
  from the reconstruction directory. Logs and review results live in
  `incident_reports/track1_improvement/rollout/` and the isolated episode folders.

The local quality measures are diagnostics, not predictions of hidden-reference
leaderboard scores. Object accuracy improved substantially in the pilot, but its
interpenetration score worsened slightly; contact must remain part of the review.
