# CD-O gap against official CARI4D baseline

Verified the live Kaggle CD-O leaderboard through the authenticated CLI on 2026-10-14: CARI4D Basline 23.16246 cm; Javier Medel 34.15571 cm. Gap: 10.99325 cm (+47.4615% relative to baseline); reaching baseline requires a 32.1857% reduction from our current score. Our prior 75.48250 cm result was our own submission, not the official CARI4D baseline.

## Metric contract verified from official local kit

CD-O is the sum of both directed mean nearest-surface distances in 3D, in centimetres. Predicted object geometry is transformed by mesh scale, object rotation, and translation. One similarity transform is fitted from human alignment vertices at the first scored frame and applied to the predicted object too. Scores are averaged over frames within each episode and then equally across episodes. There is no independent per-frame object alignment to eliminate pose/scale errors. ACC-O compares translation second differences and does not directly assess rotations or mesh shape.

## Evidence and limitations

- Retained Episodes 1, 2, 12, 14 have selected local visible-object IoUs 0.00055, 0.01133, 0.06578, 0.06824. These are 4/30 equally weighted episodes. Their alternatives had insufficient alignment (1) or worse sampled surface acceleration (2,12,14). Retention traded potential spatial gains for motion conservatism. This does not quantify their hidden-reference CD-O contributions.
- Selected Episodes 19,20,10,26,29 have local IoUs 0.452,0.425,0.500,0.496,0.492. They improve on our old results but remain uncertain spatial fits. Rollers 19/20 have 57/23 interframe rotation jumps over 45 degrees. Symmetry can make some jumps geometrically equivalent, so jump counts alone are not a CD-O attribution.
- SAM3D metric meshes are generated from a single reviewed image plus aligned monocular depth, then reused per object group. Structural/scale sanity checks and silhouette agreement do not validate hidden surfaces or absolute metric dimensions against reference geometry.
- Refinement fixes object rotation at the CoCoNet prediction in this workflow. Translation/silhouette improvements therefore cannot repair incorrect orientation.
- Human-dependent first-scored-frame alignment couples body and object spatial error. Human CD-H regressed from 14.99887 to 15.43955; an alignment contribution is plausible but not established by this aggregate change.
- No per-episode reference CD-O scores or baseline prediction artifacts were obtained in this analysis. Exact attribution of the 10.99 cm gap is unavailable. Our custom CARI4D-based run is not a verified reproduction of the official baseline recipe.

## Next experiments

1. Audit an exact official baseline reproduction and its inputs, checkpoint, calibration, geometry, and postoptimization configuration before claiming an improvement over CARI4D.
2. Diagnose 1,2,12,14 and moderate-fit 19,20,10,26,29 first. Compare static multi-frame scale/depth/center consistency and human-object relative placement, not only silhouettes or acceleration.
3. Separate rotation/geometry validation from translation ACC-O. Resolve symmetry-aware pose selection and pan handle orientation; validate any orientation refinement in isolated outputs.
4. Use controlled ablations separating human prediction, object trajectory, and mesh changes. Evaluate locally where permitted reference data exist; use remaining authorized submission budget deliberately rather than blindly expanding all changes.

Leaderboard: https://www.kaggle.com/competitions/v2d-challenge-track1-cd-o/leaderboard
Metric source: reconstruction/v2d_submission_kit/v2d_submission_kit/v2dlb/mhr_metrics.py and metric_code/track_1/CD-O.py.
