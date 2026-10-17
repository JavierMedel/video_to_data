# v10 contact-constrained rotation: local results

A complete30-episode candidate is staged. Only Episode18 object rotations change; all humans, meshes, object scales/translations and the other29 episode pairs are identical to submitted v8. The rejected temporal pilot and all earlier scored archives remain immutable.

| Episode18 diagnostic | v8 | Candidate |
|---|---:|---:|
| Visible IoU,420 scored frames |0.414821|0.437859|
| Common inferred-depth residual cm |13.78924|13.37159|
| Surface acceleration magnitude, all video cm |5.575357|4.851728|
| Surface acceleration magnitude, consecutive scored triples cm |5.323954|4.770283|
| >45 degree rotation jumps, all video |48|43|
| Full-hand penetration,420 scored frames cm unaligned |0.000786719|0.000709568|
| FPS512 penetration,420 scored frames cm unaligned |0.000562711|0.000517551|

Every selected frame passes per-frame full-hand/FPS512 non-increase and the0.5 cm contact-like gap bound. Those complete-frame results reuse exact finite-option fitting evaluations, explicitly not an independent hand rerender. A separate33-frame eager actual-mesh check passes: full-hand means0.000926545→0.000824035 cm and FPS512 means0.000547972→0.000490573 cm. Two535-frame videos decode fully at30fps. Six object crops and two full-scene frames were visually inspected; modest outline changes and inherited geometry mismatch are recorded in `visual_reviews.json`.

Episodes7 and20 are rejected by the fixed0.02 IoU-gain criterion. No candidate is promoted just because a less strict general image tool reports PASS. The first-order path and its failed motion outcome remain preserved separately from the final surface-regularized selector.

These are local predicted-geometry diagnostics, not official scores. Since human identity/poses and object translations are unchanged, the official CD-H, ACC-H and ACC-O inputs remain identical to v8. Official CD-O/PEN outcomes are unknown until scoring. The candidate has not yet been submitted. See `submission_policy_check.json` for the current weekly-limit discrepancy and `package_validation.json` for exact package provenance/checks.
