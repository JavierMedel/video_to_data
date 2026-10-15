# All-episode contact-constrained smoothing

All 30 episodes completed without execution failures, covering 16,563 video frames. Episode 7 reused its completed validated candidate; the other 29 received isolated fits. Every submitted v6 source hash is preserved.

Final selection promotes only translations for Episodes **1, 2, 6, 7, 8, 9, 11, 14, 22, 24, 26 and 29**. The other **18 episodes retain v6**. All 30 staged NPZ/GLB pairs pass integrity, finiteness, rotation, scale and scored-frame coverage checks. Every human field, object rotation, object scale and mesh is identical to v6.

| Selected complete-package local proxy | v6 control | Candidate | Reduction |
|---|---:|---:|---:|
| Translation acceleration magnitude (cm) | 0.827185 | 0.704264 | 14.86% |
| Sampled-surface acceleration magnitude (cm) | 2.285616 | 2.203472 | 3.59% |

These aggregate magnitudes weight 9,747 consecutive scored triples, retaining the control values for rejected episodes. They are neither hidden-reference acceleration errors nor official leaderboard scores. Per-episode scored and all-video diagnostics are preserved in `selection_review.json` and summarized in `episode_decisions.md`.

Sixteen candidates pass the original all-video numerical screen. Four provisional passes (Episodes 10, 12, 17 and 27) fail the additional meaningful-gain check on consecutive scored triples and retain v6. Both temporal scopes and the original image/depth/hand checks must pass. The twelve remaining candidates have sampled source-image/mask/mesh-contour review, with image hashes and specific observations in `visual_reviews.json`. Existing outline/geometry errors remain; the reviewer found no gross new displacement in those samples. These reviews are not complete reconstructed-human videos. Episode 7 additionally retains its previously decoded full comparison videos.

All-frame exact mesh checks evaluate original and smoothed options before selection. The selected trajectory introduces no additional pose option, so its hand review reuses the measured option-level values after hash and array checks. This reuse is explicit; it is not an independent hand rerender or held-out validation. Unknown host hand-role IDs, alignment scale and unavailable vertexwise deltas remain limitations of the local PEN diagnostics.

Eight solver, contact-summary and scored-gap regression cases pass. The fused evaluator matches eager depths on six analytic/real mesh cases, including open/reversed winding and dense surfaces; maximum observed depth difference is zero in that test set. The fused arithmetic preserves the original threshold and float64 fallback. All completed artifacts and source fingerprints were verified by the campaign runner. No existing submission artifact is overwritten.

The complete 30-episode candidate is staged under `data/v2d_challenge/track_1/contact_smoothing_v8/submission_episodes/`. It covers the official 740,780-row template. Official packing, public producing-code provenance and uploader dry-run are recorded separately in `package_validation.json` and the submission logs. The recorded v6 official scores remain the reconstruction control until a new version is scored.
