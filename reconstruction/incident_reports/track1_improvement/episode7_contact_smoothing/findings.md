# Episode 7 contact-constrained smoothing

The constrained fitter produces a candidate that passes all existing local image/depth and hand-interaction gates. It smooths 377 of 777 video frames, falling back to submitted v6 elsewhere. The fit considers both full anatomical-hand and FPS512 mean penetration at every video frame, with a bounded proximity change and a 3 cm translation bound.

| Local diagnostic | Submitted v6 | Candidate | Change |
|---|---:|---:|---:|
| Translation acceleration proxy (cm) | 1.765503 | 1.229681 | -30.35% |
| Sampled-surface acceleration proxy (cm) | 3.466227 | 3.053195 | -11.92% |
| Full-hand penetration, unaligned (cm) | 0.0017015244 | 0.0016160792 | -5.02% |
| FPS512 penetration, unaligned (cm) | 0.0011671393 | 0.0010782925 | -7.61% |
| Visible silhouette IoU | 0.705783 | 0.705500 | -0.000284 |
| Common-pixel cached depth error (cm) | 9.548092 | 9.412301 | -0.135791 cm |

The image/depth, full-hand/FPS512 penetration and grasp-proximity gates all pass on all 597 scored frames. The p95 increase in the sampled grasp-gap proxy is 0.0202 cm. Unlike pure smoothing in the previous experiment, the candidate retains a motion improvement while passing the penetration guard. This is evidence about local predicted geometry, not official leaderboard improvement.

Every field except object translation is identical to v6, including human pose/shape/scale, object rotations, and object scale. The mesh hash is unchanged. Individual translations move at most 2.9947 cm. There is no constant ray-depth shift; the mean frame-depth change is -0.1464 mm. The first/last two frames retain v6. The original 48 rotation jumps above 45 degrees remain and are outside this experiment's scope.

Five solver tests pass, including an exhaustive feasible-path oracle and a check that the fixed-rotation surface objective reduces exactly to the centroid objective plus a path-independent term. Syntax and diff checks pass. The fit's all-frame contact costs are checkpointed and fingerprinted. Final scored-frame hand checks recompute the candidate independently; the control uses the preserved hash-matched reference summary. These are checks on the same video used for fitting, not held-out evaluation.

Both exact comparison videos decode completely: 777 frames each at 30 fps. Six source-frame mask comparisons and six actual-video object close-ups (frames 90, 209, 328, 448, 567 and 686) have been visually inspected. Gross alignment, object size and hand interaction remain comparable to v6 in the inspected floor, pickup and held-object views. The midpoint object is occluded, so it cannot establish object quality. This is sampled visual review; full decoding does not mean every frame was visually assessed. See `visual_review.json` for scope and image hashes.

The complete runner was rerun successfully from its saved contact-cost cache. Candidate arrays matched exactly, both scored-frame reviews were reused after their hash checks, and the completed videos were retained. No expensive contact fitting or full-frame hand calculation was repeated during the resume check.

Candidate: `data/v2d_challenge/track_1/episode7_contact_smoothing/episode_000007/episode_000007.npz` and `episode_000007_object.glb`. Exact metrics, hashes, source-preservation result and video validation are in `status.json`. The reproducible command is in `execution_plan.md`.

No package, promotion or upload has been performed. Submitted v6 remains the reconstruction control. Official CD-O and ACC-O gains cannot be inferred from these proxies.
