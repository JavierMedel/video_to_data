# Fifth Track 1 submission results

All five leaderboard submissions completed. Object reconstruction, object acceleration and penetration improve over v4. Both human metrics remain exactly unchanged. Lower is better for every metric.

| Metric | Previous v4 | New v5 | Change | Submission ID |
|---|---:|---:|---:|---:|
| cd_h_cm | 15.43955 | 15.43955 | +0.00% | 57197586 |
| cd_o_cm | 28.45924 | 28.14623 | -1.10% | 57197588 |
| acc_h_cm | 1.15894 | 1.15894 | +0.00% | 57197591 |
| acc_o_cm | 0.88239 | 0.86183 | -2.33% | 57197592 |
| interpenetration_cm | 0.00331 | 0.00325 | -1.81% | 57197593 |

CD-O improves by 1.10% (0.31301 cm), ACC-O by 2.33%, and PEN by 1.81%. The CD-O baseline target remains 23.16246 cm: the remaining gap is 4.98377 cm, requiring another 17.71% reduction from v5.

Audited all 24 remaining episodes and tested the five largest rotation-failure cases: 23, 18, 21, 16, 13. Promoted object predictions for **16, 21, 23**; retained v4 for 13 and 18 after contact/penetration rejection. All other episode pairs reuse v4. All 30 human pose, scales and shape arrays match v4 exactly.

The previous penetration check used the opposite containment direction. New review checks predicted hand vertices inside the submitted mesh with the released generalized-winding convention, plus contact-like proximity checks. Each promoted candidate passed every scored frame (240, 347, 399 frames for 16, 21, 23), using full anatomical hands and a local FPS512 subset. Exact host hand-role indices and hidden positive alignment scale remain unavailable, so local checks are not official PEN scores. The official improvement supports this correction on the tested batch; it does not prove universal equivalence. See `penetration_definition_audit.md` and `v4_penetration_diagnosis.md`.

Validation: nine tests passed (five temporal-selection tests and four direct PEN parity tests, including actual dense meshes), all 30 artifact pairs validated, all 60 original v4 hashes checked, official preflight passed, and the official dry-run passed. One v5 version was uploaded across five metrics with the exact receipt preserved; no duplicate uploads.

Package: `/home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v5.parquet` — 740,780 rows, 30 episodes, 15,757,311 bytes. SHA256 `03cff99bcb9bbb8f0d869226580fd0c1740eb4b2c0c31856f062c06c823be68c`. Producing code: `https://github.com/JavierMedel/video_to_data/commit/1c3a1efca3c049c8cff229f25163254e518558d3`.

Episode 21 before/after video: `data/v2d_challenge/track_1/cdo_v5_experiments/review_videos/episode_000021/episode_000021_v4_vs_v5.mp4`; matching close-ups: `episode_000021_v4_vs_v5_object_closeups.mp4`. Both contain all 563 frames at 30fps and were decoded completely; see `episode_000021_video_validation.json`.

This iteration is complete. Next priorities: revisit Episode 1 under the corrected hand-penetration definition, improve object depth/geometry, and close the remaining CD-O baseline gap. Further changes require another controlled experiment; no additional submission was issued in this iteration.
