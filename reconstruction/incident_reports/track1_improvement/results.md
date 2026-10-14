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

---

# Fourth Track 1 submission results

All five metric submissions completed. Object reconstruction and object acceleration improve; human metrics are unchanged. Penetration regresses slightly. All metrics are lower-is-better.

| Metric | Previous v3 | New v4 | Change | Submission ID |
|---|---:|---:|---:|---:|
| cd_h_cm | 15.43955 | 15.43955 | +0.00% | 57196175 |
| cd_o_cm | 34.15571 | 28.45924 | -16.68% | 57196176 |
| acc_h_cm | 1.15894 | 1.15894 | +0.00% | 57196178 |
| acc_o_cm | 1.10054 | 0.88239 | -19.82% | 57196179 |
| interpenetration_cm | 0.00318 | 0.00331 | +4.09% | 57196182 |

CD-O improves by 16.68%, from 34.15571 to 28.45924 cm. The previously verified CARI4D baseline remains 23.16246 cm: the gap is 5.29678 cm, requiring another 18.61% reduction from v4. This iteration does not beat that baseline.

Object changes were promoted for Episodes 2, 12, 14, 19 and 20. Episode 1 retains v3 after failed penetration validation; the remaining 24 episodes also retain v3. All 30 episodes' human pose, scale and shape arrays are exactly identical to v3, as expected from the unchanged official human scores.

The signed local check used the predicted MHR human and a 4000-vertex proxy. Improvements in that check did not guarantee official PEN: the new score is 0.00331 versus 0.00318, a +4.09% regression. This distinction must remain a limitation of the selection procedure. Episode 2 also retains an unresolved unsigned hand-proximity warning.

Package: `/home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v4.parquet`. SHA256 `29ee18c4ec3e9343d1869d3d41c8b46942e9738443303d6ad0fdda5987b9dedb`; 30 episodes / 740,780 rows / 15,757,529 bytes. Producing code: `https://github.com/JavierMedel/video_to_data/commit/6be2d01585ddd1f98fa52c0fc25534621afc1493`. All original 60 v3 artifacts match their recorded hashes. Official local validation, authenticated preflight and upload dry-run passed. One upload version was issued across five metrics; no duplicate uploads.

Exact receipt: `submission_receipt.json`; per-artifact validation: `candidate_validation.json`; local review and selection: `final_review.md` and `promotion_decisions.json`.

Before/after video: `data/v2d_challenge/track_1/cdo_v4_experiments/review_videos/episode_000012/episode_000012_v3_vs_v4.mp4`. Matching close-ups: `episode_000012_v3_vs_v4_object_closeups.mp4`. Both contain 405 frames at 30fps and were decoded completely; see `episode_000012_video_validation.json`.

---

# Track 1 improvement results

All five leaderboards accepted and scored the third submission. Lower is better for all metrics.

| Metric | Second submission | Third submission | Change | Submission ID |
|---|---:|---:|---:|---:|
| cd_h_cm | 14.99887 | 15.43955 | +2.94% | 57188874 |
| cd_o_cm | 75.48250 | 34.15571 | -54.75% | 57188875 |
| acc_h_cm | 1.55253 | 1.15894 | -25.35% | 57188879 |
| acc_o_cm | 5.43410 | 1.10054 | -79.75% | 57188881 |
| interpenetration_cm | 0.00233 | 0.00318 | +36.48% | 57188885 |

23 newly reviewed episode predictions were included. Episodes 1, 2, 12, and 14 retain the second-submission predictions; Episodes 6, 18, and 28 reuse the prior approved pilot predictions. All 27 reruns were evaluated.

Object reconstruction and both motion metrics improve. Human reconstruction and interpenetration regress; local silhouette and unsigned contact proxies did not guarantee hidden-reference improvements. Prior submission packages remain preserved. No duplicate uploads were made.

Official package: `data/v2d_challenge/track_1/submission_packages/track1_improvement_rollout_v1.parquet` (740,780 rows, 30 episodes). SHA256: `02ee693389aa310a74dbcd0cf2a968d67c277c39a12129319b757c90ee256ad8`. Producing code and review commit: `246ecc37927ec36278b7c855c071accdc635dba2`. Exact upload IDs and terminal states are in `submission_receipt.json`; validation and per-artifact hashes are in `candidate_validation.json`.
