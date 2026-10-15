# Seventh Track 1 submission results

The all-episode sweep is complete, with mixed official results: object acceleration improves 4.68%, while object reconstruction worsens 0.55%. **Keep v6 as the best CD-O reconstruction result.** Penetration and both human metrics remain unchanged. Lower is better for every metric.

| Metric | Previous v6 | New v7 | Change | Submission ID |
|---|---:|---:|---:|---:|
| cd_h_cm | 15.43955 | 15.43955 | +0.00% | 57215485 |
| cd_o_cm | 23.48489 | 23.61430 | +0.55% | 57215489 |
| acc_h_cm | 1.15894 | 1.15894 | +0.00% | 57215493 |
| acc_o_cm | 0.81013 | 0.77219 | -4.68% | 57215494 |
| interpenetration_cm | 0.00325 | 0.00325 | +0.00% | 57215499 |

V7 CD-O increases by 0.12941 cm, so it does not advance the reconstruction target. Best CD-O remains v6 at 23.48489 cm versus CARI4D baseline 23.16246 cm: gap 0.32243 cm, requiring another 1.37% reduction. V7 is useful for its object-motion score, not as a replacement for v6 reconstruction.

All **30 episodes** were audited and tested with bounded mesh-scale, ray-depth and temporal translation candidates. Fourteen passed the disjoint-frame geometry screen; sixteen retained v6. Thirteen contact-aware correction trials produced three sampled survivors, plus one initial survivor. Full interaction review rejected Episodes 0 and 26 on proximity. Episode 5 passed hand/contact checks but lost its meaningful improvement after correction. An additional geometry-specific Episode 0 hoop trial had no feasible path within the unchanged clearance/proximity constraints. See `episode_decisions.md` for every episode.

Only **Episode 7 (iron)** changed in the submitted package: translation smoothing and a net +0.07 m camera-ray depth shift. All 30 human pose, shape and scale arrays, every object rotation, all object scales/meshes, and the other 29 episode pairs remain identical to v6.

Episode 7 passed actual-mesh hand penetration and proximity review on all 597 scored frames, plus image/depth review on all 597 frames. Common-pixel cached depth error decreased 9.53994 -> 5.68788 cm over 537 supported frames; held-out depth error decreased 9.24343 -> 4.23253 cm. Surface acceleration proxy decreased 3.46623 -> 2.66645 cm. These diagnostics predicted smoother motion, but **did not predict improved official CD-O**. This iteration does not establish that the ray-depth shift is correct; separating translation denoising from depth shifts is the next controlled ablation. No additional upload was made to chase the score.

Local hand diagnostics use full anatomical hands and FPS512, not exact host hand-role IDs or hidden alignment. Official PEN remains 0.00325. Both exact comparison videos decoded completely: 777 frames at 30 fps. Six actual-video close-ups and six mask overlays were inspected, including visible floor/held-object views; the midpoint object is largely occluded.

Validation: five temporal-selection tests passed, real-data selector and real-mesh tube-bound regressions passed, syntax/diff checks passed, all 30 pairs and 60 control artifact hashes validated, official preflight and dry-run passed. One v7 version was uploaded across five leaderboards; every score completed and the exclusive receipt is preserved.

Package: `/home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v7.parquet` — 740,780 rows, 30 episodes, 15,707,940 bytes. SHA256 `121b97736636aa88ec9d6a0f0fc40e6919415ddd37377e1b2c0169937486cafe`. Producing code: `https://github.com/JavierMedel/video_to_data/commit/6facea9b1432998092dfba4408564338af2441db`.

Video: `data/v2d_challenge/track_1/cdo_v7_experiments/review_videos/episode_000007/episode_000007_v6_vs_v7.mp4`; close-ups: `episode_000007_v6_vs_v7_object_closeups.mp4`. Read-only CLI: `reconstruction/.venv/bin/python reconstruction/scripts/track1_cdo_v7_status.py` from the workspace root.

This sweep and submission are complete. Use **v6 as the next reconstruction control**, and retain v7 as the object-motion comparison. The CARI4D CD-O baseline remains unbeaten.

---

# Sixth Track 1 submission results

All five leaderboard submissions completed. Episode 1 hoop reconstruction improves CD-O by 16.56% and ACC-O by 6.00%. Penetration and both human metrics remain unchanged. Lower is better for every metric.

| Metric | Previous v5 | New v6 | Change | Submission ID |
|---|---:|---:|---:|---:|
| cd_h_cm | 15.43955 | 15.43955 | +0.00% | 57201702 |
| cd_o_cm | 28.14623 | 23.48489 | -16.56% | 57201706 |
| acc_h_cm | 1.15894 | 1.15894 | +0.00% | 57201708 |
| acc_o_cm | 0.86183 | 0.81013 | -6.00% | 57201712 |
| interpenetration_cm | 0.00325 | 0.00325 | +0.00% | 57201713 |

CD-O decreases by 4.66134 cm. The CARI4D baseline target is 23.16246 cm: the remaining gap is 0.32243 cm, requiring another 1.37% reduction. The baseline is not yet surpassed.

Changed only Episode 1 object geometry/trajectory, adopting the existing metric circular hoop with image-fitted orientation and adding a bounded, smooth camera-ray depth correction. Three preserved candidates and a constant depth correction failed the corrected hand check and were rejected. The final proposal constrains full predicted hands using a conservative mesh tube envelope, then passes independent actual-mesh generalized-winding and triangle-distance checks on all 174 scored frames: zero local penetration for both full anatomical hands and FPS512, matching control. No hidden ground truth is used. All 30 human arrays and the other 29 episode pairs are identical to v5.

Image overlap is 0.629370 across all 174 scored frames, and large rotation jumps fall from nine to zero. This overlap is a local diagnostic, not official CD-O. Exact host hand-role indices and alignment are unavailable. The control hoop has no contact-like frames, so the proximity guard cannot establish grasp accuracy. Official PEN is unchanged at 0.00325.

Validation: five temporal-selection tests passed, syntax/diff checks passed, all 30 artifact pairs and 60 control artifact hashes validated, official preflight and dry-run passed. Both exact reconstruction videos were decoded completely: 668 frames at 30 fps. Visual samples cover six mask comparisons and the reconstruction close-up at frame 334. One new version was uploaded across five metrics; the exact receipt is preserved.

Package: `/home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v6.parquet` — 740,780 rows, 30 episodes, 15,707,998 bytes. SHA256 `eb014ea2b52fb3faace7ae71209cec512bd528fbd8a984b2d09ee45c1cc76a14`. Producing code: `https://github.com/JavierMedel/video_to_data/commit/f53d46d3b4220a216c258b5306c438721a20653b`.

Episode 1 video: `data/v2d_challenge/track_1/cdo_v6_experiments/review_videos/episode_000001/episode_000001_v5_vs_v6.mp4`; matching close-ups: `episode_000001_v5_vs_v6_object_closeups.mp4`. See `episode_000001_video_validation.json`.

This iteration is complete. Further leaderboard experiments should target the remaining 0.32243 cm CD-O baseline gap with object depth/geometry refinements. No additional submission was issued in this iteration.

---

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
