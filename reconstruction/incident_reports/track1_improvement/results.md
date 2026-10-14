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
