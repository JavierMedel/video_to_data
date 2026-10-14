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
