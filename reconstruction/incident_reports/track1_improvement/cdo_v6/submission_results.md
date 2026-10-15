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
