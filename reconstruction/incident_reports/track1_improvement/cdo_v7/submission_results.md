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
