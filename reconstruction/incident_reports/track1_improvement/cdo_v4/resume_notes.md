# Track 1 fourth-submission handoff

The six-episode experiment batch and all review gates are complete. Promoted object predictions: 2, 12, 14, 19, 20. Episode 1 retains v3 after its signed penetration check failed. All 30 human pose/scales/shape arrays match scored v3 exactly. Seven mathematical tests, official pack validation, authenticated preflight and upload dry-run passed.

Producing commit: 6be2d01585ddd1f98fa52c0fc25534621afc1493 (pushed).

Package: `data/v2d_challenge/track_1/submission_packages/track1_cdo_v4.parquet`, 740,780 rows / 30 episodes, SHA256 `29ee18c4ec3e9343d1869d3d41c8b46942e9738443303d6ad0fdda5987b9dedb`.

All five metric submissions completed; no current-batch work is pending. One upload command was issued. Preserve the exclusive receipt. Never repeat the upload after an interruption. Query only the recorded IDs using the official receipt mode:

```bash
cd /home/alien51/video_to_data/reconstruction/v2d_submission_kit/v2d_submission_kit
/home/alien51/video_to_data/reconstruction/.venv/bin/python submit/v2d_submit.py --track track_1 --poll-receipt /home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v4.kaggle-receipt.json
```

Current state is in `submission_status.json`. Final score comparison is recorded in `score_comparison.json` and `submission_results.md`. Official v3 comparison: CD-H 15.43955, CD-O 34.15571, ACC-H 1.15894, ACC-O 1.10054, PEN 0.00318. All lower is better. Previously verified baseline CD-O: 23.16246.

Exact Episode 12 v3/v4 reconstruction and close-up videos: `data/v2d_challenge/track_1/cdo_v4_experiments/review_videos/episode_000012/`. Both contain all 405 frames at 30fps and were decoded for validation. Original data and packages are preserved.

Latest scores: CD-H 15.43955, CD-O 28.45924, ACC-H 1.15894, ACC-O 0.88239, PEN 0.00331. Current iteration is complete. The next iteration should resolve Episode 1 physical intersections and official penetration regression before aiming at the remaining baseline gap.
