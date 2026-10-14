# Fifth submission handoff

Promoted episodes: 16, 21, 23. Retained tested episodes: 13, 18. All 30 human parameter arrays match scored v4 exactly. Full scored-frame hand-penetration and contact-like proximity checks passed for every promoted episode. Nine tests passed, including numerical comparisons with the released PEN metric on real dense meshes.

Producing commit: 1c3a1efca3c049c8cff229f25163254e518558d3 (pushed).

Package: `data/v2d_challenge/track_1/submission_packages/track1_cdo_v5.parquet`, 30 episodes / 740,780 rows. SHA256 `03cff99bcb9bbb8f0d869226580fd0c1740eb4b2c0c31856f062c06c823be68c`.

Use `pipeline_status.json` for the latest stage. One controlled fifth version is authorized. Once its exclusive receipt exists, never repeat the upload; use only the official read-only receipt poll mode:

```bash
cd /home/alien51/video_to_data/reconstruction/v2d_submission_kit/v2d_submission_kit
/home/alien51/video_to_data/reconstruction/.venv/bin/python submit/v2d_submit.py --track track_1 --poll-receipt /home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v5.kaggle-receipt.json
```

Compare with v4: CD-H 15.43955, CD-O 28.45924, ACC-H 1.15894, ACC-O 0.88239, PEN 0.00331. Lower is better for all. Exact host hand-role indices and hidden positive alignment scale are unavailable; local checks are not official scores. See `penetration_definition_audit.md` and `v4_penetration_diagnosis.md` for the corrected explanation of the previous PEN regression.

COMPLETE: all five metric submissions scored. CD-H 15.43955, CD-O 28.14623, ACC-H 1.15894, ACC-O 0.86183, PEN 0.00325. No pending steps or GPU jobs remain in this iteration. Exact receipt is in `submission_receipt.json`; comparison is in `score_comparison.json`. Do not upload again.
