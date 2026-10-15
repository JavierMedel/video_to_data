# Resume v6 safely

Control: immutable v5 submission, CD-O 28.14623 cm. Only Episode 1 changes in this experiment.

Final proposal: `data/v2d_challenge/track_1/cdo_v6_experiments/hand_clearance_candidates/episode_000001`. Its actual-mesh full-scored-frame hand review passed. Earlier candidates and constant-depth probe failed; do not promote those.

Follow `pipeline_status.json` for remaining steps. Review videos under `cdo_v6_experiments/review_videos/episode_000001`; video helper refuses completed outputs. Before packaging, record visual evidence in `promotion_decisions.json`. Package with `scripts/track1_cdo_package.py --iteration 6 --commit HASH`.

Official upload must happen once, using a new exclusive `track1_cdo_v6.kaggle-receipt.json`. If it exists or any upload outcome is ambiguous, inspect/poll the receipt; never re-upload blindly. All 30 human arrays and 29 other episode pairs must remain identical to v5. Final results and Git push remain required.

All five official scores are COMPLETE. See `submission_results.md` and `score_comparison.json`. This v6 iteration is complete after the final results Git push; do not rerun packing/upload or overwrite these artifacts.
