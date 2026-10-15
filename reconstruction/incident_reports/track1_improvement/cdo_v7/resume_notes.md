# V7 sweep complete

All 30 episodes audited/tested; only Episode 7 passed all local checks and entered one v7 submission. All five official scores are complete. CD-O regressed 23.48489 -> 23.61430 cm; ACC-O improved 0.81013 -> 0.77219 cm. Human metrics and PEN are unchanged. See submission_results.md, score_comparison.json and episode_decisions.md.

Keep v6 as the reconstruction control. Do not automatically fork v7 for future CD-O experiments. A future controlled experiment should separate translation denoising from depth shifts; the cached monocular-depth improvement did not predict improved official CD-O. No further experiment or upload is active.

V7 archive and exclusive receipt are under data/v2d_challenge/track_1/submission_packages/track1_cdo_v7*. Preserve them. Never re-upload this version blindly; poll its receipt if needed. The video has 777 frames at 30fps. All original v6 artifacts and all human arrays are preserved.

Code commits: 408a62f6 and producing commit 6facea9b1432998092dfba4408564338af2441db, pushed to v2d-track1-blackwell. Final result reports follow in the results commit. Unrelated pre-existing rollout inventory JSON files remain untracked.
