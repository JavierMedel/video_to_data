# All-episode contact-constrained smoothing campaign

The user explicitly selected applying contact-constrained smoothing to all 30 episodes before a new submission. The campaign starts from submitted v6, preserving all 60 source NPZ/GLB hashes. Episode 7 reuses its completed, independently checked candidate. Every other episode is fitted into its own output directory. Episode 16 is processed as a new isolated candidate under this all-30 authorization; its original reconstruction is not rerun or modified.

For each episode, the fixed seven-frame quadratic translation smoother supplies the second option alongside unchanged v6. Every frame/option receives actual-mesh full anatomical-hand and FPS512 penetration checks and a sampled proximity calculation. A feasible smoothed option cannot increase either per-frame mean penetration, cannot add more than 0.5 cm to a control contact-like gap, and cannot move more than 3 cm. The first/last two frames remain v6. A second-order dynamic program selects the feasible sequence minimizing squared sampled-surface acceleration plus a small fidelity penalty. There is no constant ray-depth shift, mesh/scale change, object-rotation change or human-parameter change.

The final hand review reuses these exact frame/option evaluations after verifying the selected translations and source hashes: the discrete fit creates no additional geometry. Reports explicitly label this reuse and do not claim an independent hand rerender. Max vertex penetration and vertexwise change are unavailable from the small summary cache and are null. Full scored-frame silhouettes and common-pixel cached depth are evaluated separately. These checks use the fitting video and are not held-out evidence or official scores.

Existing gates are preserved. Image alignment must remain acceptable, cached depth must pass, full-hand/FPS512 means must not worsen, proximity must pass, and a meaningful gain must remain. The existing meaningful-gain test accepts at least 0.005 mean IoU gain, at least 1 cm cached-depth gain, or at least 5% sampled-surface acceleration-proxy reduction. A small translation-only gain is insufficient by itself. Failed candidates are retained for audit and their episodes retain submitted v6 in any eventual package.

Final selection also recomputes motion proxies using only consecutive triples of the public scored-frame indices, without bridging gaps. This matches the evaluator's temporal inclusion rule; it does not supply hidden-reference acceleration errors. Both the original all-video screen and the same meaningful-gain threshold on scored triples must pass. The previous 10% motion-regression guards apply to translation and sampled surfaces on both frame scopes. This additional conservative screen can reject provisional all-video passes (for example Episodes 12 and 17); it never rescues a failed original screen. `track1_contact_campaign_review.py` records both scopes and visual-review status in `selection_review.json` without altering fit artifacts or original diagnostics.

Two independent GPU processes run at once. Every eight frames checkpoint the contact summaries, input fingerprints and progress. Completed episodes are verified and reused on resume. Original Episode 7 results remain in their original directory. Root progress is `campaign.json`; each episode has `status.json`, `run.log`, masks and object crops. Six scored source frames per episode supply matching full-scene overlays and object close-ups for visual review.

Kernel fusion is enabled for fresh caches after matching eager depths on six meshes, including open/reversed cases and three dense real meshes (103,772, 352,796 and 757,976 faces). The observed maximum depth difference was zero on that test set; measured speedup was 1.75 under concurrent load. The fused checker retains the original winding threshold, float64 accumulation, uncertain-boundary float64 fallback, AABB screen and triangle distances. A per-episode evaluator manifest and fused-source hash bind new caches to their implementation. Legacy caches continue with eager evaluation. See `fused_validation.json`.

Run from the workspace root:

```bash
reconstruction/.venv/bin/python reconstruction/scripts/track1_contact_campaign.py --workers 2
```

The runner holds an exclusive process lock and resumes without duplicating live campaigns. Read-only progress: add `--status`. Tests: eight solver/contact-summary/scored-gap cases pass; six mesh/depth equivalence cases pass. The authenticated official environment preflight passed for all five Track 1 leaderboards before the campaign started. Official packing and uploader dry-run still require the final accepted episode set and a producing-code commit.

No packing or upload occurs in this runner. All 30 episodes must finish and passing candidates must receive visual review before assembling a proposed complete package. Local improvements cannot establish a better official CD-O score.
