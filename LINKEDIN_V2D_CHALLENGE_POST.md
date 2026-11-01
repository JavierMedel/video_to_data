# Video to Data (V2D) Challenge — LinkedIn draft

Author: Javier Medel, MSc.

Prepared: November 1, 2026.

Angle: an independent challenge entry with a concrete motion-error improvement, a real visual example, and an honest next experiment.

## 1. Completed facts

- **Challenge / project:** NVIDIA’s Video to Data (V2D) Challenge 2026. Independent work across reconstruction, robotic grounding, and egocentric transfer.
- **My contribution:** assembled and exercised the workflow, reviewed masks and geometry, implemented contact-constrained translation smoothing, and preserved evaluation and failure evidence. The underlying vision models and simulation tools are upstream work.
- **NVIDIA technology:** the NVIDIA V2D toolkit and FoundationPose for object pose estimation/tracking. Other pipeline components include Grounding DINO, Detectron2, SAM2, MoGe, SAM3D, and CARI4D; do not attribute all these models to NVIDIA.
- **Problem:** videos show what a person does, while robot learning needs usable geometry, motion, and contact information.
- **Input → output:** third-person video → reconstructed human–object interaction; supplied motion trajectories → robot-policy evaluations; egocentric video → hand/object reconstruction and downstream policy evaluation.
- **Selected verified result:** Track 1 ACC-O = **0.71908 cm** for `track1_contact_smoothing_v8.parquet`, submitted October 15, 2026; submission **57237210**. Lower is better.
- **Comparison:** my v6 baseline scored **0.81013 cm** on the same 30-episode cohort. Absolute reduction: **0.09105 cm**. Relative reduction: **11.24%**, calculated as `(0.81013 − 0.71908) / 0.81013 × 100`.
- **Evaluation scope:** official public scoring, 30 episodes, episode-mean aggregation under the released evaluator. Twelve episodes adopt reviewed translation corrections; eighteen retain v6. This is one version comparison, not a repeated-run confidence interval. No hardware or inference-speed comparison is claimed.
- **Limitation:** saved pose trajectories still have discontinuities. Track 2 Tier 3 has no scored submission in the saved October 25 snapshot. Smoothing does not establish accurate geometry or successful robot behavior.
- **Baseline boundary:** the comparison is against my own v6 package. The recorded CARI4D object-distance reference remains better than v8 on the separate CD-O metric: 23.16246 cm versus 23.48324 cm. Do not present the ACC-O improvement as an overall win over CARI4D.
- **Visual:** the real Episode 1 CARI4D comparison, showing initialization, CoCoNet prediction, and contact-guided refinement. This early stability run is separate from the v8 scoring packet. Website derivative: `website/public/media/t1-comparison.mp4`.
- **Resources:** public code, project website, official challenge, upstream toolkit, dataset, and the preserved comparison report are in the first-comment draft below.
- **Hashtags:** use the general topic tags in the post. A dedicated official challenge hashtag was not confirmed from the primary sources checked, so none is presented as official.

### Evidence used

- `reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/submission_results.md`
- `reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/score_comparison.json`
- `reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/campaign_verification.json`
- `reconstruction/incident_reports/cross_track_validation_plan/snapshots_20261025_completion_v1/summary.json`
- `reconstruction/v2d_submission_kit/v2d_submission_kit/metric_code/track_1/ACC-O.py`
- `website/asset-inventory.json` and `website/src/data/project.ts`

## 2. Selected hook

**Your challenge entry:** lead with what I built, name the challenge immediately, and use the verified motion-error improvement as the central result. This is a project update, not an NVIDIA release announcement.

## 3. Copy-ready LinkedIn post

🦾🤖 I built a video-to-data research workflow for NVIDIA’s Video to Data (V2D) Challenge—turning human demonstrations into inspectable 3D reconstructions and robot-policy evaluations.

The goal: understand what survives the journey from watching a person move an object to teaching a robot.

Using NVIDIA’s V2D toolkit and FoundationPose, I worked across reconstruction, robotic grounding, and egocentric transfer. My changes include contact-constrained motion smoothing.

→ Detect and segment people and objects.
→ Reconstruct surfaces and estimate motion over time.
→ Inspect projections, check contact, and evaluate outputs.

The featured clip shows initialization, prediction, and refinement of a generated human-and-object reconstruction. This early example is separate from the scored result below.

📊 On October 15, my Track 1 ACC-O score reached 0.71908 cm, versus 0.81013 cm for my v6 baseline, across the same 30-episode cohort.

That’s an 11.24% reduction in the official object-motion error metric. Lower is better.

My main lesson: smoother motion still needs geometry and contact checks.

There’s more work ahead: saved poses can jump, and Track 2’s third tier remains unsubmitted. Next, I want to test reinitialization around those jumps.

For robotics researchers, I’m sharing code, real outputs, and failure diagnostics so others can inspect the trade-offs.

🔗 Project and resources in the comments.

Which would you test first: occlusion handling or recovery after a pose jump?

#NVIDIA #RobotLearning #ComputerVision

**Post word count: 220**, counting whitespace-separated words in the post body, including emoji and hashtags. Section headings, comments, and preparation notes are excluded.

## 4. First comment — resources

🔗 Resources for my Video to Data (V2D) Challenge project:

• Project / code: https://github.com/JavierMedel/video_to_data/tree/v2d-track1-blackwell

• Interactive examples and results: https://javiermedel.github.io/video_to_data/

• Featured reconstruction clip: https://javiermedel.github.io/video_to_data/media/t1-comparison.mp4

• Official challenge: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/

• NVIDIA V2D toolkit: https://github.com/nvidia-isaac/video_to_data

• Dataset: https://huggingface.co/datasets/nvidia/video_to_data_challenge

• My v6 → v8 evaluation report: https://github.com/JavierMedel/video_to_data/blob/035bed6af955a59cf9747f1e4f3d20bbd764b52a/reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/submission_results.md

To explore locally: use Node.js 22.12+, then `cd website`, `npm ci`, and `npm run dev`. Reconstruction and policy learning have separate GPU, model-weight, and dataset requirements; follow the repository documentation.

This is an independent project. NVIDIA’s toolkit and the upstream model authors receive credit in the project resources.

## 5. Optional second comment — evaluation details

A little more context on the result:

• Evaluated: the same 30 Track 1 episodes in official public scoring. Scores were submitted October 15 and remain recorded in the October 25, 2026 snapshot.

• Metric: ACC-O compares predicted and reference second frame differences of object translation. Lower is better. The scoring implementation reports cm, without multiplying by source FPS squared; these values should not be relabeled cm/s².

• Baseline: my v6 package, 0.81013 cm. Candidate: my contact-constrained v8 package, 0.71908 cm. Difference: −0.09105 cm, or −11.24%.

• Setup: twelve episodes use reviewed translation corrections; eighteen retain v6. Other packed numeric fields remain identical to v6. Producing code: `b29d19010a14c39c5402ae7a8d0420bf6e0c925e`. This comparison does not report hardware throughput, repeated-trial uncertainty, or an overall leaderboard rank.

• Main failure: pose discontinuities remain in saved diagnostics. The featured CARI4D video is an early illustrative output, not a visual comparison of the scored v6 and v8 packets. Geometry and robot-policy success require separate evaluation.

The exact score comparison and submission provenance are in the evaluation report linked in the first comment.

## 6. Visual plan

### Recommended: the existing reconstruction video

1. Attach `website/public/media/t1-comparison.mp4` before publishing the post. It is an existing browser-friendly derivative, not a newly generated or improved result.
2. Show the generated reconstruction immediately. Keep all three original panel headings visible: initialization, CoCoNet prediction, and contact-guided refinement. These are generated output views, not an input-versus-ground-truth comparison.
3. Put this caption outside the scientifically relevant image: **“Track 1 · Episode 1 · Early CARI4D reconstruction example. Separate from the v8 scored package.”**
4. If adding an end card, use a black background, white text, and green accents: **“ACC-O: 0.81013 → 0.71908 cm · −11.24% vs my v6 · 30 episodes · lower is better.”** Label it as an aggregate scoring result, separate from the example video.
5. Add one readable limitation: **“Pose jumps remain. Motion error, geometry, and robot success are evaluated separately.”** Keep Javier Medel’s independent-project identity and NVIDIA dataset attribution visible.

### Optional carousel

1. **Cover:** “Video to Data (V2D) Challenge” / “From Human Video to Grounded Data” / “Javier Medel, MSc. — independent project.”
2. **What went in / what came out:** a real source frame next to a clearly labeled generated output. Use matched frames only when alignment is verified. Otherwise, label the episode/run of each example separately.
3. **Workflow:** detection → segmentation; parallel depth estimation; mesh reconstruction → pose tracking → validation. Identify NVIDIA V2D and FoundationPose alongside upstream models.
4. **Measured result:** the exact ACC-O comparison, units, cohort, date, and own-v6 baseline.
5. **Failure and resources:** the real frame-212 pose diagnostic from `website/public/media/e15-failure.webp`, explicitly labeled Episode 15 and separate from the scored comparison. Close with the project name and the same technical question as the post.

Dataset imagery is attributed to NVIDIA under CC BY 4.0. Model/code licenses and NVIDIA trademarks remain separate. Preserve the original field of view and panel labels; avoid crops, interpolation, stabilization, or recoloring that changes the apparent experimental result.

## 7. Publication notes

- The copy-ready post contains no external URLs; paste the separate resource comment afterward.
- Attach the recommended clip, or revise the visual sentence to describe the actual media selected.
- Keep the v6 baseline, the 30-episode scope, and the October scoring date with the numerical claim.
- Keep the early visual example separate from the aggregate v8 result.
- Use the closing question already in the post; avoid adding a second call for comments.
- This file is a draft for manual publication. No LinkedIn post or comment has been sent.
