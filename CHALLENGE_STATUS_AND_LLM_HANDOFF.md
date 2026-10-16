# V2D Challenge: complete submission status and cross-machine LLM handoff

**Status snapshot: 2026-10-16.** Live authenticated Kaggle retrieval: `2026-10-16T21:04:17.392752+00:00`. Repository/experiment state reviewed at base commit `0b966629` on branch `v2d-track1-blackwell`. This file is a committed, portable context packet for planning the next improvement round. Values describe this snapshot, not future standings.

## Contents

- [1. Read this first: the intended workflow](#1-read-this-first-the-intended-workflow)
- [2. Challenge scope, primary references, and freshness](#2-challenge-scope-primary-references-and-freshness)
- [3. Official Track 1 submission history](#3-official-track-1-submission-history)
- [4. Current live leaderboard comparison](#4-current-live-leaderboard-comparison)
- [5. What the metrics actually measure](#5-what-the-metrics-actually-measure)
- [6. V8 implementation, selection and exact scope](#6-v8-implementation-selection-and-exact-scope)
- [7. Experiments already tried, including rejections](#7-experiments-already-tried-including-rejections)
- [8. Execution machine specifications and current capacity](#8-execution-machine-specifications-and-current-capacity)
- [9. Filesystem, source map and artifact ownership](#9-filesystem-source-map-and-artifact-ownership)
- [10. Reproducible verification and execution commands](#10-reproducible-verification-and-execution-commands)
- [11. Evidence index for the next model/executor](#11-evidence-index-for-the-next-modelexecutor)
- [12. Open problems and useful planning directions](#12-open-problems-and-useful-planning-directions)
- [13. Required output from the planning LLM](#13-required-output-from-the-planning-llm)
- [14. How this snapshot was checked and how to refresh it](#14-how-this-snapshot-was-checked-and-how-to-refresh-it)

## 1. Read this first: the intended workflow

The user will take **this Markdown file to another laptop that can access another LLM**. That laptop is the **planning machine**. It will not run the challenge, reconstruction, GPU experiments, dataset processing, Docker workloads, or submissions. The other LLM must read this document and generate a **new detailed Markdown improvement plan**. The user will bring that plan back to the assistant in this repository on **this execution machine**, where implementation, experiments, validation, packaging, and any authorized submission will occur.

The planning model should produce a file named `V2D_NEXT_IMPROVEMENT_PLAN.md` (or an equivalent clearly named `.md`), not claim that any experiment has been executed. The executing assistant must reconcile the incoming plan with the actual current checkout, preserved artifacts, and user instructions before acting. This status file supplies evidence and constraints; it is not itself authorization to rerun every pipeline or upload another version.

The planning laptop need not have the dataset, model weights, CUDA, Docker, Kaggle credentials, or this machine's filesystem. Absolute paths below belong to the execution machine. GitHub contains source and committed reports; cloning it does **not** recreate the local data/cache/weights. If the planning model only receives this one file, all essential numerical findings and execution constraints are embedded here. Relative links provide optional deeper evidence when the repository is available.

### 1.1 Current outcome in one place

- Eight Track 1 candidate versions have been uploaded to all five metric competitions: **40 accepted, completed metric submissions**. The authenticated history currently lists eight submissions in each of the five competitions.
- Latest version: **v8, contact-constrained translation smoothing**. Complete package: **30 episodes, 740,780 rows**, all five scores COMPLETE. No scoring job remains pending for v8.
- V8 versus v6: **CD-O 23.48489 → 23.48324 cm**, **ACC-O 0.81013 → 0.71908 cm**, **PEN 0.00325 → 0.00322 cm**; human metrics unchanged. Main gain: **11.24% lower object acceleration error**. Spatial improvement is only **0.00165 cm, about 0.007%**.
- V8 improves or equals v6 on all five reported metrics and is the natural latest complete-package starting point. Preserve v6 as the immutable experiment control and v7 as a useful failed spatial/positive temporal comparison. Existing campaign scripts hardcode v6; passing a new campaign name alone does not rebase them onto v8.
- The official CARI4D leaderboard entry, spelled **`CARI4D Basline`**, still has better object distance: **23.16246 cm**. V8 is **0.32078 cm** worse, **1.385% above** that reference. Reaching it requires **1.366% reduction from v8**.
- The current best team on CD-O scores **17.91571 cm**. Matching it requires approximately **23.71% reduction from v8**. Beating CARI4D is a useful milestone, not equivalent to winning the live challenge.
- Our **best historical human distance** is **14.99887 cm from v2**, better than v8's 15.43955. Our **best historical PEN** is **0.00213 cm from v1**, better than v8's 0.00322. No claim is made that v8 dominates every older version.
- The Kaggle leaderboard displays the team's best scores separately per metric. Its row therefore mixes versions; it is **not the score vector of one reconstruction package**. Do not propose a mixed package by copying displayed scores: predictions, alignment and interaction must be revalidated.
- Only Track 1 has verified local challenge submissions in this work. This checkout has Track 2/3 toolkit support, but no local Track 2/3 dataset directories or completed submission artifacts were found in the inspected challenge workspace. This is a local inventory statement, not an audit of all possible external work by this user.

## 2. Challenge scope, primary references, and freshness

Official challenge website: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/ . Dataset: https://huggingface.co/datasets/nvidia/video_to_data_challenge . Repository used here: https://github.com/JavierMedel/video_to_data . Upstream: https://github.com/NVIDIA-ISAAC/video_to_data . Local toolkit: `reconstruction/v2d_submission_kit/v2d_submission_kit/`.

The official website was checked on 2026-10-16. It describes three tracks, a freeze on **November 4, 2026, 5:00 p.m. EST**, and per-metric Kaggle scoring. It says five submissions per week, with unlimited submissions during the last three days. The local toolkit also records a five-per-day Kaggle configuration. **API quota is not evidence that weekly policy permits more experiments.** Recheck the current official rules and actual remaining quota before any future upload. Do not infer future authorization from historical successful uploads.

The website says dashboard refresh can lag Kaggle. In this inspection, fetching its public `leaderboard.json` returned `generated: 2026-09-29T01:00:05Z`, the note `No submissions scored yet. Written by aggregate/empty_payload.py.`, and empty tables. The repository's `docs/v2d_challenge/leaderboard.json` is also an empty old payload. **Neither is current competitive evidence.** Current comparisons below came from authenticated, read-only Kaggle API calls, not from the stale JSON. No dashboard code or payload was modified for this handoff.

| Area | What it evaluates | Local status and scope of this handoff |
|---|---|---|
| Track 1: Reconstruction | Third-person monocular video to metric 4D human-object reconstruction; CD-H, CD-O, ACC-H, ACC-O, PEN | Active work; 30 episodes; 8 fully scored versions; full history below |
| Track 2: Robotic grounding, Tier 1 | Policy evaluation using supplied clean multiview trajectories | Toolkit/config present; no completed local run/submission established here |
| Track 2, Tier 2 | Policy evaluation from synthetically corrupted trajectories | Same inventory limitation |
| Track 2, Tier 3 | Policy evaluation from reconstructed trajectories | Same inventory limitation; Track 1 outputs may eventually support this, but no transfer result has been demonstrated |
| Track 3: Egocentric | Egocentric reconstruction and policy transfer, plus separate object-mesh CD-O competition | Toolkit and broader repo modules exist; no completed challenge submission established here |

Track 2 configured metrics: AUC, SP-SR, MP-SR (higher is better) and MPPE (lower is better). Track 3 additionally includes RPE and a separate object-reconstruction leaderboard. These details are context from local `config/tracks.json`, not evidence of local scores. The proposed next plan should focus on Track 1 unless the user explicitly expands scope.

## 3. Official Track 1 submission history

All five Track 1 metrics are **lower is better**, reported in centimetres by the released evaluator. This table records **actual package scores**, not mixed best-per-metric dashboard values. Versions v1–v8 are our local chronological labels, not official baseline versions.

| Version | Main change | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---:|---:|---:|---:|---:|
| v1 | Initial recovery/full package | 15.11851 | 152.72099 | 1.62217 | 7.10147 | 0.00213 |
| v2 | Initialization/refinement pilots 6, 18, 28 | 14.99887 | 75.48250 | 1.55253 | 5.43410 | 0.00233 |
| v3 | All-episode rollout; selective promotion | 15.43955 | 34.15571 | 1.15894 | 1.10054 | 0.00318 |
| v4 | Temporal/geometry fixes 2, 12, 14, 19, 20 | 15.43955 | 28.45924 | 1.15894 | 0.88239 | 0.00331 |
| v5 | Corrected hand checks; promote 16, 21, 23 | 15.43955 | 28.14623 | 1.15894 | 0.86183 | 0.00325 |
| v6 | Episode 1 metric hoop reconstruction | 15.43955 | 23.48489 | 1.15894 | 0.81013 | 0.00325 |
| v7 | Episode 7 smoothing + +0.07 m ray shift | 15.43955 | 23.61430 | 1.15894 | 0.77219 | 0.00325 |
| v8 | Contact-constrained smoothing: 12 selected of 30 | 15.43955 | 23.48324 | 1.15894 | 0.71908 | 0.00322 |

### 3.1 Interpretation of the submission sequence

**V1 — initial recovery.** The goal was first to finish a complete, valid package without losing existing work. Twenty-three runs had S01–S07 verified with S08 videos deferred; six had all eight stage identities verified; Episode 16 used preserved legacy artifacts. Newly resumed episodes used a speed-oriented one-step refinement configuration in that recovery, not a faithful official-baseline reproduction. Thirty NPZ/GLB pairs were validated. Initial object initialization and tracking contained severe failures. Extremely poor CD-O (152.72099) was not the official CARI4D baseline.

**V2 — three pilots.** Episodes 6, 18, 28 received repaired masks/meshes/tracking and 100-step refinement. The other 27 reused v1. CD-O fell 50.57%, ACC-O fell 23.48%, human distance improved, but PEN worsened by 0.00020 cm. V2 still holds our best historical human-distance score. Evidence: [submission progress](reconstruction/incident_reports/track1_submission_progress.md), pilot scripts and local pilot receipts. That progress file contains older appended recovery states; read its dated sections as history.

**V3 — rollout.** Evaluated 27 reruns and promoted 23 newly reviewed episode predictions. Episodes 1, 2, 12, 14 retained v2, while 6, 18, 28 reused the pilots. CD-O fell another 54.75%, ACC-O 79.75%, ACC-H 25.35%. CD-H regressed 2.94% and PEN 36.48%. Image/unsigned-contact selection did not protect every hidden-reference metric. Evidence: [v3 results](reconstruction/incident_reports/track1_improvement/rollout/submission_results.md).

**V4 — object depth, geometry and temporal selection.** Promoted 2, 12, 14, 19, 20; Episode 1 failed the then-used penetration gate. A dynamic-programming orientation selector and hoop-specific plane transport addressed upstream rotation flips; limited depth corrections were reviewed separately. CD-O improved 16.68%, ACC-O 19.82%, but PEN regressed 4.09%. The local physical gate used the wrong containment direction and a different surface from official PEN. Episode 2 also retained an unsigned hand-proximity warning. This history must inform future interaction checks. Evidence: [v4 execution](reconstruction/incident_reports/track1_improvement/cdo_v4/execution_plan.md), [results](reconstruction/incident_reports/track1_improvement/cdo_v4/submission_results.md).

**V5 — fix PEN definition and revisit rotations.** Audited the 24 remaining episodes, tested the five largest rotation cases (23, 18, 21, 16, 13), and promoted 16, 21, 23. Rejected 13 and 18 on contact/penetration. Switched to predicted hands inside the actual submitted object, released winding convention and triangle distance. CD-O, ACC-O and PEN all improved. V5 changed isolated Episode 16 object predictions under that iteration's authorization; it did not overwrite the protected legacy reconstruction. Evidence: [v5 results](reconstruction/incident_reports/track1_improvement/cdo_v5/submission_results.md), [penetration audit](reconstruction/incident_reports/track1_improvement/cdo_v5/penetration_definition_audit.md).

**V6 — major Episode 1 hoop correction.** Changed only Episode 1 object geometry/trajectory, using an existing metric circular hoop with image-fitted orientation and a bounded smooth camera-ray depth correction. Three earlier candidates and a constant shift failed corrected hand checks and were rejected. Final actual-mesh full-hand/FPS512 checks passed all 174 scored frames with zero local penetration matching control. CD-O improved by 4.66134 cm (16.56%), far more than the later generic smoothing gain. Large rotation jumps decreased nine to zero. This was a strong, object-specific reconstruction correction. It did not establish grasp accuracy: the control had no contact-like frames. Evidence: [v6 results](reconstruction/incident_reports/track1_improvement/cdo_v6/submission_results.md), [final review](reconstruction/incident_reports/track1_improvement/cdo_v6/final_review.md).

**V7 — a controlled failure for spatial reconstruction.** All 30 episodes were swept through bounded scale/depth/temporal proposals, with strict filtering. Only Episode 7 (iron) changed in the submitted package: smoothing plus a net +0.07 m camera-ray depth shift. All 597 scored frames passed local image/depth/hand/proximity review. Common-pixel cached depth error improved 9.53994→5.68788 cm over 537 supported frames; separate held-out-frame depth improved 9.24343→4.23253 cm. Yet official CD-O worsened 23.48489→23.61430 (+0.55%), while ACC-O improved 4.68%. The local depth source is inferred, not hidden truth. Do not repeat this combined change as if its spatial benefit were established. Evidence: [v7 results](reconstruction/incident_reports/track1_improvement/cdo_v7/submission_results.md), [decisions](reconstruction/incident_reports/track1_improvement/cdo_v7/episode_decisions.md).

**V8 — contact-constrained smoothing of all 30, selective package.** The user explicitly chose applying the new smoother to all 30 episodes, rather than just packaging the Episode 7 pilot with unchanged v6. All 30 fits/checks completed; the package promotes 12 and retains 18 from v6. All human arrays, object rotations/scales/meshes are unchanged. Only 2,056 packed translation rows differ from v6. All five final scores are complete. ACC-O improves 11.24% against v6, and is better than v7 as well. CD-O and PEN improve only slightly. It is not a full geometry/rotation refinement, and it does not solve the remaining distance gap. Evidence: [v8 final results](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/submission_results.md), [selection](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/selection_review.json).

### 3.2 Exact upload IDs, completion and dates

IDs identify accepted uploads and allow read-only rechecks without creating duplicates. Dates below come from the authenticated submission history for the CD-O upload of each version (API timestamp shown as supplied). Each version has one package uploaded once to each competition.

| Version | CD-O upload date | CD-H ID | CD-O ID | ACC-H ID | ACC-O ID | PEN ID | State |
|---|---|---:|---:|---:|---:|---:|---|
| v1 | 2026-10-11 19:52:32.437000 | 57084318 | 57084320 | 57084322 | 57084324 | 57084325 | COMPLETE ×5 |
| v2 | 2026-10-12 20:22:24.240000 | 57118072 | 57118073 | 57118075 | 57118076 | 57118077 | COMPLETE ×5 |
| v3 | 2026-10-14 16:30:06.407000 | 57188874 | 57188875 | 57188879 | 57188881 | 57188885 | COMPLETE ×5 |
| v4 | 2026-10-14 21:56:30.793000 | 57196175 | 57196176 | 57196178 | 57196179 | 57196182 | COMPLETE ×5 |
| v5 | 2026-10-14 23:19:43.887000 | 57197586 | 57197588 | 57197591 | 57197592 | 57197593 | COMPLETE ×5 |
| v6 | 2026-10-15 01:38:20.597000 | 57201702 | 57201706 | 57201708 | 57201712 | 57201713 | COMPLETE ×5 |
| v7 | 2026-10-15 03:58:38.597000 | 57215485 | 57215489 | 57215493 | 57215494 | 57215499 | COMPLETE ×5 |
| v8 | 2026-10-15 16:21:27.253000 | 57237204 | 57237207 | 57237209 | 57237210 | 57237212 | COMPLETE ×5 |

### 3.3 Immutable package/provenance inventory

All archives are local under `reconstruction/data/v2d_challenge/track_1/submission_packages/`. Each has 740,780 rows and 30 episodes. A Parquet archive is the official payload; no ZIP of raw checkpoints is needed. The actual `code_commit_url` was read from each archive, and SHA256 was recomputed on this machine for this handoff. Full raw payloads are local, not embedded in this Markdown or normally committed to Git.

**v1** — `track1_first.parquet` (15,743,857 bytes).

- SHA256: `a290c7b14ea6bf24972187423e6967e12b56f5ee077c27f6f87f639208b144ef`.
- Producing commit: https://github.com/JavierMedel/video_to_data/commit/247c46a66cab1719311be186f652da859f57ecf8 .
- Local exact receipt: `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_first.kaggle-receipt.json`.

**v2** — `track1_improvement_pilot_v1.parquet` (15,781,211 bytes).

- SHA256: `c67d08c935bc839e1b21f339a79c0fc3ac749d1e8b51a006cac85bc600db6a33`.
- Producing commit: https://github.com/JavierMedel/video_to_data/commit/811b00735b793033d48047217ab1a41872d7637f .
- Local exact receipt: `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_improvement_pilot_v1.kaggle-receipt.json`.

**v3** — `track1_improvement_rollout_v1.parquet` (15,762,580 bytes).

- SHA256: `02ee693389aa310a74dbcd0cf2a968d67c277c39a12129319b757c90ee256ad8`.
- Producing commit: https://github.com/JavierMedel/video_to_data/commit/246ecc37927ec36278b7c855c071accdc635dba2 .
- Local exact receipt: `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_improvement_rollout_v1.kaggle-receipt.json`.

**v4** — `track1_cdo_v4.parquet` (15,757,529 bytes).

- SHA256: `29ee18c4ec3e9343d1869d3d41c8b46942e9738443303d6ad0fdda5987b9dedb`.
- Producing commit: https://github.com/JavierMedel/video_to_data/commit/6be2d01585ddd1f98fa52c0fc25534621afc1493 .
- Local exact receipt: `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v4.kaggle-receipt.json`.

**v5** — `track1_cdo_v5.parquet` (15,757,311 bytes).

- SHA256: `03cff99bcb9bbb8f0d869226580fd0c1740eb4b2c0c31856f062c06c823be68c`.
- Producing commit: https://github.com/JavierMedel/video_to_data/commit/1c3a1efca3c049c8cff229f25163254e518558d3 .
- Local exact receipt: `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v5.kaggle-receipt.json`.

**v6** — `track1_cdo_v6.parquet` (15,707,998 bytes).

- SHA256: `eb014ea2b52fb3faace7ae71209cec512bd528fbd8a984b2d09ee45c1cc76a14`.
- Producing commit: https://github.com/JavierMedel/video_to_data/commit/f53d46d3b4220a216c258b5306c438721a20653b .
- Local exact receipt: `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v6.kaggle-receipt.json`.

**v7** — `track1_cdo_v7.parquet` (15,707,940 bytes).

- SHA256: `121b97736636aa88ec9d6a0f0fc40e6919415ddd37377e1b2c0169937486cafe`.
- Producing commit: https://github.com/JavierMedel/video_to_data/commit/6facea9b1432998092dfba4408564338af2441db .
- Local exact receipt: `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_cdo_v7.kaggle-receipt.json`.

**v8** — `track1_contact_smoothing_v8.parquet` (15,708,476 bytes).

- SHA256: `bd73a931e9f9ec85b920884415c5dc2123d86e2d3d91395c7820ebf564fd3198`.
- Producing commit: https://github.com/JavierMedel/video_to_data/commit/b29d19010a14c39c5402ae7a8d0420bf6e0c925e .
- Local exact receipt: `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_contact_smoothing_v8.kaggle-receipt.json`.

## 4. Current live leaderboard comparison

The following is a time-stamped snapshot of the five separate Kaggle competitions. Each API response returned 26 team rows. Team names below are preserved as returned; names/IDs can differ across competitions (for example `RoboPI Lab` versus `RobotPI Lab`). Do not join numeric team IDs across separate competitions without checking identity.

For our best-score row, position is `1 + number of displayed teams with a strictly lower score`. This avoids inventing a unique order for tied scores. These are **per-metric positions**, not an official aggregate challenge rank. Historical package positions would depend on today's field and are not reconstructed as if they were historical ranks.

### 4.1 Our displayed best scores, latest v8, baseline and leader

| Metric | Latest v8 | Our displayed best | Version supplying our best | Position / returned rows | CARI4D baseline | Current leading score | Leading team |
|---|---:|---:|---|---|---:|---:|---|
| CD-H | 15.43955 | 14.99887 | v2 | 8/26 | 15.57527 | 13.43306 | EviisKitson |
| CD-O | 23.48324 | 23.48324 | v8 | 8/26 | 23.16246 | 17.91571 | EviisKitson |
| ACC-H | 1.15894 | 1.15894 | v3–v8 tied | 23/26 | 1.97387 | 0.13738 | EviisKitson |
| ACC-O | 0.71908 | 0.71908 | v8 | 21/26 | 0.72622 | 0.07111 | RobotPI Lab |
| PEN | 0.00322 | 0.00213 | v1 | 14/26 | 0.00423 | 0.00000 | QY Lin |

The latest v8 package beats CARI4D on CD-H, ACC-H, ACC-O and PEN, but still loses on CD-O. CARI4D is not the leading team on most metrics. Our acceleration scores remain far behind the current leaders despite beating CARI4D: ACC-H 0.13738 and ACC-O 0.07111 are the leading scores. This exposes a larger temporal opportunity than simply matching the released baseline.

The team's leaderboard CD-H 14.99887 and PEN 0.00213 **must not be reported as v8 scores**. Those came from different artifacts. Human recombination can change first-frame alignment and therefore object metrics/PEN; any hybrid must be packaged and evaluated as a new candidate. PEN near zero can indicate separation, not correct grasp/contact.

### 4.2 Full returned standings for reproducible context

These tables preserve all 26 returned teams per metric, not just the leader and our row. `Position` uses strict-lower-score ranking; equal displayed scores share the same position. The direct Kaggle links are the current primary source; this copy is a snapshot, not an auto-refreshing dashboard.

#### CD-H — https://www.kaggle.com/competitions/v2d-challenge-track1-cd-h/leaderboard

| Position | Team | Public score (cm) |
|---:|---|---:|
| 1 | EviisKitson | 13.43306 |
| 2 | RobotPI Lab | 14.01671 |
| 3 | ByeByeHelloHaHaHa | 14.20399 |
| 4 | Lodestar | 14.28963 |
| 5 | um | 14.85738 |
| 6 | sv1105 | 14.94909 |
| 7 | Lattice Studios | 14.98258 |
| 8 | Javier Medel, MSc. | 14.99887 |
| 9 | Chivatam | 15.11194 |
| 10 | QY Lin | 15.12774 |
| 11 | Xueyi Wang (FF) | 15.19253 |
| 12 | Mochii | 15.20277 |
| 13 | Open4D | 15.41265 |
| 14 | TheTensorGeek | 15.44549 |
| 15 | SNU CVLAB | 15.48515 |
| 16 | xpz666 | 15.51196 |
| 17 | CARI4D Basline | 15.57527 |
| 18 | AI_CENTER | 15.71611 |
| 19 | xiaopai7788 | 16.17115 |
| 20 | Marcell | 16.81291 |
| 21 | Swithinraj | 18.99071 |
| 22 | Null Walker | 35.52454 |
| 23 | Punya Malaiya | 43.73273 |
| 24 | Kelly To the H | 67.09885 |
| 25 | Tony | 70.64959 |
| 26 | Furqan Zaheer | 73.64771 |

#### CD-O — https://www.kaggle.com/competitions/v2d-challenge-track1-cd-o/leaderboard

| Position | Team | Public score (cm) |
|---:|---|---:|
| 1 | EviisKitson | 17.91571 |
| 2 | xpz666 | 18.54107 |
| 3 | RoboPI Lab | 19.10040 |
| 4 | Lodestar | 19.52520 |
| 5 | sv1105 | 23.05002 |
| 6 | SNU CVLAB | 23.13746 |
| 7 | CARI4D Basline | 23.16246 |
| 8 | Javier Medel, MSc. | 23.48324 |
| 9 | Open4D | 24.56680 |
| 10 | Lattice Studios | 25.03212 |
| 11 | Swithinraj | 25.22105 |
| 12 | Xueyi Wang (FF) | 26.64580 |
| 13 | QY Lin | 29.77010 |
| 14 | ByeByeHelloHaHaHa | 31.40578 |
| 15 | Mochii | 32.18318 |
| 16 | AI_CENTER | 37.42007 |
| 17 | Chivatam | 38.43505 |
| 18 | xiaopai7788 | 49.19655 |
| 19 | Punya Malaiya | 78.69475 |
| 20 | TheTensorGeek | 84.22687 |
| 21 | um | 105.26281 |
| 22 | Marcell | 105.87997 |
| 23 | Tony | 145.40682 |
| 24 | Furqan Zaheer | 169.91766 |
| 25 | Kelly To the H | 422.22380 |
| 26 | Null Walker | 525.36396 |

#### ACC-H — https://www.kaggle.com/competitions/v2d-challenge-track1-acc-h/leaderboard

| Position | Team | Public score (cm) |
|---:|---|---:|
| 1 | EviisKitson | 0.13738 |
| 2 | RobotPI Lab | 0.13809 |
| 3 | SNU CVLAB | 0.13826 |
| 4 | Swithinraj | 0.14171 |
| 5 | Lattice Studios | 0.14174 |
| 6 | Lodestar | 0.14381 |
| 7 | Tony | 0.14528 |
| 7 | Furqan Zaheer | 0.14528 |
| 9 | Xueyi Wang (FF) | 0.14854 |
| 10 | QY Lin | 0.15089 |
| 11 | Marcell | 0.15586 |
| 12 | ByeByeHelloHaHaHa | 0.16955 |
| 13 | AI_CENTER | 0.17446 |
| 14 | Open4D | 0.17835 |
| 15 | Kelly To the H | 0.17963 |
| 16 | Punya Malaiya | 0.21159 |
| 17 | Mochii | 0.22359 |
| 18 | Chivatam | 0.25584 |
| 19 | sv1105 | 0.89057 |
| 20 | um | 0.89932 |
| 21 | xpz666 | 0.94279 |
| 22 | TheTensorGeek | 0.97204 |
| 23 | Javier Medel, MSc. | 1.15894 |
| 24 | CARI4D Basline | 1.97387 |
| 25 | xiaopai7788 | 2.15369 |
| 26 | Null Walker | 2.49281 |

#### ACC-O — https://www.kaggle.com/competitions/v2d-challenge-track1-acc-o/leaderboard

| Position | Team | Public score (cm) |
|---:|---|---:|
| 1 | RobotPI Lab | 0.07111 |
| 2 | Lattice Studios | 0.07136 |
| 3 | EviisKitson | 0.07288 |
| 4 | Marcell | 0.07429 |
| 4 | Tony | 0.07429 |
| 4 | Furqan Zaheer | 0.07429 |
| 7 | Swithinraj | 0.07693 |
| 8 | Xueyi Wang (FF) | 0.07760 |
| 9 | Kelly To the H | 0.08356 |
| 10 | Lodestar | 0.09232 |
| 11 | ByeByeHelloHaHaHa | 0.12236 |
| 12 | AI_CENTER | 0.12737 |
| 13 | QY Lin | 0.14036 |
| 14 | Punya Malaiya | 0.18698 |
| 15 | Open4D | 0.19278 |
| 16 | Chivatam | 0.22094 |
| 17 | SNU CVLAB | 0.25458 |
| 18 | Mochii | 0.30525 |
| 19 | xiaopai7788 | 0.54868 |
| 20 | sv1105 | 0.63201 |
| 21 | Javier Medel, MSc. | 0.71908 |
| 22 | CARI4D Basline | 0.72622 |
| 23 | xpz666 | 0.74298 |
| 24 | TheTensorGeek | 0.74841 |
| 25 | um | 0.99505 |
| 26 | Null Walker | 12.96689 |

#### PEN — https://www.kaggle.com/competitions/v2d-challenge-track1-pen/leaderboard

| Position | Team | Public score (cm) |
|---:|---|---:|
| 1 | QY Lin | 0.00000 |
| 1 | Tony | 0.00000 |
| 1 | Furqan Zaheer | 0.00000 |
| 1 | Kelly To the H | 0.00000 |
| 5 | Lodestar | 0.00005 |
| 6 | RobotPI Lab | 0.00025 |
| 7 | EviisKitson | 0.00029 |
| 8 | TheTensorGeek | 0.00030 |
| 9 | Marcell | 0.00053 |
| 10 | ByeByeHelloHaHaHa | 0.00058 |
| 11 | Punya Malaiya | 0.00149 |
| 12 | Lattice Studios | 0.00151 |
| 13 | Xueyi Wang (FF) | 0.00204 |
| 14 | Javier Medel, MSc. | 0.00213 |
| 15 | Chivatam | 0.00239 |
| 16 | Open4D | 0.00286 |
| 17 | CARI4D Basline | 0.00423 |
| 18 | Mochii | 0.00452 |
| 19 | xiaopai7788 | 0.00487 |
| 20 | Swithinraj | 0.00575 |
| 21 | SNU CVLAB | 0.01011 |
| 22 | sv1105 | 0.01867 |
| 23 | Null Walker | 0.03795 |
| 24 | xpz666 | 0.06833 |
| 25 | um | 0.07020 |
| 26 | AI_CENTER | 0.10070 |

## 5. What the metrics actually measure

The local primary implementation is `reconstruction/v2d_submission_kit/v2d_submission_kit/v2dlb/`, especially `mhr_submission.py`, `mhr_metrics.py`, `mhr_body.py`, and the released `metric_code/track_1/` scripts. Read the implementation when designing an objective; names alone are insufficient. The docstring in `mhr_metrics.py` also mentions combined Chamfer `cd_c_cm`; that is **not** one of the current five Track 1 competition keys.

### 5.1 Human representation, object representation and alignment

- Human predictions are **MHR parameters**, not arbitrary SMPL joint arrays. Local NPZ contains `pose[T,136]`, fixed per-episode `scales[68]`, and fixed per-episode `shape[45]`. The host decodes its MHR model to joints, surfaces, alignment vertices and hands.
- Object predictions are a triangle mesh, per-episode positive scalar scale, per-frame rotation matrices `[T,3,3]` and translation `[T,3]`. Scene points follow `s * R @ mesh_point + t`. Local object geometry/translations use metres; reported distance/proxy values may be converted to centimetres. Check units at model boundaries.
- One global similarity transform is fit using the first scored frame's human alignment vertices. It is applied to the predicted sequence. There is no independent per-frame ICP that removes trajectory errors. Changing human root, shape or identity can affect alignment and object scores even if object parameters are copied.
- CD-H and CD-O compare sampled reconstructed surfaces with the hidden multiview reference after the configured alignment. A silhouette fit is not a 3D surface-distance certificate.
- ACC-H compares selected body-joint second differences against reference motion. ACC-O compares object-translation second differences against reference. Official acceleration **error** is not the same thing as the magnitude of predicted acceleration. Constant or frozen predictions are not guaranteed to score well.
- Scored frames come from the official public sample and can have gaps. Acceleration is computed within contiguous stretches. **Never form differences across missing frame indices.** The current 30 episodes contain 9,877 scored frames and 9,747 valid consecutive scored triples.
- Public scored-frame metadata is available locally; hidden per-frame/episode reference geometry and official per-episode scores were not obtained in this work. Do not invent which episode contributes most to aggregate official CD-O from a local proxy alone.

### 5.2 Official PEN and what remains approximate locally

Official PEN transforms **predicted hand vertices into the submitted object frame**, classifies containment by absolute generalized winding number greater than 0.5, measures distance to actual triangles, averages over the host-selected hand points/frames, and applies object scale plus the positive human alignment scale. The old v4 local gate instead measured object points inside a body proxy: opposite containment direction and a different sample set. That error was diagnosed and corrected in v5.

The exact host `role_left_hand`/`role_right_hand` indices and hidden alignment scale are not available in the participant metadata inspected here. Local corrected review uses all 4,636 anatomically selected hand vertices and a local FPS512 subset. Those are explicit approximations to the host's selection, not official labels. Using winding on the actual mesh matters for open/concave/thin surfaces; do not silently substitute ray parity, a convex hull or a watertight proxy. Winding uses absolute orientation so reversed faces alone are not the entire issue.

With fixed human arrays, the positive alignment scale is fixed per episode. Exact-host-point non-increase would support a monotonic PEN claim, but our hand-subset mismatch prevents a universal guarantee from local means. Low penetration alone does not establish physical contact or a valid grasp: moving the object away can reduce PEN while corrupting the reconstruction. Preserve proximity and observed interaction checks.

### 5.3 Local evidence and its limitations

| Local diagnostic | What it supports | What it does not establish |
|---|---|---|
| Visible silhouette IoU | Agreement between submitted projected mesh and observed object mask, accounting for human occlusion | Hidden 3D position/shape accuracy; masks may be imperfect |
| Common-pixel rendered-depth residual | Agreement to cached human-aligned monocular inferred depth on shared supported pixels | A measured depth sensor or hidden-reference depth; v7 improved this while CD-O worsened |
| Translation/surface acceleration magnitude | Amount of predicted jitter; only consecutive scored triples match evaluator inclusion | Official acceleration error to hidden motion; true acceleration can be removed |
| Full-hand/FPS512 exact-mesh penetration | Physical consistency of predicted human/object under the tested point sets | Exact host PEN or grasp correctness |
| Unsigned nearest-vertex gap | A coarse separation/contact-loss proxy | A semantic contact label or exact surface distance |
| Converter vertex error | Numerical fidelity of exported MHR fit to the source predicted body | Accuracy to hidden multiview human reconstruction |
| Completely decoded MP4 | Video exists with expected frame count and is decodable | Every frame visually assessed or every contact verified |
| Six inspected frames/crops | Sampled qualitative check at recorded frames | Exhaustive full-clip visual validation |

Any new plan must keep these evidence levels separate. A disjoint-frame local depth split is not hidden-ground-truth validation, and a reused fit cache is not an independent evaluation.

## 6. V8 implementation, selection and exact scope

### 6.1 Algorithm actually run

The proposal is a **seven-frame quadratic Savitzky–Golay translation smoother**, with no constant ray-depth shift. Each video frame offers two finite choices: unchanged submitted v6 translation or the smoothed proposal. Rotations, human parameters, scale and meshes are fixed.

Every frame/option is evaluated using the actual object mesh and predicted anatomical hands. A smoothed option is feasible only when:

1. Both full-hand and FPS512 **per-frame mean** penetration do not increase (tiny numerical tolerance only).
2. On control-contact-like frames (nearest-vertex proxy at most 1.5 cm), the gap increases by no more than 0.5 cm.
3. Translation displacement is at most 0.03 m, and the candidate's camera depth exceeds 0.1 m.
4. The first and last two frames remain unchanged. The original option is always available as the fallback.

An exact second-order dynamic program selects a feasible sequence minimizing squared acceleration of the transformed mesh centroid plus a fidelity penalty of 0.001 times squared displacement from control. With rotations fixed, the option-dependent part of average squared sampled-surface acceleration reduces to the transformed centroid objective; rotational variance is path-independent. This is why the DP can be exact for this limited finite-choice problem. It is not a global optimum over all physically possible SE(3) trajectories or over the hidden metric.

The fit covers **16,563 video frames and 33,126 frame-option evaluations**. Episode 7 reuses its completed fit. Other 29 episodes receive isolated fits. Contacts are checkpointed every eight frames. Resume verifies complete artifacts, inputs and evaluator-source fingerprints. Completed state and all original v6 source hashes were audited.

### 6.2 Promotion gates

The original full-video numerical screen must pass. A second conservative screen requires a meaningful gain on the **consecutive scored triples**, or in image/depth evidence: at least 5% sampled-surface acceleration-magnitude reduction, or at least 0.005 mean IoU increase, or at least 1 cm common cached-depth reduction with sufficient supported frames. Translation-magnitude reduction alone is not enough. Motion guard thresholds are `max(0.1 cm, 1.1 * control)` for translation and sampled-surface acceleration proxies on both full-video and scored scopes. Unchanged translations do not count as an improvement.

Sixteen candidates passed the original numerical screen. Episodes 10, 12, 17, 27 then failed the additional meaningful-gain screen on scored triples. The final 12 received sampled visual review. The additional screen never rescued an original failure.

**Promoted:** 1, 2, 6, 7, 8, 9, 11, 14, 22, 24, 26, 29. **Retained v6:** 0, 3, 4, 5, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 25, 27, 28.

For the selected complete package, frame-weighted local translation acceleration magnitude on 9,747 scored triples fell 0.827185→0.704264 cm (14.86%), while sampled-surface acceleration magnitude fell 2.285616→2.203472 cm (3.59%). These are local magnitudes, not the official 0.81013→0.71908 acceleration-error comparison.

### 6.3 Complete episode ledger

Indices below are original video frame indices in arrays; the listed counts distinguish full video from the scored subset. Gain columns are the isolated candidate's scored-triple **local magnitude reductions against v6**, including rejected candidates. Rejected gains are not included as if they were submitted. `Retain` means the final v8 package reuses v6 for that episode.

| Episode | Object | Video frames | Scored frames | Consecutive scored triples | Local translation gain | Local surface gain | V8 selection |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | hula_hoop | 790 | 169 | 157 | +50.48% | +3.53% | Retain v6 |
| 1 | hula_hoop | 668 | 174 | 162 | +13.53% | +12.84% | Promote |
| 2 | hula_hoop | 866 | 84 | 78 | +23.56% | +21.71% | Promote |
| 3 | big_red_bowl | 592 | 340 | 326 | +24.40% | +0.83% | Retain v6 |
| 4 | big_red_bowl | 747 | 519 | 515 | +20.23% | +3.17% | Retain v6 |
| 5 | big_red_bowl | 668 | 412 | 400 | +26.17% | +1.16% | Retain v6 |
| 6 | iron | 816 | 636 | 634 | +31.89% | +10.00% | Promote |
| 7 | iron | 777 | 597 | 595 | +20.61% | +6.55% | Promote |
| 8 | iron | 634 | 454 | 452 | +16.73% | +7.00% | Promote |
| 9 | white_desk | 415 | 325 | 323 | +23.31% | +6.91% | Promote |
| 10 | white_desk | 577 | 418 | 416 | +21.89% | +4.53% | Retain v6 |
| 11 | white_desk | 877 | 803 | 801 | +37.64% | +13.12% | Promote |
| 12 | black_pan | 405 | 209 | 199 | +11.54% | +3.93% | Retain v6 |
| 13 | black_pan | 425 | 106 | 98 | +2.09% | +0.48% | Retain v6 |
| 14 | black_pan | 442 | 83 | 75 | +32.52% | +9.90% | Promote |
| 15 | foam_grass_block | 501 | 356 | 354 | +16.38% | +1.47% | Retain v6 |
| 16 | foam_grass_block | 360 | 240 | 238 | +7.29% | +0.85% | Retain v6 |
| 17 | foam_grass_block | 419 | 277 | 275 | +21.78% | +2.52% | Retain v6 |
| 18 | paint_roller | 535 | 420 | 418 | +26.26% | +1.59% | Retain v6 |
| 19 | paint_roller | 443 | 283 | 281 | +18.94% | +1.34% | Retain v6 |
| 20 | paint_roller | 549 | 375 | 371 | +8.68% | +1.44% | Retain v6 |
| 21 | pink_foam_roll | 563 | 347 | 345 | +7.97% | +0.60% | Retain v6 |
| 22 | pink_foam_roll | 533 | 452 | 450 | +23.52% | +7.92% | Promote |
| 23 | pink_foam_roll | 552 | 399 | 397 | +11.27% | +1.05% | Retain v6 |
| 24 | short_wood_stool | 420 | 242 | 240 | +37.48% | +21.06% | Promote |
| 25 | short_wood_stool | 365 | 219 | 217 | +13.06% | +0.88% | Retain v6 |
| 26 | short_wood_stool | 399 | 281 | 279 | +44.22% | +14.48% | Promote |
| 27 | white_laptop_cart | 440 | 243 | 241 | +21.68% | +4.32% | Retain v6 |
| 28 | white_laptop_cart | 366 | 147 | 145 | +5.20% | +1.14% | Retain v6 |
| 29 | white_laptop_cart | 419 | 267 | 265 | +23.66% | +5.73% | Promote |

The names map to ten object groups with three episodes each. The circular hoop is a special geometry case; pans/handled objects must not inherit the hoop's symmetry assumptions. A candidate may reduce translation jitter but leave rotational flips unchanged; rejecting it for insufficient surface gain is intentional.

### 6.4 Review coverage and evaluator acceleration

Episode 7 has an independent final all-scored-frame hand recomputation and full comparison videos (777 frames each at 30 fps), fully decoded and visually sampled. Its unconstrained smoother would violate the hand gate; constrained selection smooths 377 of 777 frames and leaves the original 48 rotation jumps above 45° untouched. Local all-video surface-magnitude reduction 11.92% differs from its scored-triple 6.55% reduction; never mix temporal scopes.

For the other campaign outputs, final hand summaries explicitly **reuse exact precomputed option costs** after verifying the chosen translations and input hashes. The DP introduces no new pose outside those evaluated options. This is valid finite-option reuse but not an independent rerender or held-out test. Per-vertex maximum depth and vertexwise deltas unavailable from the compact cache are recorded as `null`, not invented zeroes. All-scored-frame image/common-support depth checks run separately.

The 12 promoted episodes have six sampled source RGB/mask/projected-mesh comparisons and matching object crops. This is **not 30 fully inspected reconstruction videos**. Existing mesh/outline mismatch remains; no gross new drift was found in the inspected samples. Visual-review hashes and reasons are in `visual_reviews.json`.

A Torch-compiled fused generalized-winding checker accelerated fresh option caches. It preserves the eager checker geometry, threshold, float64 accumulation, near-boundary float64 fallback, AABB filtering and triangle distances. Six comparisons covered a box, open box, reversed box, and real dense meshes for episodes 1, 16, 25 (103,772; 352,796; 757,976 faces). Maximum observed depth difference was zero at the tested 2e-7 m absolute tolerance. One benchmark under concurrent load measured 0.597705 s eager versus 0.342167 s fused,1.75×. This is not a guaranteed whole-campaign speedup or a performance promise for new geometry.

Source fingerprints: fused `114c32d03dbe2e2ddecf8cea96081a0e8b484f28392ba6f70e9339ea0b103692`; eager `ba0d6f56b4beb31b3dce3db87d462d2fa48655a576a2b162393a7a981eb24cb3`. Cached evaluations bind the chosen implementation and source. Changing these files invalidates the assumption that old caches match current arithmetic. Do not silently relabel old caches after source edits.

## 7. Experiments already tried, including rejections

The next planner must account for negative evidence rather than propose a clean-slate list of already-tested tricks.

| Experiment | Observed result | Planning consequence |
|---|---|---|
| Simply increasing refinement to 300 steps on Episode 12 | Did not eliminate upstream flips; not promoted | Inspect tracking/ambiguity before spending a full 30 rerun budget |
| Enabling refinement rotation on Episode 12 | Flip failure persisted; not promoted | Rotation optimization alone is not a demonstrated fix |
| Full-clip 100-step refinement vs seeded 64-frame batches | IoU 0.71401→0.71302; translation proxy 3.70186→3.47259 cm; 72 large jumps remain | Full-clip batching helps a proxy slightly, but did not resolve ambiguity |
| Blind constant ray offsets | Can reduce inferred-depth residual and harm silhouette/contact | Require common-support evidence, interaction review, single-factor ablations |
| V4 body-proxy intersection gate | Official PEN worsened despite local acceptance | Use corrected hand-inside-object definition and disclose subset mismatch |
| V6 hoop-specific shape/plane/depth repair | Major official CD-O improvement | Object-specific geometry can be higher value than generic denoising |
| V7 Episode 7 smoothing plus+0.07 m ray depth | ACC-O improved but CD-O worsened | Separate smoothing from depth; inferred depth may be biased |
| Episode 7 unconstrained SG7 smoothing | Lower jitter, increased hand penetration | Do not apply an unconstrained global filter blindly |
| V8 contact-constrained SG7 selection | ACC-O 11.24% better, CD-O almost unchanged | Strong temporal result, not a solved spatial/rotation problem |

### 7.1 Episode 7 four-way ablation (local only)

All branches use submitted v6, differ only in translation, and evaluate 597 scored frames. The combined branch reproduces submitted v7 within 1 micrometre. Only the combined historical v7 branch has its known official package score; the other isolated branches were not separately submitted.

| Branch | Constant ray shift | Translation magnitude proxy (cm) | Surface magnitude proxy (cm) | IoU | Cached depth residual (cm) | Hand gate | Proximity gate |
|---|---:|---:|---:|---:|---:|---|---|
| Control | 0.00 m | 1.765503 | 3.466227 | 0.705783 | 9.549499 | Pass | Pass |
| Smoothing only | 0.00 m | 0.611534 | 2.665002 | 0.705158 | 9.552169 | **Fail** | Pass |
| Depth only | +0.07 m | 1.773966 | 3.470335 | 0.707713 | 5.823755 | Pass | Pass |
| Combined | +0.07 m | 0.615009 | 2.666447 | 0.706794 | 5.687881 | Pass | Pass |

These depth comparisons use branch-specific common support, so even their support sets can differ. The unchanged control intentionally does not pass a meaningful-*improvement* gate; that is not evidence its arrays are invalid. See [ablation results](reconstruction/incident_reports/track1_improvement/episode7_translation_ablation/results.md) and [findings](reconstruction/incident_reports/track1_improvement/episode7_translation_ablation/findings.md).

### 7.2 Official baseline reproduction remains unverified

V4 audited the local CARI4D checkpoint and resolved configuration against the released manifest and found exact hash matches. Repository default refinement uses 300 steps and full-clip batches; v3 used 100 steps and 64-frame batches. Geometry inputs, custom gating/tracking and export details differ. Exact organizer baseline input artifacts and complete production recipe were not obtained. Therefore describe our work as a **CARI4D-based local pipeline**, not a verified reproduction of the official 23.16246 score. The scorer implementation is visible, but hidden reference geometry is not.

## 8. Execution machine specifications and current capacity

Everything in this section describes the machine on which the next plan will run. It does not describe the separate planning laptop. Measurements were taken on 2026-10-16 and must be rechecked when executing a future plan.

| Component | Verified local specification |
|---|---|
| Workspace root | `/home/alien51/video_to_data` |
| Reconstruction root | `/home/alien51/video_to_data/reconstruction` |
| Git branch at handoff | `v2d-track1-blackwell` |
| Base source/report commit before this handoff | `0b966629` (the handoff itself is a subsequent documentation commit) |
| Shell | Bash on Linux/WSL2 |
| Linux distribution | Ubuntu 26.04 LTS, Resolute Raccoon |
| Kernel | `6.6.87.2-microsoft-standard-WSL2`, x86_64 |
| CPU | Intel Core Ultra 9 275HX; 24 visible CPUs; one visible thread per core |
| Memory visible to WSL | Approximately 31 GiB RAM; 29 GiB available at measurement |
| Swap | 8 GiB; unused at measurement |
| GPU | **NVIDIA GeForce RTX 5090 Laptop GPU**, one device |
| GPU memory | **24,463 MiB** as reported by `nvidia-smi` (do not assume desktop 5090 32 GB) |
| GPU compute capability | **12.0 / sm_120 (Blackwell)** |
| NVIDIA driver | 616.92 |
| Docker server | 29.8.2; `docker ps` returned no active containers at inspection |
| Main filesystem | Approximately 1007 GiB total,326 GiB used,631 GiB available |
| Track 1 local data tree | Approximately 273 GiB including prior runs, candidates and caches |
| Shared local weights | Approximately 32 GiB |
| Host challenge Python | `reconstruction/.venv/bin/python` (Python 3.11) |
| Host packaging libraries | NumPy 2.4.6, pandas 3.0.6, PyArrow 25.0.1, trimesh 5.1.1, Kaggle 2.2.4; fast_simplification installed |
| Host missing heavy libraries | `torch`, `scipy`, `cv2`, `h5py` not importable in the host challenge venv at inspection |
| Main GPU runtime image | `v2d_cari4d:blackwell` |
| Inside that image | Python 3.11.10; Torch 2.8.0+cu128; CUDA 12.8; NumPy 1.26.3; SciPy 1.16.3; OpenCV 4.11.0 |
| Compiled CUDA architectures | Main image reports sm_120 support (also 70/75/80/86/90/100) |

The GPU runtime was actually tested with a short read-only container import/device probe. Hardware was visible and working. Old recovery reports describing broken Docker/WSL integration are historical, not the current blocker. Do not rebuild or migrate the environment merely because an old incident file says Docker was absent.

Installed image inventory at inspection (reported sizes are not additive disk use because layers are shared):

| Image | Docker reported size | Role |
|---|---:|---|
| `v2d_cari4d:blackwell` | 59.9 GB | Main tested Track 1 runtime, refinement, exact rendering/contact checks |
| `v2d_sam3d:blackwell` | 34 GB | Object reconstruction support |
| `v2d_detectron2:blackwell` | 28.3 GB | Segmentation/detection support |
| `v2d_sam2:blackwell` | 34 GB | Mask tracking support |
| `v2d_grounding_dino:blackwell` | 29.5 GB | Detection support |
| `v2d_cari4d:latest` | 41.2 GB | Older runtime; not the tested Blackwell execution default |
| `v2d_moge:latest` | 28.1 GB | Monocular depth support |
| `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` | 26.5 GB | Runtime/build base |
| `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` | 20.9 GB | Older base; do not assume sm_120 support |
| `nvidia/cuda:12.8.1-base-ubuntu24.04` | 411 MB | CUDA base/probe |
| `python:3.11-slim` | 189 MB | Lightweight utility runtime |

### 8.1 Practical compute limits

Treat this as **one laptop GPU with about 24 GiB VRAM**, not a datacenter cluster. V8 successfully used at most two independent GPU subprocesses; that does not prove two arbitrary full refinements or two training jobs fit. Use sequential heavy jobs or measure peak memory before increasing concurrency. Account for WSL RAM, thermal limits, disk growth, container caches and dense-mesh winding cost. Plan a small pilot with measured runtime/VRAM before committing to all 30. No new hardware, cloud account, remote training host or planning-laptop GPU is assumed.

The main Docker pattern mounts the reconstruction root to `/repo`, preserving container-local dependencies under `/workspace/...`. Scripts generally run as UID/GID 1000:1000 with explicit cache/temp directories. Some libraries call `getpass.getuser()` and require `LOGNAME` when that UID lacks a passwd entry. The completed contact runner already handles the fused path. Keep writable caches separate and do not expose credentials or personal filesystem contents in reports.

## 9. Filesystem, source map and artifact ownership

Paths here are relative to the workspace root unless stated otherwise. Large local data directories are not guaranteed to exist on another clone.

| Path | Meaning / how to use it |
|---|---|
| `reconstruction/data/v2d_challenge/track_1/episode_000000` … `episode_000029` | Input videos and per-episode prepared artifacts |
| `reconstruction/data/v2d_challenge/track_1/meta/episodes_metadata.jsonl` | Episode/object metadata |
| `reconstruction/incident_reports/track1_scored_frames.json` | Public-sample-derived scored original video indices; not hidden labels |
| `reconstruction/data/v2d_challenge/track_1/submission_episodes/` | First-package local source pairs; preserve |
| `reconstruction/data/v2d_challenge/track_1/improvement_pilot_v1/candidate_episodes/` | Complete v2 candidate pairs |
| `reconstruction/data/v2d_challenge/track_1/improvement_rollout_v1/candidate_episodes/` | Complete v3 candidate pairs |
| `reconstruction/data/v2d_challenge/track_1/cdo_v4_experiments/candidate_episodes/` … `cdo_v7_experiments/candidate_episodes/` | Per-version object candidates/packaging sources; verify exact existence in that version's validation manifest |
| `reconstruction/data/v2d_challenge/track_1/cdo_v6_experiments/candidate_episodes/` | Immutable v6 control used by ablation/v8 |
| `reconstruction/data/v2d_challenge/track_1/episode7_translation_ablation/` | Four translation-only local branches |
| `reconstruction/data/v2d_challenge/track_1/episode7_contact_smoothing/episode_000007/` | Original independently checked constrained Episode 7 result |
| `reconstruction/data/v2d_challenge/track_1/contact_smoothing_v8/episode_000NNN/` |29 other isolated fits, option caches, selections, mask sheets/crops |
| `reconstruction/data/v2d_challenge/track_1/contact_smoothing_v8/submission_episodes/` | **Final complete v8 NPZ/GLB pairs; recommended immutable latest-package starting point** |
| `reconstruction/data/v2d_challenge/track_1/submission_packages/` | Eight immutable scored Parquets and exact upload receipts |
| `reconstruction/incident_reports/track1_improvement/` | Committed evidence by iteration; source-preservation hashes, reviews, decisions |
| `reconstruction/data/weights/cari4d/2026-08-25-09-35-57/` | Released `manifest.json`, `resolved_config.yaml`, `step200000.pth` |
| `reconstruction/data/weights/track1_submission_mhr/mhr_model.pt` | Decoder used by current local hand reviews |
| `reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz` | Anatomical local hand specification |
| `reconstruction/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000NNN/run/export/episode_000NNN/edex` | Current calibration/export source for episodes 6, 18, 28 |
| `reconstruction/data/v2d_challenge/track_1/improvement_rollout_v1/episode_000NNN/run/export/episode_000NNN/edex` | Calibration/export source for other episodes in the contact pipeline |

Per-episode export contains masks, camera calibration and mesh information. The contact pipeline explicitly checks the used camera transform against identity. Do not assume that identity convention generalizes to another export, new world frame, or egocentric track. Use the actual intrinsics, transforms, mesh origin and object scale.

A standard local pair is `episode_000007.npz` and `episode_000007_object.glb`. Besides the required prediction arrays, NPZ can contain `per_frame_vertex_error_mm[T]`, `valid_input[T]` and a Unicode JSON `report`. They are not hidden ground truth. Existing verification uses `allow_pickle=False`; preserve metadata fields unless the change explicitly concerns them.

### 9.1 Script responsibilities and limitations

| Script/module | Responsibility / caution |
|---|---|
| `reconstruction/scripts/track1_contact_smoothing.py` | Pure NumPy feasibility, exact finite-choice DP, selected contact summaries, scored-gap-safe motion diagnostics |
| `reconstruction/scripts/track1_episode7_ablation.py` | Four-way translation experiment, SG proposal generation and bundle identity checks; SciPy imported lazily for smoothing |
| `reconstruction/scripts/track1_episode7_contact_smoothing.py` | GPU option checks, isolated per-episode fit and review; supports `--episode`, `--campaign`; hardcodes submitted v6 control |
| `reconstruction/scripts/track1_contact_campaign.py` | All 30 coordinator, max 2 subprocesses, exclusive lock, resume/input checks; `--status` is read-only; no packing/upload |
| `reconstruction/scripts/track1_contact_campaign_review.py` | Full-video plus scored-frame promotion screens and visual-evidence checks; writes review/decisions, no refit/upload |
| `reconstruction/scripts/track1_contact_campaign_package.py` | Stage only final promoted translations into complete package; optional actual producing commit; official pack and baseline numeric-row comparison; never uploads |
| `reconstruction/scripts/track1_cdo_full_image_review.py` | All-scored-frame projected mesh/mask and common-support depth review; supports explicit control/candidate/report directories |
| `reconstruction/scripts/track1_cdo_hand_penetration_review.py` | Correct containment direction, exact-mesh hand checks; `--sample-count 0` evaluates all scored frames; explicit directories supported |
| `reconstruction/scripts/track1_cdo_render_candidate.py` | Actual-source comparison mask sheets and full reconstruction videos; explicit directories and labels supported |
| `reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/mesh_penetration_review.py` | Eager exact-mesh winding/distance checker; frozen-source cache identity matters |
| `reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/fused_mesh_penetration_review.py` | Compiled equivalent checked on six cases; frozen-source cache identity matters |
| `reconstruction/scripts/track1_convert_refined.py` | Convert source refined body outputs to official MHR-compatible parameters and report fitting fidelity |
| `reconstruction/scripts/track1_cdo_temporal_candidates.py` and hoop/depth/geometry scripts | Previous object-specific and rotation hypotheses; read version report before reuse |
| `reconstruction/scripts/track1_rollout_common.py` and rollout/pilot scripts | Input/export path mapping, rerun orchestration and older selection |
| `reconstruction/scripts/track1_cdo_package.py` | Older iteration-specific packer; only 4–7 supported; not a generic future-version packer |
| `reconstruction/scripts/track1_finalize_submission.py` | First-submission-specific workflow; **do not rerun for a new version** |
| `reconstruction/v2d_submission_kit/v2d_submission_kit/eval_reconstruction.py` | Official Track 1 Parquet packing entry point |
| `reconstruction/v2d_submission_kit/v2d_submission_kit/submit/v2d_submit.py` | Official validation, authenticated quota/entry check, upload, exact-ID receipt polling |

A new campaign name currently isolates output folders but **does not** make the smoother or package checker generic for v8 or future baselines. Explicitly parameterize controls and baseline Parquet, bind their hashes, update tests, and report the chosen parent if proposing a v9 experiment. Do not accidentally replace the useful 12 v8 trajectories with v6 through a hardcoded fallback.

### 9.2 Preservation and historical protected work

All submitted archives, source NPZ/GLB pairs, original checkpoints and accepted receipts must remain immutable. Use a new experiment directory and record parent hashes. Never overwrite the only copy of an accepted candidate. Keep rejected trials for attribution and resume.

Episode 16 has historical special protection: original legacy reconstruction/stages were preserved; full reruns were not generally authorized. The v5 object-candidate change and later all 30 v8 isolated smoothing were authorized in their specific context. This does not imply blanket future permission to delete, overwrite, or rerun its original expensive pipeline. Prefer isolated derived candidates; honor the current user's scope if a future plan needs broader work.

At handoff start, Git had no tracked modifications, but there were pre-existing **untracked** `reconstruction/incident_reports/track1_improvement/rollout/*inventory.json` files. They are unrelated local work and were not included in this handoff commit. Do not `git add .`, delete them as cleanup, or treat their uncommitted state as evidence that the scored artifacts are missing.

## 10. Reproducible verification and execution commands

These commands are for the **execution machine**. The planning laptop should cite/adapt them in the returned plan rather than execute them. Commands are examples with clearly described effects. Inspect script defaults and current state before running a write-capable command.

### 10.1 Read-only machine/repository checks

```bash
cd /home/alien51/video_to_data
git status --short
git log -5 --oneline
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
docker ps --format '{{.Names}} {{.Status}}'
df -h /home/alien51/video_to_data
reconstruction/.venv/bin/python reconstruction/scripts/track1_contact_campaign.py --status
```

The last command reads **original campaign progress**, whose original local-pass list can include 16 provisional passes. Final package selection is the 12 in `selection_review.json`; the campaign's `uploaded:false` describes that runner's non-uploading scope, not current overall submission state. Likewise old pre-submission `execution_plan.md` text saying packing is pending is historical. Final `submission_results.md`, `package_validation.json` and exact receipt take precedence.

### 10.2 Read-only final-score recheck; never re-upload to check status

```bash
cd /home/alien51/video_to_data/reconstruction/v2d_submission_kit/v2d_submission_kit
../../.venv/bin/python submit/v2d_submit.py \
  --track track_1 \
  --poll-receipt /home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_contact_smoothing_v8.kaggle-receipt.json
```

This command completed with all five terminal COMPLETE states in the previous turn. A separate authenticated history/leaderboard retrieval for this handoff again matched the scores. Use exact submission IDs for evidence; do not poll "latest" when unrelated uploads may exist.

### 10.3 Targeted tests already available

```bash
cd /home/alien51/video_to_data
reconstruction/.venv/bin/python reconstruction/scripts/test_track1_contact_smoothing.py
```

Eight solver/summary/scored-gap cases passed for v8. The ablation suite has five cases (including SciPy-dependent smoothing), and the fused GPU probe has six geometry comparisons. Run SciPy/GPU tests inside the tested image when needed; the host venv lacks those dependencies. A normal diagnostic container shape is:

```bash
docker run --rm --gpus all --user 1000:1000 \
  --entrypoint python \
  -e LOGNAME=uid1000 -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
  -v /home/alien51/video_to_data/reconstruction:/repo \
  v2d_cari4d:blackwell \
  /repo/scripts/test_track1_episode7_ablation.py
```

The fused test may compile kernels and consume GPU time; it is not necessary for a documentation-only change. Prior suites also cover temporal pose selection, mesh-penetration parity, motion diagnostics, circular hoop fitting, tracking checkpoints and quality gates. Choose tests relevant to actual changes rather than rerunning expensive suites by habit. Do not invent a single aggregate test count across historical experiments with overlapping suites.

### 10.4 Future run/pack/submit sequence

For a future candidate, the returned plan must describe concrete controlled mutations, not just invoke an old all 30 runner with a new version label. Use this sequence:

1. Confirm workspace, actual source commit, process state, data availability and baseline hashes.
2. Define an isolated new experiment name, exact parent package (usually v8), touched fields and unchanged-field invariants.
3. Run a small diagnostic/pilot with fixed seeds and preserved input fingerprints. Measure cost and decide whether the hypothesis is supported before expanding.
4. Complete any authorized all 30 evaluation, select only justified candidates, and retain the chosen parent elsewhere. Preserve hard rejected cases and reasons.
5. Validate all 30 NPZ/GLB pairs, finite arrays, positive scale, orthonormal rotations, frame coverage and mesh integrity. Verify unchanged fields/hashes and ensure packing changed only intended numeric rows.
6. Commit the actual producing source and make the genuine GitHub commit accessible. Pack with that exact full 40-character SHA URL; never fabricate a hash or use a historical unrelated commit because it is convenient.
7. Run official validation, environment preflight, authenticated entry/quota check and uploader dry-run for the exact archive. Record SHA256 and row count.
8. Submit one reviewed version only when within the current user-authorized scope, using a unique exclusive receipt. Distinguish proposing a plan from authorizing an upload. The current request authorizes this status-document commit, not an additional challenge submission.
9. Poll the saved IDs. If an upload outcome is ambiguous, inspect remote history before retrying. Never retry all five after one ambiguous metric. Preserve all previous versions and compare completed official scores.
10. Record actual outcomes, failure cases and next decisions, then commit/push reports without changing the producing-code URL in the archived payload.

Generic official packing arguments are `eval_reconstruction.py --episodes DIR --sample data/track_1_sample_submission.parquet --commit REAL_GITHUB_COMMIT_URL --out NEW_PARQUET`. `tools/preflight.py --track track_1` checks the environment/entry, not all numerical content. `submit/v2d_submit.py NEW_PARQUET --track track_1 -m DESCRIPTION --dry-run` validates without uploading. Removing `--dry-run` performs external uploads; it is not part of preparing this handoff. Use `--wait --receipt NEW_UNIQUE_JSON` for an authorized actual submission.

The final v8 upload guard is `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_contact_smoothing_v8.upload-started.json`; receipt is the path above. Their existence means that version has already been issued. Do not delete them to bypass duplicate protection.

### 10.5 Official row contract and package-level comparisons

Canonical row IDs use `t1_{episode:06d}_{frame:06d}_{array_index}_{point:06d}`. Episode-level arrays use frame 999999. Canonical array order is:

| Index | Array | Scope |
|---:|---|---|
| 0 | `mhr_pose` | Per scored frame; 136 scalars packed three per row |
| 1 | `mhr_scales` | Per episode; 68 scalars |
| 2 | `mhr_shape` | Per episode; 45 scalars |
| 3 | `object_rotation` | Per scored frame; 3 rows of 3 |
| 4 | `object_translation` | Per scored frame; 1 row of 3 |
| 5 | `object_scale` | Per episode;scalar in x, y/z zero |
| 6 | `object_mesh_vertices` | Per episode;fixed template budget |
| 7 | `object_mesh_faces` | Per episode;fixed template budget |

Mesh faces use vertex indices written as numeric triples; padded degenerate faces do not contribute area. Parameter tail padding must be zero. Use the actual official sample's row set and budget, not a guessed schema. Translation rows contain numeric array index 4 in the row ID, not the literal string `object_translation`. The v8 check aligned packed rows by `row_id`, compared every x/y/z with the v6 archive, allowed differences only in promoted translation rows, and detected 2,056 changed rows. This also guards against incidental mesh simplification/re-encoding or body conversion drift.

## 11. Evidence index for the next model/executor

Everything essential to planning is summarized above; these committed files provide deeper detail when available.

| Evidence | What to extract |
|---|---|
| [Initial inventory](reconstruction/incident_reports/track1_submission_inventory.md) and [progress](reconstruction/incident_reports/track1_submission_progress.md) | Original recovery/stage history, first two score vectors; old blockers are historical |
| [Initial improvement plan](reconstruction/incident_reports/track1_score_improvement_plan.md) and `track1_score_audit.json` | Severe initialization failures, tracker-confidence mismatch, old frame diagnostics; not current per-episode hidden scores |
| [V3 results](reconstruction/incident_reports/track1_improvement/rollout/submission_results.md) | Rollout promotion/tradeoffs, producing archive |
| [CARI4D gap analysis](reconstruction/incident_reports/track1_improvement/rollout/cdo_baseline_analysis.md) | Historical baseline verification and limitations |
| [V4 execution](reconstruction/incident_reports/track1_improvement/cdo_v4/execution_plan.md) and [final review](reconstruction/incident_reports/track1_improvement/cdo_v4/final_review.md) | Refinement ablations, rotation selection, rejected/proposed corrections |
| [PEN definition audit](reconstruction/incident_reports/track1_improvement/cdo_v5/penetration_definition_audit.md) and [v4 diagnosis](reconstruction/incident_reports/track1_improvement/cdo_v5/v4_penetration_diagnosis.md) | Exact direction/source of old gate mismatch |
| [V5 final review](reconstruction/incident_reports/track1_improvement/cdo_v5/final_review.md) | Rotation/contact candidates and rejections |
| [V6 final review](reconstruction/incident_reports/track1_improvement/cdo_v6/final_review.md) | Hoop geometry/hand constraints, video scope |
| [V7 decisions](reconstruction/incident_reports/track1_improvement/cdo_v7/episode_decisions.md) and [results](reconstruction/incident_reports/track1_improvement/cdo_v7/submission_results.md) | All 30 sweep, only Episode 7 promotion, hidden spatial regression |
| [Episode 7 ablation](reconstruction/incident_reports/track1_improvement/episode7_translation_ablation/results.md) | Separates unconstrained smoothing and ray depth |
| [Episode 7 constrained pilot](reconstruction/incident_reports/track1_improvement/episode7_contact_smoothing/findings.md) | Exact contact/DP design, independent final check, videos and limitations |
| [V8 execution](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/execution_plan.md) | Fit implementation, caches, frozen evaluator and scored-scope requirements |
| [V8 findings](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/findings.md) | Completed all 30 result and evidence scope |
| [V8 decisions](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/episode_decisions.md) and [selection JSON](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/selection_review.json) | All 30 candidate gains, final 12 selection, rejection reasons and hashes |
| [V8 campaign audit](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/campaign_verification.json) |16,563 frames,33,126 options, original inputs preserved, all selected constraints pass |
| [V8 visual review](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/visual_reviews.json) | Sample frames, specific observations and image hashes |
| [V8 fused validation](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/fused_validation.json) | Numerical equivalence scope and measured speedup |
| [V8 package validation](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/package_validation.json) |740,780 rows, archive hash, actual producing code and packed nontranslation identity |
| [V8 official receipt](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/submission_receipt.json), [comparison](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/score_comparison.json), [final report](reconstruction/incident_reports/track1_improvement/contact_smoothing_v8/submission_results.md) | All five final scores and exact IDs |

Historical sentences saying "v6 remains best" were correct before v8 was scored; v8 is now marginally better on CD-O. Historical `uploaded:false` fields in fit-only manifests are not authoritative submission state. Preserve history rather than rewriting every old report; use final receipts and this dated synthesis for current status.

## 12. Open problems and useful planning directions

These are evidence-driven hypotheses for the planning model to assess, not a preapproved implementation plan or claims of untested gains.

1. **Close the remaining spatial gap with geometry/pose evidence.** V8's motion improvement barely changed CD-O. Reaudit the latest submitted meshes, silhouette boundaries, scale/depth ambiguity, occlusions and camera conventions. Prioritize high-value *local evidence of failure* but do not claim it reveals aggregate hidden per-episode error. V6 shows that correcting one poor object-specific reconstruction can matter more than blanket smoothing.
2. **Handle remaining rotation ambiguity.** Translation-only smoothing cannot fix rotations. Episode 7 still had 48 >45° jumps; other reports expose substantial flips. Distinguish true object rotation from tracker discontinuities using rendered evidence and symmetry-aware hypotheses. Only verified symmetries may be quotiented out; handles, bowl openings and asymmetric shapes matter. Evaluate rotation choices with actual mesh/image/contact evidence, not only angle reduction.
3. **Improve temporal accuracy while preserving genuine motion.** Our ACC-H/ACC-O trail leading teams substantially. A contact-aware SE(3) trajectory estimator or confidence/occlusion-aware smoothing is plausible, but needs its own controlled hypothesis, no-gap motion checks, and real-motion preservation tests. The proven v8 finite translation solver is a conservative fallback, not evidence for arbitrary larger smoothing windows.
4. **Revisit human quality with coupled alignment accounted for.** V2's best CD-H is better than v8's, while later ACC-H is better. Diagnose identity/root/body motion rather than blindly selecting older human arrays. A proposed hybrid must explain how global similarity alignment changes object metrics and hand penetration, and how to validate those interactions locally before upload.
5. **Improve diagnostic reliability.** Track support/visibility when using masks/depth; a few cached monocular pixels cannot resolve metric depth. Compare common pixels, preserve independent frame splits where useful, and expose unsupported frames. Unknown exact host hand subsets remain a limitation. Avoid declaring contact success from near-zero PEN.
6. **Check baseline reproducibility before claiming methodological superiority.** Released checkpoint parity alone is insufficient. Record preprocessing, object geometry/origin/scale, tracker initialization, camera frame, temporal batching, refinement weights and conversion settings. Separate reproducibility repair from novel changes in the ablation table.
7. **Use the leaderboard sparingly.** Broad local diagnostics plus a small controlled experiment should precede any upload. Do not search offsets or per-episode choices by repeatedly querying hidden scores. No hidden ground truth, organizer-private artifacts or future results are available to this planner.

The planner should rank at most a few concrete experiments by expected benefit, evidence, feasibility, risk and cost. A plan that simply reruns every episode with 300 steps or shifts all objects along inferred depth ignores prior negative evidence. A plan that immediately expands to policy training on another track ignores current scope and compute constraints.

## 13. Required output from the planning LLM

**Generate a new, self-contained Markdown file for the executing assistant on the execution machine.** Do not merely discuss possibilities in chat, do not run workloads on the planning laptop, and do not claim fabricated test/leaderboard outcomes. The user will copy your generated `.md` back here for execution. Include the following sections, with concrete details proportional to each experiment:

1. **Objective and measurable success.** State primary metric(s), latest parent package and exact reference score. Separate matching CARI4D from catching the live leader. Define acceptable regressions/tradeoffs; do not silently replace accuracy with smoothness.
2. **Facts, hypotheses and unknowns.** Cite the specific evidence from this handoff. Identify which proposed improvements are untested. Flag missing files you would need the executing assistant to inspect; do not imply you inspected linked files if you only received this document.
3. **Ranked experiment table.** For each candidate: hypothesis, targeted failure, affected episodes/objects, touched arrays/code, parent version, expected mechanism, measured evidence to seek, resource estimate with uncertainty, and abort criteria. Predicted score changes must be labeled guesses, not promised results.
4. **Small pilot design and ablations.** Choose a manageable representative set based on known failure modes. State controlled factors, seeds, data/frame splits, fallback behavior, and how to avoid mixing scale/depth/smoothing changes in one unexplained comparison. Explain how to decide whether expansion to all 30 is justified.
5. **Implementation specification.** Name existing modules to extend, exact interfaces/CLI additions, data shapes, units, transform conventions, source hashes, caches and output layout. New paths must be isolated. Explicitly address hardcoded v6 controls if the new parent is v8. Do not invent flags that existing scripts do not support; label proposed new flags as required implementation work.
6. **Losses/constraints and rationale.** If proposing optimization, write its variables, objectives, masks, robust terms, contact/occlusion treatment, priors and constraints. Explain how genuine motion is preserved and how thin/open/asymmetric objects are handled. Distinguish the optimized local objective from official hidden metrics.
7. **Validation matrix.** Define relevant synthetic/numerical tests, real failure-case regressions, all-scored-frame image/depth/contact checks, gap-safe motion diagnostics, transform/scale checks, full-array integrity, and sampled visual scope. Preserve the corrected PEN direction. Define numerical tolerances and handling of missing evidence.
8. **Compute and scheduling.** Target the actual 24,463 MiB laptop GPU and 31 GiB WSL RAM. State which work runs in host venv versus Docker. Use the existing sm_120-capable image. Bound concurrency, disk/cache growth and runtime; require measurement before extrapolating. Explain checkpoint/resume and recovery from an interrupted job.
9. **Promotion and rollback policy.** Define exact per-episode candidate selection, meaningful-gain thresholds, interaction/motion regression limits and visual review requirements. State when to retain the parent. List immutable artifacts and protected Episode 16 treatment. Retain failed trials and their reasons.
10. **Complete-package assembly.** Require all 30 pairs, exact public row coverage, mesh/rotation/scale validity, unchanged-field/hash comparisons and packed-row diff against the actual chosen baseline. No accidental v6 fallback when the parent is v8.
11. **Commit/provenance and authorized submission procedure.** Use an actual accessible producing-code commit, pack official format, run preflight and authenticated dry-run, respect current policy/quota, then submit only within the user's active authorization. Use exclusive guard/receipt and poll exact IDs. Never schedule duplicate uploads or assume this status file authorizes a new version.
12. **Execution checklist and deliverables.** Provide ordered steps, commands where known, implementation tasks where commands do not yet exist, report paths, decision checkpoints, expected artifacts, and final report contents. Every step must be executable or have an explicit evidence-based decision rule; no vague "optimize until good".
13. **User-facing questions only when necessary.** The executing assistant should continue autonomous reversible authorized work. Surface genuine missing choices (e.g., scope/compute spend/tradeoff) without turning every routine check into a permission gate. Do not embed a blanket approval barrier that contradicts the user bringing the file back with an implementation request.

### 13.1 Suggested structure for the returned file

```markdown
# V2D Next Improvement Plan

## A. Execution context and scope
- Planning laptop: reasoning and Markdown only.
- Execution machine: /home/alien51/video_to_data; one RTX5090 Laptop GPU.
- Parent package: [choose explicitly, normally immutable v8].
- Primary objective and current reference values: ...

## B. Evidence and hypotheses
[Known facts with source references; untested hypotheses; missing information.]

## C. Ranked experiments and pilot
[Limited, controlled experiments; touched fields; measurable stop/go decisions.]

## D. Implementation changes
[Actual files/interfaces, transforms, data contracts, objectives, constraints.]

## E. Validation and promotion
[Tests; all-scored-frame evidence; contact/motion/image checks; fallback rules.]

## F. Resource plan and resume
[VRAM/RAM/disk/runtime estimates, uncertainty, checkpoints, bounded concurrency.]

## G. All 30 package, provenance, submission and rollback
[Integrity checks, actual code commit, official dry-run, authorization scope,
exclusive receipt, exact-ID polling, baseline comparison, immutable originals.]

## H. Ordered execution checklist and final deliverables
[Concrete steps and outputs for the assistant on the execution machine.]
```

### 13.2 Message the user can paste alongside this file

> Read the attached V2D status/handoff Markdown completely. You are the planning model on a separate laptop. Do not run the challenge here. Generate a detailed `V2D_NEXT_IMPROVEMENT_PLAN.md` that I can transfer back to the executing assistant in `/home/alien51/video_to_data` on my other machine. Use the actual submission history and current Kaggle comparisons, account for prior failed experiments and the single RTX 5090 Laptop GPU limits, and propose a controlled next improvement round. Distinguish facts from hypotheses and provide concrete implementation, validation, selection, resume, packaging and submission steps. Do not invent results or assume access to hidden ground truth. The returned Markdown will be used as implementation instructions on the execution machine.

## 14. How this snapshot was checked and how to refresh it

This document was assembled from the eight local immutable Parquets, their exact receipts, committed iteration reports, current metadata/scored-frame lists, direct read-only hardware/runtime probes, and authenticated Kaggle API results. SHA256 and producing URLs were read/recomputed from all eight payloads. The authenticated history returned eight COMPLETE submissions per metric. Each recorded package ID/score was checked against that history. The episode ledger was computed from final submitted v8 pairs plus the public scored indices. Full standings came from `KaggleApi.competition_leaderboard_view(..., page_size=100)`; own submission records came from `competition_submissions(..., page_size=100)`. No new reconstruction or submission was run to produce this document.

Future readers should refresh timestamps, ranks, policy/quota, process state, available disk/RAM, current source commit and parent artifacts before execution. Do not reinterpret a dated snapshot as a promise of current rank or current spare resources. When a future version finishes, append its exact package scores/IDs/provenance and update best-per-metric versus latest-package distinctions. Keep failed versions in the history: their evidence is useful.
