# V2D Cross-Track Improvement Plan — Tracks 1, 2 and 3

**Prepared:** 2026-10-20, from `CHALLENGE_STATUS_AND_LLM_HANDOFF.md`, `TRACK2_STATUS.md`, `TRACK3_STATUS.md`, the v9/v10 Track 1 reports and the **official metric code** on branch `v2d-track1-blackwell` (`reconstruction/v2d_submission_kit/v2d_submission_kit/metric_code/`).
**Supersedes:** the Track 1-only `V2D_NEXT_IMPROVEMENT_PLAN.md` (v9/v10 round, commit `718579c`). That plan's outcomes (v9 rejected, v10 prepared) are inputs here.
**Planning status:** nothing in this file has been executed. Predicted gains are labelled **estimate**. Facts read from metric code or committed reports are labelled **fact** with their source.
**Executor:** the assistant at `/home/alien51/video_to_data` (WSL2, one RTX 5090 Laptop GPU, 24,463 MiB, `sm_120`, ~31 GiB RAM).
**Hard deadline:** challenge freeze **Wednesday 2026-11-04 17:00 EST (22:00 UTC)**. That leaves about 15 days, or roughly **290 usable GPU-hours** after reserving 24 h for packaging and buffer.

---

## 0. Executive summary (read this first)

### 0.1 Highest-value moves, ranked by expected leaderboard gain per GPU-hour

| Rank | Move | Track / metric | Cost | Why it should work (evidence) | Effect (**estimate**) |
|---:|---|---|---|---|---|
| 1 | **Submit v10 now** (already packed and dry-run-verified; held only by weekly quota) | T1 CD-O (Ep18) | 0 GPU-h, 1 upload | v10 audit: Ep18 rotation fix, every other row is byte-identical to v8 | Small CD-O gain; provides the new parent |
| 2 | **Re-centre object mesh origins, then apply evidence-gated strong temporal smoothing to object translation** | T1 ACC-O (+ small CD-O) | ~1 day CPU, <5 GPU-h | **Fact:** ACC-O = mean ‖Δ²t_pred − Δ²t_ref‖ over contiguous scored stretches (`ACC-O.py` L261–277, L672–697). Three teams with poor CD-O share ACC-O = 0.07429, the signature of near-zero predicted acceleration. Leader 0.07111; ours 0.71908. | 0.72 → **0.08–0.15**; position 21/26 → ~9–13 |
| 3 | **Joint-space temporal refit of the human** (not raw-parameter smoothing) | T1 ACC-H (+ maybe CD-H) | 2–3 days of implementation, ~10 GPU-h | **Fact:** ACC-H uses decoded MHR body joints with the same Δ² error. Several teams sit at 0.137–0.145; ours is 1.159. v9 failed because it smoothed Euler parameters (115–174 cm p99 deformation), not because smoothing is wrong. | 1.16 → **0.15–0.30**; position 23/26 → ~10–18 |
| 4 | **Zero-penetration projection inside the object smoother** | T1 PEN | Same run as #2 | **Fact:** PEN = mean relu(−sdf) of host hand points inside the submitted mesh (`ACC-O.py` L874–882). Several teams have 0.00000; ours is 0.00322 (v8). Pushing every vertex of a generous superset of hand vertices outside the mesh makes PEN exactly 0 for any host subset. | 0.0032 → **0.0000–0.0002**; position 14 → ~1–6 |
| 5 | **Submit Track 2 Tier 2 and Tier 3 baselines** (currently absent) | T2 Tier 2/3, all metrics | ~20 GPU-h per tier | No entry exists on those 8 leaderboards. The Tier 1 short pipeline is proven end to end. | "No entry" → ranked on 8 leaderboards |
| 6 | **Reallocate policy budget and select checkpoints with the exactly reproducible local scorer** | T2 Tier 1 (AUC, SP-SR, MP-SR, MPPE) | 60–90 GPU-h | **Fact:** for Tier 1 the packer's local scores matched Kaggle to every printed digit (0.24466 / 0.10 / 0.03333 / 53.84820). Every decision can be made offline. Only Ep16 received a large budget (350M); the other 29 received 5M. | SP-SR 0.10 → **0.20–0.40** |
| 7 | **Run a Track 3 dev-set decomposition** (oracle reference vs. reconstructed reference), then fix the binding constraint | T3, all six metrics | ~75 GPU-h total | **Fact:** public dev episodes include scans and reference trajectories, and the official scorer runs locally. All local success is 0 today, even Ep38 at 60M steps. | Cannot be estimated until §5.1. Target: SP-SR 0.05 → 0.20+, CD-O 0.57 → 0.30 |

Track 3 is the hardest track: we are last or near last on all six metrics. Its plan therefore **starts with diagnosis** (§5.1) to determine whether reconstruction or policy is the binding constraint before more GPU time is spent.

### 0.2 What *not* to do (negative evidence carried forward)

- Do not repeat raw-Euler MHR parameter smoothing (v9: 0/7 human passes at every λ).
- Do not apply inferred-depth (MoGe) ray shifts on Track 1 objects (v7: CD-O worsened despite better local depth).
- Do not rerun all Track 1 episodes with 300 refinement steps (no flip fix on Ep12).
- Do not use Track 2 Tier 1 *clean* trajectories to repair or select anything for Tier 2/3 (§4.1).
- Do not substitute public development scans for Track 3 evaluation meshes (FAQ rule), even though some leaders' CD-O (0.065 cm) looks scan-like.
- Do not cherry-pick the best of N stochastic rollouts per episode. Use the evaluator's deterministic, documented single-world rollout.
- Do not upload to discover which variant is better when a local exact (T2 Tier 1) or decision-safe proxy exists.
- Do not exceed 5 uploads per week per competition again (7 were made in week Oct 12; see §C.2).

---

## A. Execution context, deadlines and budgets

### A.1 Calendar (verify the week boundary in §C.2)

| Window | Dates | Track 1 | Track 2 (per tier) | Track 3 |
|---|---|---|---|---|
| Week W1 | Mon 10-19 → Sun 10-25 | 5 (the v10 hold should be lifted if weeks start Monday) | Tier 1: check used count; Tier 2: 5; Tier 3: 5 | Check used count (one upload was made around 10-19/20 UTC) |
| Week W2 | Mon 10-26 → Sun 11-01 | 5 | 5 each | 5 |
| Final 3 days | Mon 11-02 → Wed 11-04 17:00 EST | Unlimited (official site; confirm) | Unlimited | Unlimited |

Kaggle's own daily quota still applies. Never use the final three days as the *first* test of a new pipeline: scoring can lag, and errors there are unrecoverable.

### A.2 GPU-hour budget (single GPU, sequential heavy jobs)

| Bucket | GPU-h | Notes |
|---|---:|---|
| T1 v11/v12 (smoothing, refit, PEN, reviews) | 15 | Mostly CPU. GPU only for the MHR refit and fused winding checks. Can run beside training if total VRAM stays < 21,000 MiB. |
| T2 Tier 2 baseline (30 eps × 5M) | 20 | ~30–35 min/episode incl. eval (derived from status timestamps). Re-measure. |
| T2 Tier 3 baseline (30 eps × 5M) | 20 | Same pipeline, different input. |
| T2 Tier 1 pilot (§4.3) | 20 | 3 arms × 3 episodes × ~20M |
| T2 Tier 1 scale-up (§4.5) | 60 | Successive halving |
| T3 dev decomposition + mesh reconstruction (§5.1–5.2) | 20 | |
| T3 reference repair + policy pilot (§5.3–5.4) | 15 | |
| T3 scale-up (§5.5) | 40 | 20 eps × ~40M at ~6.4k steps/s ≈ 35 h |
| **Buffer** (WSL driver stalls, reruns, compile) | 80 | The Oct 17/18 incidents cost many hours. |
| **Total** | **290** | |

**GPU scheduling rule:** only one training job at a time. CPU-only Track 1 work runs in parallel. Every GPU job uses the existing 21,000 MiB guard and writes a heartbeat file. If no heartbeat appears for 15 minutes, inspect the job before restarting it. Never kill a job that is still writing.

### A.3 Parent artifacts (immutable)

- **T1:**
  - v8: `track1_contact_smoothing_v8.parquet` (SHA `bd73a931…3198`), episode pairs in `reconstruction/data/v2d_challenge/track_1/contact_smoothing_v8/submission_episodes/`.
  - v10 candidate: SHA `78d56ea2…e87a`, commit `8feab58e16031461639d7e3a1ec7dbef1705c3a3`. Only Ep18 rotations differ from v8.
- **T2 Tier 1:** packet `robotic_grounding/data/track2/tier1/submission_short/track2_tier1.csv` and policies under `robotic_grounding/data/track2/tier1/`.
- **T3:** packet `robotic_grounding/data/track3/submission_today/track3_submission.parquet`, policies `policies_today/`, rollouts `results_today/`, meshes `submission_meshes/`.

Take the exact full SHA-256 values from the handoff or recompute them. All new work goes in **new, versioned directories**, and every new `execution.json` records the parent SHA-256s.

---

## B. Facts from the official metric code that drive this plan

These are read from the metric scripts. Re-confirm line numbers on the current checkout before relying on them.

### B.1 Track 1 (`metric_code/track_1/*.py`: identical code, different `_METRIC`)

1. **One similarity transform per episode**, fit on `mhr_vertices[0]`: the alignment-role vertices of the *first scored frame*, fit to the reference (`_episode_metrics`, ~L682). It is applied to human joints, human surface, object surface and object translation. **Consequences:**
   - Any change to the first scored frame's human pose changes *all five* metrics for that episode. **Anchor that frame in every smoother.**
   - Object error is effectively measured **relative to the human**.
2. **ACC-H / ACC-O** (L261–277, L687–697): computed per contiguous scored stretch (`_stretches`, L672) as `a = x[:-2] − 2x[1:-1] + x[2:]`; error = ‖a_pred − a_ref‖, averaged over joints (ACC-H), then frames, then episodes.
   - ACC-O uses **`object_translation` only**: the mesh *origin*, not the centroid and not the rotation.
   - A prediction with zero acceleration scores exactly mean‖a_ref‖. The ties at 0.07429 (ACC-O) and 0.14528 (ACC-H) among teams with poor CD scores strongly suggest the reference's own acceleration magnitudes are ≈0.074 (object) and ≈0.145 (body joints), in metric units.
   - **Inference:** our 0.72 / 1.16 is ~10× the reference's own acceleration. Our error is dominated by predicted jitter, not missing true motion. Even the leaders (0.071 / 0.137) beat "static" by only 4–6%. **Target: ≤ 0.09 (ACC-O) and ≤ 0.17 (ACC-H).**
3. **CD-O / CD-H:** per-frame symmetric Chamfer on surface samples after the similarity transform. Object samples are drawn area-proportionally from our mesh with a fixed seed, so disconnected floater components receive samples in proportion to their area.
4. **PEN** (~L874–882): predicted hand vertices (host roles `left_hand`/`right_hand`; the exact host index subset is not visible to us) inside the submitted mesh by |winding| > 0.5, depth × object scale × alignment scale, `relu(−sdf)` mean.
   - If **no** vertex from a superset of all plausible hand vertices is inside the mesh in any scored frame, PEN is exactly 0 regardless of the host subset.
5. **Re-centring a mesh is metric-neutral for CD-O, CD-H and PEN.** Replacing vertices `p → p − p̄` and `t → t + s·R·p̄` yields identical world surfaces. Only `object_translation` changes, so only ACC-O changes.
   - Area-weighted sampling with the same seed on the same faces produces the same barycentric samples, so the world points are identical up to floating-point error.
   - Still verify this numerically with the official code (test T1-U3 in §3.1).

### B.2 Track 2 (`metric_code/track_2/*.py`)

1. Achieved trajectories are **warped to the reference start pose** (`_warp_to_reference_start`, ~L425) before every metric. Initial placement offsets do not matter; drift and failure to follow do.
2. **SP-SR:** an episode succeeds if mean position error ≤ **10 cm** and mean orientation error ≤ **0.5 rad** over the whole trajectory (L19–30).
3. **MP-SR:** mean position error < **3 cm** and mean orientation error < **30°**, per object.
4. **AUC:** ADD (mean vertex distance) AUC over thresholds 1–9 cm.
5. **MPPE:** six keypoints at ±5 cm along body axes, mean distance.
6. The score is the mean over episodes, so SP-SR/MP-SR are multiples of 1/30. **Each extra successful episode adds +0.0333.**
7. **Tier 1 references are the released clean trajectories.** Local packer recomputation equalled the Kaggle public score. **Tier 1 can be optimised offline exactly.** Tier 2/3 references are presumably the same clean trajectories, but we must not use them for selection (§C.2 Q4).

### B.3 Track 3 (`metric_code/track_3/*.py`)

1. Same warp-to-start (`_ALIGN_INITIAL_POSE = True`); body 0 drives the warp. Pose metrics are **symmetry-aware**.
2. `_REGISTER_OBJECT_MESH = True`: our mesh is registered to the reference scan to relate body frames. **A better mesh shape also improves pose metrics**, not only CD-O.
3. **CD-O is shape-only** (CD-O.py L111–186). The candidate is rescaled so its area-weighted radius equals the reference's, then aligned with trimmed ICP from several PCA starts, then scored as symmetric mean NN distance × 100. Absolute scale and placement do not matter; proportions, floaters, holes, thickness and surface noise do.
4. **RPE = 0 for single-object episodes.** With 36 objects in 20 episodes, 16 episodes have two objects. Our 22.29 cm mean therefore implies about **27.9 cm** on the two-object episodes (estimate; recompute locally).
5. SP-SR/MP-SR thresholds are identical to Track 2.

---

## C. Phase 0: refresh leaderboards, confirm rules, build per-episode diagnostics (Day 0, ≤ 4 h, no GPU)

Phase 0 blocks every upload decision. It does not block local computation.

### C.1 Refresh leaderboards (read-only)

Covers 5 T1 + 4 T2 Tier 1 + 4 T2 Tier 2 + 4 T2 Tier 3 + 6 T3 competitions. Use the existing authenticated pattern `KaggleApi.competition_leaderboard_view(slug)` and `competition_submissions(slug)`, and save each to `reconstruction/incident_reports/leaderboard_snapshots/2026-10-2X/<slug>.json`. Then:

- Take the Track 2 Tier 2/3 slugs from the official site; do not guess them.
- Record, per competition: our best score, our latest score, the leader's score, and the scores at positions 3, 5 and 10. These are the targets in the decision rules below.
- Check whether the 0.07429 / 0.14528 "static" signature is still present, and whether newer teams cluster near 0.07. Either result tests §B.1.2.

### C.2 Rules to confirm

Write the answers into `reconstruction/incident_reports/rules_check_2026-10-2X.md`, quoting the text from the official site or FAQ PDF that each answer relies on.

1. **Weekly window:** when does it start (Monday? rolling 7 days?), and is the "unlimited last three days" period Nov 2–4? Determine how the 7 uploads in week Oct 12 were counted, and whether any penalty applies.
2. **May different packages go to different per-metric competitions of the same track?** (For example, a smoother package to ACC-O only.)
   - **Default: NO.** Every upload in this plan sends one coherent package to all metrics of its track.
   - If the rules explicitly permit it, record the quote. Even then, prefer coherent packages, because a combined ranking may use only one package.
3. **Kaggle final selection:** how many submissions per competition count at freeze, and must we "select" them manually? Add a calendar item for 11-04 12:00 EST to verify the selections.
4. **Track 2 Tier 2/3:** confirm the released inputs (paths in the HF dataset) and that denoising the noisy input with our own filter is allowed (expected yes). Ask whether Tier 1 clean data may be used for *model selection* on Tier 2/3. **Default: NO.**
5. **Policy initialisation:** may a policy be warm-started from another episode's trained policy with the same embodiment? Expected yes, since it is still a learned policy with a distinct checkpoint per episode, but confirm. The packer currently checks for "distinct per-episode checkpoints".
6. **Track 3:**
   - Confirm floating Sharpa hands remain allowed.
   - Confirm that development scans and trajectories may be used for *method development and local validation*, since they are public development assets.
   - Confirm they may not be substituted into evaluation meshes or poses.
7. **Track 1:** are object templates or meshes provided for evaluation episodes? Expected no; the handoff says we reconstruct them.

### C.3 Per-episode diagnostic tables (CPU only)

Create `reports/phase0/` and produce three tables.

**T1:** per episode for v8 and v10, on scored frames only:

- Mean ‖Δ²t‖ (translation), ‖Δ²c‖ (mesh centroid) and ‖Δ²J‖ (metric body joints, `MHR_TABLE3_BODY_JOINT_INDICES`).
- Joint acceleration split into root-translation, root-rotation and articulation contributions (§3.3.1).
- Mesh origin offset s·‖p̄‖ (m), fraction of mesh area outside the largest connected component, and mask IoU (from the v10 geometry audit).
- PEN proxy (all hand vertices) and the number of scored frames with any penetrating hand vertex.

**T2 Tier 1:** per episode, from stored trajectories and the local official scorer:

- AUC, SP-SR flag, MP-SR flag, MPPE, mean position error and mean orientation error.
- Training steps and the last curriculum stage reached.
- `pos_margin = mean_pos − 0.10 m` and `ori_margin = mean_ori − 0.5 rad`. These identify **near-miss** episodes.
- Episode 16 (350M steps) is the reference point: if it succeeds and most 5M episodes do not, budget is the binding constraint.

**T3:** per episode and object, local diagnostics against our reconstructed reference:

- Local tracking position and orientation errors, and an MPPE-like error.
- Number of frames where the reference object moves with no hand within 2 cm ("floating motion", a sign of physical infeasibility).
- Reference object height relative to the estimated support plane (penetration or floating), and the tilt of the gravity estimate.

**Decision outputs from Phase 0:**

- the list of T2 near-miss episodes (feeds §4.5);
- whether T1 joint jitter is mostly root (cheap fix T1-B1) or articulation (needs T1-B2);
- the T3 episodes with infeasible references (feeds §5.3).

---

## 3. Track 1: reconstruction (v10 → v11 → v12)

**Goal:** large rank gains on ACC-O, ACC-H and PEN without giving back CD-O/CD-H. All Track 1 work is CPU-dominant and runs in parallel with Track 2/3 training.

**Shared output contract:** every candidate is a full 30-episode `pred_*.parquet` / `meta.json` package. It must pass `eval_reconstruction.py` and the submission kit's validation and dry-run before any upload.

### 3.0 T1-D: submit v10 (Day 0–1)

- **Preconditions:** §C.2 Q1 confirms the new week has started and the W1 count is < 5. Revalidate the v10 package SHA (`78d56ea2…e87a`) against `package_validation.json`.
- **Action:** one upload to each of the 5 Track 1 competitions, using the existing exclusive receipt flow and `--poll-receipt`. Never re-upload a package that already has a receipt.
- **Record:** the public scores. If v10's CD-O is better than v8's, v10 becomes the parent of v11. Otherwise v8 remains the parent; the Ep18 change is then reconsidered within v11, not resubmitted alone.
- **Cost:** 5 uploads (1 per competition). It does not block anything else.

### 3.1 Verification harness and unit tests (Day 0–1, CPU)

Build `reconstruction/scripts/track1_metric_harness.py`. It imports the official metric functions unchanged (copy, never edit) and:

- **T1-U1 (self-consistency):** scores v8 against itself; all errors must be 0.
- **T1-U2 (perturbation sensitivity):** adds Gaussian translation jitter of σ ∈ {0.1, 0.5, 1} cm to v8 objects. The measured ACC-O against v8 must grow as about √6·σ.
- **T1-U3 (re-centring neutrality):** re-centres every mesh (`p → p − p̄`, `t → t + sR·p̄`) and scores it against the original as the reference.
  - CD-O, CD-H and PEN must be identical to within 1e-6.
  - Only ACC-O may change.
- **T1-U4 (anchor):** any smoother must leave the first scored frame's `mhr_vertices` bit-identical. Assert this.
- **T1-U5 (pseudo-reference proxy):** the true reference is unavailable. As a *relative* proxy, score each candidate against v8 and report the change in mean‖Δ²‖ per stream.
  - The decision metric for ACC is the candidate's **own** mean‖Δ²x‖ (lower is better, down to about the reference floor of ~0.074 / ~0.145).
  - The guard metrics are the image-evidence gates in §3.2.

**Why the proxy is defensible:** by the triangle inequality, ‖a_pred − a_ref‖ ≤ ‖a_pred‖ + ‖a_ref‖. With ‖a_ref‖ ≈ 0.074, bringing ‖a_pred‖ from ~0.7 down to ≤ 0.03 bounds ACC-O by ~0.10. This guarantee does not depend on the reference.

### 3.2 T1-A: object stream (v11 core)

#### 3.2.1 T1-A0: diagnostic and re-centring (2 h)

- For each episode, compute mean‖Δ²t‖, mean‖Δ²c‖ and the "rotation leak" mean‖Δ²(sR·p̄)‖.
- The v8 DP minimised **centroid** acceleration, but the metric reads **t**. **Hypothesis:** a large share of v8's ACC-O comes from rotation jitter × origin offset.
- Re-centre every object mesh at its area-weighted surface centroid. This is metric-neutral for CD/PEN (T1-U3) and immediately makes t equal the centroid.
- **Gate:** T1-U3 passes. Report, per episode, the predicted ACC-O change.

#### 3.2.2 T1-A1: robust translation smoother (1 day)

Reuse the v9 object fits and inputs: the per-frame object pose, the mask, the contact estimates and the scored-frame stretches.

For each contiguous scored stretch, solve a convex QP (cvxpy or OSQP in the `v2d_cari4d:blackwell` image; the host venv has no SciPy):

```
minimize_t   Σ_f w_f · ρ(‖t_f − t̃_f‖)         # data term, Huber, weighted by per-frame evidence
           + λ_a Σ_f ‖t_{f-1} − 2t_f + t_{f+1}‖²   # acceleration
           + λ_c Σ_{f∈contact} ‖t_f − (h_f + d_f)‖² # keep hand-relative offset during contact
subject to   t_{f0} = t̃_{f0}  for the first scored frame of the episode (anchor)
             PEN constraints (§3.4), linearised
```

- t̃ is v8/v10 after re-centring. h_f is the contact hand point; d_f is the median hand-to-object offset over the contact window.
- Evidence weights w_f: high for frames with good mask IoU and visible contact; low for occluded or low-IoU frames.
- **Sweep:** λ_a ∈ {3, 10, 30, 100, 300}. v9 measured: λ=10 → 0.191 cm acceleration with a 1.93 cm bound; λ=30 → 0.135 cm with a 2.57 cm bound.
- **Replace v9's 2 cm hard cap with evidence gates.** A candidate λ is accepted per episode if all of the following hold:
  1. Mean displacement ≤ 0.5 cm and p99 displacement ≤ 3.0 cm relative to t̃.
  2. Projected mask IoU drops by no more than 0.01 (mean) and 0.05 (worst frame).
  3. The median contact-window hand–object gap stays within 1 cm of v8's.
  4. There is no new penetration (§3.4).
- Choose the **largest** λ_a passing all gates per episode. Episodes that fail at λ=3 keep v8/v10 translations, re-centred only.
- **Expected outcome (estimate):** mean‖Δ²t‖ ≤ 0.05 on ≥ 20/30 episodes. Ep7, 8, 18 and 22 may fail because of large rotation jumps; handle them through T1-A2.

#### 3.2.3 T1-A2: rotation stream (optional, 0.5 day)

ACC-O ignores rotation, but rotation jumps of > 45° (Ep7: 48, Ep18: 37, Ep22: 33, Ep8: 31, Ep5: 30) hurt CD-O, and after re-centring they no longer leak into t.

- Smooth rotations on SO(3) with a geodesic-acceleration penalty.
- Allow symmetric flips only when the mesh has a measured symmetry, and only when the mask IoU after the flip is no worse.
- **Gate:** mean IoU does not drop, and Ep18 is not regressed relative to v10.
- **Priority:** lower than T1-A1, T1-B and T1-C. Run it only if time permits on Day 3–4.

#### 3.2.4 T1-E: paint roller (Ep18–20) geometry (0.5 day)

- 72% of the mesh area lies outside the largest connected component, and IoU is 0.41–0.45.
- **Test (local, mask-based):**
  1. Score the projected mask IoU with (a) the full mesh, (b) only the largest connected component, (c) the largest component plus any component supported by ≥ 30% mask coverage across frames.
  2. If (b) or (c) improves IoU on all three episodes by ≥ 0.05, replace the mesh. Rebuild t with T1-A0 so the world surface stays continuous.
- **Risk:** removing true parts of the object (the handle) would worsen CD-O. The mask must support the decision.
- Apply the same test to white_laptop_cart (Ep29), stool (Ep26) and desk (Ep10). Replace a mesh only where the gate passes.

### 3.3 T1-B: human stream (ACC-H)

#### 3.3.1 Diagnostic first (2 h)

Decompose each episode's body-joint Δ² into:

- root translation (global translation parameters);
- root rotation × joint radius;
- articulation residual.

Use the decoded MHR pipeline in the official code (`_MHRBody`, ACC-O.py L485–546): params `[pose 136, scales 68]` → `parameter_transform` → 127 joints × 7 → forward kinematics → corrective MLP → linear blend skinning → /100 × FLIP.

Identify which `parameter_transform` rows drive the root translation and root rotation.

**Decision:**

- If root components account for ≥ 60% of mean‖Δ²J‖, run **T1-B1**.
- Otherwise run **T1-B2**, with T1-B1 as a fallback.

#### 3.3.2 T1-B1: root-only smoothing (0.5 day)

- Smooth **only** the root translation parameters, which act linearly on all joint positions, with the same QP as T1-A1. The first scored frame is anchored.
- Smooth the root rotation with a geodesic penalty on SO(3), not on Euler angles.
- **Coupling:** the hand moves with the root. Re-run the object contact term (λ_c) after the human change so hand–object offsets stay consistent. Run T1-A and T1-B jointly or alternate them twice.
- **Gates:**
  - Body-vertex p99 displacement ≤ 3 cm. Hand vertices during contact ≤ 1.5 cm.
  - The 2D keypoint reprojection error does not increase by more than 5% (mean).
  - The first scored frame is bit-identical (T1-U4).

#### 3.3.3 T1-B2: differentiable joint-space refit (2 days)

- Port `_MHRBody` to torch: FK, corrective MLP and LBS, using float64 for validation. Verify it against the NumPy official decode to within 1e-5 m on 100 random frames.
- Optimise the per-frame pose parameters:

  ```
  L = Σ_f ‖J(θ_f) − J̃_f‖²              # stay close to v8 joints (metric joints + hands)
    + λ_a Σ_f ‖Δ²J(θ)_f‖²            # acceleration in joint space (the metric space)
    + λ_θ Σ_f ‖θ_f − θ̃_f‖²_W          # parameter prior (prevents v9-style deformation)
    + λ_2D Σ_f reprojection(J(θ_f))  # image evidence, if 2D keypoints are stored
  ```

  - Hold θ_{f0} fixed (anchor) and keep the scale parameters constant per episode.
- **Gates:** the same as T1-B1, plus no vertex displacement above 5 cm anywhere. This is the explicit guard against the v9 failure (115–174 cm).
- **Abort criterion:** if the torch port does not match the official decode after 1 day, stop and ship T1-B1 only.

### 3.4 T1-C: PEN → 0 (within T1-A)

- Hand point set H_f is the union of **all** vertices of both hand roles, plus a 2 mm dilation along the vertex normals: a superset of any host subset.
- For each frame, compute the generalised winding number of the object mesh at H_f, using the same implementation as the official code.
- In the object QP, add linearised half-space constraints for any inside or near points: `n_i · (x_i(t_f) − y_i) ≥ ε`, with ε = 1 mm and n_i the surface normal at the closest point y_i.
- Iterate the QP with constraint updates up to 5 times.
- **Fallback:** if an occasional frame stays infeasible, apply a minimal rigid push of the object along the mean penetration normal, then re-smooth with that frame fixed.
- **Gate:** the PEN proxy over all hand vertices is exactly 0 on every scored frame. ACC and evidence gates are unchanged.

### 3.5 T1-F: contact-anchored depth (low priority, Day 6+)

- Use contact windows to tie object depth to hand depth. This targets CD-O on episodes with bad depth.
- Run it only if T1-A/B/C ship by Day 4 and there is quota left. Use the same evidence gates.

### 3.6 Packaging sequence for Track 1

| Package | Contents | Target day | Upload condition |
|---|---|---|---|
| v10 | Ep18 rotations | Day 0–1 | §3.0 |
| **v11** | Parent (v8 or v10) + T1-A0/A1 + T1-C + T1-B1 | Day 3 | All gates pass on ≥ 25/30 episodes; failing episodes are copied from the parent unchanged |
| **v12** | v11 + T1-B2 (if it passes) + T1-E (if it passes) + T1-A2 | Day 7–8 | All gates pass; local proxies are no worse than v11 on every stream |
| v13 (optional) | Final λ tuning informed by v11/v12 public scores | W2 / final 3 days | Only if v11 → v12 public changes indicate a clear direction |

**Public-score sanity checks after v11 (estimates):**

- ACC-O ≤ 0.20 confirms the mechanism. If ACC-O is ≥ 0.6, stop and re-check the re-centring and translation semantics against `ACC-O.py` before submitting more.
- If CD-O rises by more than 0.3 relative to the parent, the evidence gates are too loose. Tighten them in v12 (halve the allowed mean displacement).
- If PEN is not ≤ 0.0005, the host hand subset or the scale convention differs from our assumption. Inspect L874–882 again.

---

## 4. Track 2: G1 + Dex3 humanoid (Tier 1, 2, 3)

**Current Tier 1 scores:** AUC 0.24466, SP-SR 0.10 (3/30), MP-SR 0.03333 (1/30), MPPE 53.85 cm. Ep16 was trained for 350M steps; the other 29 episodes for 5,000,192 steps each. Tiers 2 and 3 have no submissions.

**Known settings:** FlashCHORD/FlashSAC, `sim.njmax=4096`, `nconmax=1024`, 1024 worlds, ~3k steps/s, 21,000 MiB GPU guard. Driver: `robotic_grounding/scripts/run_track2_tier1_campaign.py`.

### 4.1 Tier separation rules (fixed)

- Each tier trains on **its own tier's input trajectories only**.
- Tier 1 clean data is never used to train, denoise, select or evaluate Tier 2/3 policies, unless §C.2 Q4 explicitly permits it.
- Tier 2/3 model selection uses **tier-internal** proxies only: tracking error against the tier's own (noisy or reconstructed) input reference, plus physical plausibility.
- Every Tier 2/3 artifact stores `input_source` and its SHA in `execution.json`. The packer refuses a package whose inputs do not match its tier.

### 4.2 Tier 2 and Tier 3 baselines (Days 1–3; ~40 GPU-h total)

1. **Inputs:** download the Tier 2 (noisy) and Tier 3 (reconstructed) inputs. Record the paths and SHAs.
2. **Input conditioning (tier-internal, CPU):**
   - Tier 2: robust temporal filtering of the noisy reference (Savitzky–Golay or a Huber QP on positions; SO(3) smoothing on orientations).
   - Then enforce kinematic feasibility: joint limits, foot–ground contact and no ground penetration.
   - Tier 3: the same, plus object-support checks (the object rests on a surface or in a hand).
   - **Gate:** the filtered trajectory stays within the noise scale of the input (estimate it from the high-frequency residual). Do not over-smooth real motion.
3. **Training:** the same short pipeline as Tier 1 (5M steps/episode, same curriculum) on the conditioned inputs. Run Tier 2 first, then Tier 3.
4. **Packaging:** the existing packer, with the tier flag. Validate, dry-run, then **one** upload per tier competition.
5. **Ablation (only if time permits):** raw vs. filtered inputs on 3 episodes, judged by the tier-internal proxy. Ship filtered by default: a policy trained on noise-free targets tracks better, and the score is evaluated against clean references (expected, not verified).

**Throughput note:** if the observed rate is ≤ 2.5k steps/s, cut the per-episode budget to 4M rather than letting the run spill into Tier 1 time.

### 4.3 Tier 1 pilot: how should extra budget be spent? (Days 2–4; ~20 GPU-h)

**Episodes:** choose 3 near-miss episodes from §C.3: mean position error in 10–20 cm, or position OK but orientation above 0.5 rad.

**Arms** (each ~20M total steps per episode):

| Arm | Description |
|---|---|
| R (resume) | Continue the existing 5M checkpoint, final curriculum stage, for +15M |
| F (fresh, scaled) | Train from scratch with the 11-stage curriculum scaled to a 20M budget |
| W (warm-start) | Initialise from Ep16's 350M policy (only if §C.2 Q5 allows), short curriculum, 15M |

**Scoring:** the local official scorer every 2.5M steps, using deterministic single-world evaluation (the same as the packer). Save every checkpoint that is scored.

**Decision:** choose the arm with the best mean AUC at 20M, tie-broken by SP-SR count. If W > R by ≥ 0.05 AUC, warm-start becomes the default for the scale-up.

### 4.4 Checkpoint selection (free; apply immediately)

- For **every** existing Tier 1 policy with intermediate checkpoints, score each checkpoint with the official local scorer. Keep the best by the following rule: `SP-SR flag`, then `MP-SR flag`, then AUC, then −MPPE.
- Tier 1's reference is the clean released trajectory, so local selection is exact. This is legitimate model selection on the released data, not test-set probing.
- **Package `t1-v2`** = the best checkpoints. Upload it only if it improves local AUC or the SP-SR count.

### 4.5 Tier 1 scale-up by successive halving (Days 4–10; ~60 GPU-h)

Use the winning arm from §4.3.

| Round | Episodes | Additional steps | Approx. GPU-h |
|---|---|---|---:|
| 1 | All 29 non-Ep16 episodes | → 15M each | ~25 |
| 2 | Top 15 by "closeness to threshold" | +15M (→ 30M) | ~20 |
| 3 | Top 6–8 near-miss episodes | +30M (→ 60M) | ~15 |

**Closeness score:**

```
min(max(pos_err / 0.10, ori_err / 0.5), 3)
```

Episodes already succeeding on both SP-SR and MP-SR are excluded from further rounds unless their AUC is < 0.6.

**Expected outcome (estimate):** 3/30 → 6–12/30 SP-SR successes. Each success adds +0.033 SP-SR, and AUC/MPPE improve along with it.

**Packaging:**

- `t1-v3` after Round 1.
- `t1-v4` after Round 3.
- Final Tier 1 package: the best-by-local-score checkpoints for all 30 episodes.

### 4.6 Tier 2/3 improvement (Days 8–12, if budget allows)

- Apply the winning arm from §4.3 to Tier 2/3, choosing episodes by the tier-internal proxy only.
- If the Tier 1 pilot shows warm-start works, warm-start Tier 2/3 episodes from their **own tier's** policies.
  - Cross-tier warm-start (from a Tier 1 policy) is a gray area. Do not do it unless the rules explicitly permit it.
- Submit at most 2 more packages per tier.

### 4.7 Track 2 failure handling

- **WSL/GPU stall:** keep the heartbeat and 21,000 MiB guard. Resume from the last checkpoint and never restart an episode from scratch. Log each incident under `reconstruction/incident_reports/`.
- **Simulator contact overflow:** raise `njmax`/`nconmax` only for the affected episode, and record the change.
- **Policy collapse during resume:** if AUC drops by > 0.1 between evaluations, revert to the best checkpoint. Retry with half the learning rate once, then stop for that episode.

---

## 5. Track 3: floating Sharpa hands (20 episodes, 36 objects)

**Current scores:** AUC 0.16307, SP-SR 0.05, MP-SR 0, RPE 22.29 cm, MPPE 20.82 cm, CD-O 0.57037 cm. Leaders are at 0.823 / 0.85 / 0.70 / 3.05 / 2.62 / 0.065.

**Known settings:** 5M steps per episode (Ep38: 60M; native budget 250M), 11-stage curriculum, ~6,379 steps/s at 512 worlds. Gravity comes from a PCA heuristic. All local tracking success is 0.

**Key uncertainty:** the leaders' gap is ~5–8×. We do not know whether the binding constraint is (a) our reconstructed object trajectories and meshes, (b) world/gravity alignment, or (c) policy training. §5.1 answers this before scaling anything.

### 5.1 Dev-set decomposition (Days 1–3; ~15 GPU-h)

**Setup:**

- Pick 4 public dev episodes: 2 single-object and 2 two-object, with object classes similar to the evaluation set.
- For each, use the official dev scan and reference trajectory as the **oracle**.

**Experiments** (each 4 episodes × 10M steps):

| ID | Reference trajectory | Mesh | World/gravity | Diagnoses |
|---|---|---|---|---|
| O | Oracle (dev reference) | Oracle (dev scan) | Oracle | Policy ceiling at 10M |
| M | Oracle | **Ours** (our pipeline run on the dev video) | Oracle | Mesh cost |
| R | **Ours** | Oracle | **Ours** | Trajectory + world cost |
| A | **Ours** | **Ours** | **Ours** | Full pipeline (should match the evaluation behaviour) |

Score each experiment with the official Track 3 scorer against the **oracle** reference and scan.

**Decision rules (estimates):**

- If O itself has SP-SR ≈ 0 at 10M, **policy training is the bottleneck.** Prioritise §5.4/5.5: reward weights, curriculum, budget.
- If O ≫ R and O ≈ M, **trajectory/world reconstruction is the bottleneck.** Prioritise §5.3.
- If O ≫ M, **mesh quality is the bottleneck** (it also drives CD-O). Prioritise §5.2.
- Also report CD-O for our dev meshes against the dev scans. This gives the **local CD-O** for each mesh method in §5.2.

**Using dev data:** dev scans are used only for local evaluation and development (§C.2 Q6). They are never substituted into evaluation packages.

### 5.2 Mesh improvement (Days 3–6; ~10 GPU-h)

**Candidate methods**, all validated with the official CD-O code against dev scans:

1. Current pipeline (baseline).
2. Largest-connected-component cleanup plus hole filling plus Laplacian/Taubin smoothing.
3. Multi-view BundleSDF over more frames, or using stereo if Track 3 provides it.
   - See the repository skills `hoi-object-reconstruction-run` and `hoi-object-reconstruction-doctor`.
4. SAM3D candidate mesh with scale fitted to observations (if the BundleSDF quality gates fail).

**Selection:** per object class, choose the method with the lowest mean local CD-O on dev objects. Require a ≥ 15% improvement over the baseline to adopt it.

**Body-frame consistency (critical):**

- A new mesh must be expressed in the **same body frame** as the rollout poses. ICP-align the new mesh to the old mesh (a rigid transform plus a scale check).
- Apply the inverse transform to the mesh, not to the poses.
- **Verify:** after the swap, the same poses place the new mesh within 5 mm of the old surface (median).

**Expected outcome (estimate):** CD-O 0.57 → 0.25–0.40. Pose metrics also improve through `_REGISTER_OBJECT_MESH`.

### 5.3 Reference feasibility fixes (Days 3–7; CPU + ~5 GPU-h)

Address the issues found in §C.3 and §5.1:

1. **Gravity:** replace the PCA gravity heuristic with a support-plane fit (RANSAC on scene depth or object resting poses). Where an object rests on a table, its resting orientation should be stable.
   - **Gate:** improves dev experiment R relative to the oracle gravity tilt (measured on dev).
2. **Support and contact:**
   - Snap resting objects onto the support plane (remove floating and penetration).
   - Remove "floating motion": frames where the object moves without a hand within 2 cm. Interpolate those frames, or attach the object to the nearest hand.
3. **Two-object relative placement (RPE):** jointly refine both objects' trajectories so their relative pose is consistent with image evidence and contact. Our two-object RPE is ~28 cm, which suggests a relative scale or depth error between the objects.
   - **Test:** compare our relative distance to dev oracle values on the 2 two-object dev episodes.
4. **Temporal smoothing:** apply the same robust QP as T1-A1 to object trajectories. Policies track smooth references better.

### 5.4 Policy pilot (Days 5–7; ~15 GPU-h)

Run on 3 evaluation episodes (or dev O/A settings), 20M steps each.

**Arms:**

- **Baseline:** the current config with the curriculum scaled to 20M.
- **Reward retune:** increase the object pose tracking weight, add an orientation term weighted to the 0.5 rad threshold, and reduce the hand-imitation weight.
- **Warm-start:** initialise from Ep38's 60M policy (subject to §C.2 Q5).

**Scoring:** the local scorer on dev (exact) and against our own references on evaluation episodes (proxy).

**Decision:** adopt the arm with the best dev SP-SR, tie-broken by AUC.

### 5.5 Scale-up (Days 7–12; ~40 GPU-h)

- **Budget:** all 20 episodes to ~40M with the winning arm (~35 GPU-h at 6.4k steps/s), using the improved references and meshes.
- **Successive halving:** if Round 1 (20 episodes × 20M) shows clear leaders, give the remaining budget to the top 8.
- **Checkpoint selection:** use the proxy score against our reconstructed references. This is not exact for evaluation, so prefer the last stable checkpoint unless the proxy improves by > 10%.

### 5.6 Track 3 packaging sequence

| Package | Contents | Target day |
|---|---|---|
| t3-v2 | Improved meshes (§5.2) + existing policies re-rolled with the new mesh (if the body frame is verified) | Day 6–7 |
| t3-v3 | + reference fixes (§5.3) + 20M policies | Day 9–10 |
| t3-v4 (final) | + scale-up policies and best checkpoints | Day 12–14 |

---

## 6. Dated schedule (Day 0 = Tue 2026-10-20)

GPU = the single training queue (one job at a time). CPU = parallel Track 1 / analysis work.

| Day | Date | GPU queue | CPU / analysis | Uploads (planned) |
|---:|---|---|---|---|
| 0 | Tue 10-20 | Start T2 Tier 2 baseline | Phase 0 (§C); T1 harness (§3.1); T1-A0 | T1 v10 (if §C.2 Q1 OK) |
| 1 | Wed 10-21 | T2 Tier 2 (cont.) | T1-A1 + T1-C; T1-B diagnostic; T2 checkpoint selection (§4.4) | — |
| 2 | Thu 10-22 | T2 Tier 2 → package; start T2 Tier 3 | T1-B1; T3 dev setup (§5.1) | T2 Tier 2 v1; T2 Tier 1 `t1-v2` (if better) |
| 3 | Fri 10-23 | T2 Tier 3 (cont.) | T1 v11 assembly + gates | T1 v11 |
| 4 | Sat 10-24 | T2 Tier 3 → package; T3 dev decomposition (§5.1) | T1-B2 torch port; T3 mesh methods on dev | T2 Tier 3 v1 |
| 5 | Sun 10-25 | T3 dev decomposition (cont.); T2 Tier 1 pilot (§4.3) | T1-B2; T1-E; T3 §5.3 | — |
| 6 | Mon 10-26 (W2) | T2 Tier 1 pilot (cont.) | T3 mesh selection (§5.2); T1-B2 gates | T3 t3-v2 (if body frame verified) |
| 7 | Tue 10-27 | T3 policy pilot (§5.4) | T1 v12 assembly | — |
| 8 | Wed 10-28 | T2 Tier 1 halving Round 1 | T3 reference fixes | T1 v12 |
| 9 | Thu 10-29 | Round 1 (cont.) | Analysis | T2 Tier 1 `t1-v3` |
| 10 | Fri 10-30 | T3 scale-up (§5.5) | T3 v3 assembly | T3 t3-v3 |
| 11 | Sat 10-31 | T3 scale-up (cont.) | T2 Tier 2/3 improvement selection | T2 Tier 2 v2 / Tier 3 v2 (optional) |
| 12 | Sun 11-01 | T2 Tier 1 Rounds 2–3 | — | — |
| 13 | Mon 11-02 (unlimited) | T2 Rounds 2–3 (cont.) / T3 finish | Final packaging | T3 t3-v4; T1 v13 (optional) |
| 14 | Tue 11-03 | Buffer | Final packaging | T2 Tier 1 `t1-v4` / final |
| 15 | Wed 11-04 | **No new training** | Verify final selections by 12:00 EST | Last uploads by **15:00 EST** (2 h margin before the 17:00 freeze) |

**Slip rule:** if any GPU item slips by more than one day, cut the lowest-value pending item in this order:

1. T1-F
2. T2 Tier 2/3 improvement (§4.6)
3. T3 scale-up Round 2
4. T2 halving Round 3

Never cut packaging, validation or the buffer.

---

## 7. Packaging, provenance and submission protocol (all tracks)

1. **Versioned directories:** `.../<track>/<version>/` containing `execution.json` (parent SHAs, git commit, config, seeds, commands), `package_validation.json` and the final package SHA-256.
2. **Validation:**
   - Track 1: `eval_reconstruction.py` + the kit's validation + dry-run.
   - Track 2/3: the existing packer, including local scoring (Tier 1 / dev, exact) and the "distinct per-episode checkpoints" check.
3. **Commit before upload:** commit the package manifest (not large binaries unless already tracked) on `v2d-track1-blackwell` or a new branch. The upload message cites the commit SHA.
4. **Upload through the existing `submit/v2d_submit.py`:**
   - Use the exclusive receipt file and `--poll-receipt`.
   - **Never re-upload** a package that already has a receipt.
   - Before every upload, count the week's uploads per competition from the snapshot, and refuse if the count is ≥ 5 (except in the final 3 days).
5. **After scoring:** append the public scores to `CHALLENGE_STATUS_AND_LLM_HANDOFF.md` §3 and record the rank deltas.
6. **Final selection (11-04):**
   - Choose the best *coherent* package per track/tier by public score.
   - For Tier 1 and Track 1, prefer local exact or decision-safe evidence over public noise.
   - Verify the selections in each competition's UI or API.

---

## 8. Ordered execution checklist

- [ ] C.1 Snapshot all leaderboards; fetch Track 2 Tier 2/3 slugs
- [ ] C.2 Answer rule questions Q1–Q7 with quotes
- [ ] C.3 Build per-episode diagnostic tables for T1, T2 Tier 1 and T3
- [ ] 3.0 Submit v10 (if the weekly window allows)
- [ ] 3.1 Track 1 metric harness; tests T1-U1 … T1-U5 pass
- [ ] 4.2 Start T2 Tier 2 baseline training
- [ ] 3.2.1 Re-centre meshes; confirm metric neutrality
- [ ] 3.2.2 + 3.4 Object QP smoother with PEN constraints; per-episode λ selection
- [ ] 3.3.1 Human acceleration decomposition → choose B1 or B2
- [ ] 3.3.2 Root smoothing + coupling re-run
- [ ] 4.4 Tier 1 checkpoint selection → `t1-v2`
- [ ] 3.6 Assemble, validate and upload v11
- [ ] 4.2 Tier 2 upload; Tier 3 train + upload
- [ ] 5.1 Track 3 dev decomposition → name the bottleneck
- [ ] 4.3 Tier 1 pilot → choose the arm
- [ ] 5.2 Track 3 mesh method selection on dev CD-O; body-frame verification
- [ ] 3.3.3 T1-B2 refit (abort on day-1 parity failure)
- [ ] 3.2.4 Paint-roller and other low-IoU mesh tests
- [ ] 3.6 Assemble and upload v12
- [ ] 5.3 Track 3 reference fixes
- [ ] 5.4 Track 3 policy pilot
- [ ] 4.5 Tier 1 successive halving (Rounds 1–3)
- [ ] 5.5 Track 3 scale-up
- [ ] 4.6 Tier 2/3 improvements (optional)
- [ ] 7.6 Final selections verified by 11-04 12:00 EST

---

## 9. Risks, mitigations and open questions for the user

### 9.1 Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Smoothing improves ACC but worsens CD-O/CD-H | Medium | Medium | Evidence gates; per-episode λ; fall back to the parent episode |
| The reference's own acceleration is larger than inferred (the 0.074 / 0.145 interpretation is wrong) | Low–Medium | Medium | v11 public ACC-O is the test: abort further smoothing if it does not drop below 0.2 |
| MHR torch port does not match the official decode | Medium | Medium | 1-day abort; ship root-only B1 |
| PEN host subset or scale convention differs | Low | Low | Superset + dilation; re-read L874–882 if PEN is not ~0 |
| WSL/GPU stalls | High (it happened before) | High | Heartbeat, checkpoint resume, 80 GPU-h buffer |
| Track 3 bottleneck is policy and cannot be fixed in time | Medium | High | §5.1 decides early; the mesh/CD-O gains still bank on their own |
| Weekly quota breach | Low | High | Pre-upload count check (§7.4) |
| Rule interpretation (dev data use, warm-start, cross-tier) | Medium | High | §C.2 answers with quotes; defaults are conservative |
| Last-minute scoring lag | Medium | High | Final uploads by 15:00 EST on 11-04 |

### 9.2 Questions for the user (answer before Day 2 if possible)

1. Has the challenge organiser confirmed the weekly window start, and was there any action on the 7 uploads in week Oct 12?
2. Is warm-starting a policy from another episode's policy acceptable to you (and to the rules)?
3. Should Track 2 Tier 2/3 work come before Tier 1 improvement? This plan assumes **yes**, because those boards have no entry.
4. Is a second GPU or a cloud machine available? With 2 GPUs, run T3 scale-up and T2 Tier 1 halving in parallel. The budget table would roughly double.
5. Should this plan be committed to `v2d-track1-blackwell`, or kept local?
