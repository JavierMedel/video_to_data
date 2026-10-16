# V2D Next Improvement Plan (Track 1): v9 temporal round, then v10 targeted geometry

**Plan date:** 2026-10-16 (local, EDT). **Input:** `CHALLENGE_STATUS_AND_LLM_HANDOFF.md` at `origin/v2d-track1-blackwell` commit `80b170d`. This plan cites that file as **[H §n]**.

**Extra evidence used:** the official scorer source on the same commit, `reconstruction/v2d_submission_kit/v2d_submission_kit/metric_code/track_1/{CD-O,ACC-O,...}.py`. I read it directly. It is cited as **[S file:line]**. I also inspected argument parsing and hardcoded paths in the existing `reconstruction/scripts/track1_*.py` files.

**Not done:** I ran no experiments, reconstructions, scorer runs, or uploads. Every predicted score change below is labeled a **guess** or a **bound**. Bounds are mathematical; guesses are not promises. Execute everything on the execution machine (`/home/alien51/video_to_data`, one RTX 5090 Laptop GPU with 24,463 MiB) after reconciling with the actual checkout [H §1, §8].

---

## Execution corrections (authoritative, 2026-10-16)

The user authorized correcting and implementing this plan. Local reversible implementation and the pilot proceed; expansion follows measured gates. No new upload is authorized merely by this document. The following corrections supersede conflicting original targets or policies below:

- Start from immutable v8 with all 60 pair hashes and its actual Parquet SHA256 bound. Historical v6 source predictions and evaluator caches remain untouched.
- A pinned first-scored human pose with unchanged shape/scales preserves alignment; it does not reveal the hidden positive alignment scale. Report distance bounds **per unit alignment scale**, never as guaranteed numerical official bounds.
- The 0.05 cm CD-O / 0.10 cm CD-H values are desired official outcomes, not locally certified guarantees. Local 2 cm displacement bounds cannot prove them.
- Use the conservative human bound `2 * s_A * max_v ||delta_v||` per frame, over all decoder vertices. The full-vertex mean is not a bound on the unavailable host-selected subset mean.
- Match the scorer's gaps **and repeated endpoint acceleration values** within each stretch; average within each episode, then equally across episodes. Do not compare prior pooled v8 triple magnitude directly with the official error.
- Duplicate scores from competitors do not establish their methods or reference acceleration. The 0.074/0.145 reference-acceleration assumptions and projected score ranges are unverified hypotheses and are not acceptance thresholds or error bounds.
- Preserve v8's per-frame full-hand and FPS512 mean non-increase (numerical tolerance only), and at most 0.5 cm gap increase on parent-contact-like frames. Do not apply the originally proposed +0.002 cm / +10% relaxations. Hoop gate redesign remains an unapproved, separately reported proposal, never an automatic promotion rule.
- Both the hand checker **and the reconstruction video renderer** must explicitly decode each version's own human. Test that a changed pose changes decoded geometry and both panels use it.
- Identify parameter names from the actual TorchScript model. Root translation is in cm; decoded vertices are mm and joints m. Parameters 130:136 are internal lengths, not angles. Verify periodicity numerically before unwrapping angular parameters, preserve the original branch at the pinned row, and evaluate the decoded result.
- Use an O(T*D) banded SPD solve with an independent dense-solve oracle, not repeated dense cubic solves. Guard robust zero-residual/MAD degeneracy.
- The listed half-power periods were half-amplitude approximations. Actual infinite-length uniform-weight half-power periods for lambda 10/100/1000 are approximately 13.81/24.70/44.01 frames. Finite pinned/robust filters do not have that exact stationary response.
- Keep pilot candidates and diagnostics. If the frozen-lambda pilot fails the expansion criteria, record failure and stop expansion rather than loosening gates, launching all30, or uploading. Proceed to diagnostic geometry audit if supported independently.

## 0. Executive summary

| Item | Decision |
|---|---|
| Parent package | **Immutable v8**: `contact_smoothing_v8/submission_episodes/` plus `track1_contact_smoothing_v8.parquet` (SHA256 `bd73a931…3198`) [H §3.3, §9] |
| Round 1, **v9** | **Temporal denoising**. Change only `mhr_pose` (array 0) and `object_translation` (array 4). Use a robust, pinned Whittaker smoother. Targets: **ACC-H** (now 23rd of 26) and **ACC-O** (now 21st of 26). |
| Round 2, **v10** | **Targeted CD-O geometry** on top of v9: a per-episode risk audit, then hoop episodes 0 and 2, cross-episode mesh/scale consistency, and floater cleanup. Target: **CD-O** below CARI4D's 23.16246. |
| Round 3, **v11** (optional) | Symmetry-aware rotation-flip repair for asymmetric objects, only if v10 audit evidence supports it |
| Deferred | Full 300-step reruns, blind depth shifts, human hybrids from v2, first-frame alignment changes, Track 2/3 work |
| Main new insights (verified in scorer source) | (1) Every metric is an **unweighted mean of 30 per-episode means**. (2) One similarity alignment comes **only from the first scored frame's human**. Pinning that frame makes human changes leave object metrics exactly unchanged. (3) ACC-O uses **only `object_translation`**; rotations and meshes do not enter it. (4) ACC is computed only within contiguous scored stretches. |

**Why this order:** Leaderboard evidence suggests our ACC-H and ACC-O are dominated by our own jitter (§B.3). Removing jitter is CPU-cheap and attributable. Its effect on CD can be bounded. That makes it a large, low-risk rank opportunity. CD-O needs object-specific fixes: v6 shows one episode-level geometry fix outweighed all generic smoothing [H §3.1, §7].

---

## A. Execution context and scope

- **Planning machine:** reasoning and Markdown only, already done. **Execution machine:** `/home/alien51/video_to_data`, branch `v2d-track1-blackwell`, WSL2 Ubuntu, RTX 5090 Laptop GPU (24,463 MiB, sm_120), about 31 GiB RAM, host venv `reconstruction/.venv/bin/python` (no torch/scipy/cv2), GPU image `v2d_cari4d:blackwell` [H §8].
- **Scope:** Track 1 only. Track 2/3 stay out of scope unless the user expands it [H §2].
- **Deadline:** freeze **2026-11-04 17:00 EST**. The website says five submissions per week, with unlimited submissions in the last three days [H §2]. **Recheck the official rules and actual remaining quota before any upload.** Also answer the scoring-policy question in §G.6.
- **Authorization:** this plan does **not** authorize uploads. Implementation, local experiments, and packaging are reversible and isolated, so proceed without asking. An actual upload needs the user's explicit go-ahead (§13 of the handoff, and §H.0 below).

### A.1 Reference values (Kaggle snapshot 2026-10-16T21:04Z) [H §4.1]

| Metric | v8 (parent) | Our displayed best | CARI4D | Leader | Notes |
|---|---:|---:|---:|---:|---|
| CD-H | 15.43955 | 14.99887 (v2) | 15.57527 | 13.43306 | v8 would be about 13th; the displayed v2 value is 8th |
| CD-O | 23.48324 | 23.48324 (v8) | 23.16246 | 17.91571 | 8th; only metric where v8 loses to CARI4D |
| ACC-H | 1.15894 | 1.15894 | 1.97387 | 0.13738 | 23rd; **8.4× the leader** |
| ACC-O | 0.71908 | 0.71908 | 0.72622 | 0.07111 | 21st; **10.1× the leader** |
| PEN | 0.00322 | 0.00213 (v1) | 0.00423 | 0.00000 | v8 would be about 16th; the displayed v1 value is 14th |

### A.2 Measurable objectives

**Primary objective, v9 (temporal round).** Each line includes a guess of the achievable range, not a promise.

- **ACC-H:** reduce from 1.15894 to **0.30 or better**, with a stretch target of **0.16 or better**. Guess: 0.17–0.40 is achievable if jitter dominates (§B.3). Using the 2026-10-16 table, 0.15 would rank about 10th and 0.20 about 16th.
- **ACC-O:** reduce from 0.71908 to **0.20 or better**, with a stretch target of **0.10 or better**. Guess: 0.10–0.25. In the snapshot, 0.10 would rank about 11th and 0.15 about 14th.
- **Allowed tradeoffs (pre-registered):**
  - Desired official CD-O regression limit: **0.05 cm**. Local bounds do not certify this limit because their thresholds are larger and alignment scale is unknown.
  - Desired official CD-H regression limit: **0.10 cm**; not guaranteed by local displacement screens.
  - PEN must satisfy the local PEN policy in §E.4.
  - Smoothness must never replace accuracy. Every change must stay within the displacement caps.

**Primary objective, v10 (geometry round).**

- **CD-O:** beat CARI4D (**23.16246**, a 1.366% reduction from v8).
- Because the score is an unweighted episode mean [S ACC-O.py:1010], the **sum** of per-episode CD-O reductions must exceed **0.32078 × 30 = 9.62 cm**. For example, one episode improving by about 10 cm would be enough.
- Milestones, as total summed per-episode reduction:

  | Milestone | Score (cm) | Required summed reduction |
  |---|---:|---:|
  | SNU CVLAB | 23.13746 | 10.37 cm |
  | sv1105 | 23.05002 | 13.00 cm |
  | Lodestar | 19.52520 | 118.7 cm |
  | Leader | 17.91571 | 167.0 cm |

- **Catching the leader needs several near-catastrophic episodes fixed**, not uniform refinement. Matching CARI4D and catching the leader are separate goals [H §1.1].

**Non-goals:** do not chase PEN to 0 by separating objects from hands [H §5.2]. Do not probe hidden scores with offset sweeps [H §12.7].

---

## B. Evidence and hypotheses

### B.1 Verified facts from the official scorer (new relative to the handoff)

All five metric scripts share the same core. Line numbers refer to `ACC-O.py`; `CD-O.py` uses identical lines.

1. **Unweighted episode averaging.** `score()` computes one value per episode, then `result = float(np.mean(values))` [S ACC-O.py:1010].
   - Episode 11 (803 scored frames) and Episode 14 (83 scored frames) carry **equal weight** in every metric.
   - **Consequence:** prioritize by per-episode error magnitude, not frame count. Short, broken episodes are cheap, high-value targets.
2. **Back-calculated per-episode deltas:**
   - The v6 Episode 1 change lowered aggregate CD-O by 4.66134, so **Episode 1's own CD-O fell by about 139.8 cm** (4.66134 × 30).
   - The v7 +7 cm depth shift on Episode 7 **raised that episode's CD-O by about 3.88 cm** (0.12941 × 30).
   - Across v8's 12 promoted episodes, CD-O fell by only **0.05 cm in total** (0.00165 × 30).
   - ACC-O fell by **2.73 cm in total** (0.09105 × 30).
   - These are exact consequences of averaging plus single-episode or known-subset changes.
3. **One global similarity from the first scored frame's human.** `_episode_metrics` fits `fit_similarity_transform(pred["mhr_vertices"][0], target["mhr_vertices"][0])` and applies it to all predicted arrays [S ACC-O.py:679–697].
   - `mhr_vertices` are the alignment-role vertices of the **first scored frame** in the episode's sorted scored-frame list [S ACC-O.py:634–670].
   - **Consequence:** suppose a change keeps frame-`f0` `mhr_pose` bit-identical, along with `mhr_scales` and `mhr_shape`. Then the transform is unchanged, so **CD-O and ACC-O are exactly unaffected by human changes**. Human edits then affect only CD-H, ACC-H, and PEN.
4. **ACC-O uses only `object_translation`.** The scorer aligns `pred["object_translation"]` and differences it [S ACC-O.py:686–694].
   - Rotation, mesh and scale do not enter ACC-O.
   - Translation is the **mesh-origin** position, not the centroid. v8's DP minimized **centroid** acceleration [H §6.1]. The two coincide only if rotation is fixed or the mesh origin is at the centroid.
5. **ACC uses second differences within contiguous scored stretches.** `_stretches` splits at index gaps, and stretches shorter than 3 frames raise an error [S ACC-O.py:672–695].
   - Our NPZ arrays are full-video, indexed by original frame [H §6.3]. Scored stretches are contiguous runs of video frames.
   - **Consequence:** smoothing on the full video, then extracting scored frames, is exactly evaluated.
6. **CD-O surface samples** come from `_sample_mesh_surface(..., seed=episode_index)`, area-weighted, with the count equal to the number of human "body" role vertices [S ACC-O.py:441–455, 652].
   - Spurious mesh area, such as floaters or interior walls, receives samples in proportion to its area, which inflates the prediction→reference term.
7. **PEN alignment:** `_episode_penetration` uses only the alignment **scale** from the same first-frame fit [S ACC-O.py:874–882]. It is per-episode averaged like the other metrics.
8. **Hardcoded v6 controls:** `track1_contact_campaign.py:53–54`, `track1_contact_campaign_package.py:40,101`, `track1_contact_campaign_review.py:32` and `track1_episode7_contact_smoothing.py:39` all hardcode `cdo_v6_experiments` and/or `track1_cdo_v6.parquet`. **Do not use them for v9 packaging** (see §D.4).
   - Explicit `--control-dir`/`--candidate-dir`/`--report-dir` options exist in `track1_cdo_full_image_review.py`, `track1_cdo_hand_penetration_review.py` and `track1_cdo_render_candidate.py`, so those tools can be reused.

### B.2 Facts from the handoff that constrain the plan

- v8's numerical history and IDs [H §3]. v8 changed 2,056 translation rows relative to v6, and all other arrays are identical [H §10.5].
- **Negative evidence:**
  - 300-step or rotation-enabled refinement did not fix Episode 12's flips.
  - Blind ray offsets hurt silhouette and contact.
  - A body-proxy PEN gate was wrong.
  - The v7 inferred-depth gain coincided with worse official CD-O.
  - Unconstrained SG7 increased hand penetration on Episode 7 [H §7].
- **Positive evidence:**
  - The object-specific hoop fix gave a large CD-O gain.
  - Contact-constrained smoothing improved ACC-O by 11.24% at almost no CD-O cost [H §3.1, §7].
- v8's local scored-triple translation acceleration **magnitude** was 0.704264 cm [H §6.2]. The official ACC-O **error** was 0.71908.
- Our ACC-H stayed at **1.15894 from v3 to v8**, because human arrays were never temporally processed after v3 [H §3].
- Episode 7 still has 48 rotation jumps above 45° [H §6.4]. The hoop episodes are 0, 1 and 2. Only Episode 1 received the metric-hoop fix (v6).
  - v7 rejected Episode 0 on **hand proximity**: no feasible hoop path under the unchanged clearance and proximity limits.
  - v7 rejected Episode 2 on disjoint-frame image/depth or meaningful-gain checks (`cdo_v7/episode_decisions.md`).
  - In v6, the Episode 1 control had **no contact-like frames**, so the proximity guard was vacuous there [H §3.1].
- Compute: one laptop GPU, at most two concurrent GPU subprocesses proven [H §8.1].

### B.3 Inferences (strongly supported, still not proven)

- **I1. ACC metrics may be jitter-dominated (hypothesis only).** Several apparently trivial submissions share identical scores:
  - Tony and Furqan Zaheer both have ACC-H **0.14528**, PEN **0.00000**, CD-H 70.6/73.6 and CD-O 145/170.
  - Marcell, Tony and Furqan all have ACC-O **0.07429**.
  - These look like near-static or heavily smoothed predictions.

  For such predictions, ACC ≈ mean |a_ref|. This suggests mean reference acceleration of about **0.074 cm/frame²** for objects and **0.145 cm/frame²** for selected human joints, in the aligned frame. By the triangle inequality, for any prediction:

  > `|mean|a_ref| − mean|a_pred|| ≤ ACC ≤ mean|a_pred| + mean|a_ref|` (per frame and joint, then averaged; the aligned scale is applied)

  Our local v8 object magnitude, 0.704, nearly equals official ACC-O, 0.719. Jitter dominates. **If we drive aligned mean |a_pred| to about 0.05 cm/frame², ACC-O ≲ 0.05 + 0.074 ≈ 0.12** by the bound, assuming I1 and an alignment scale near 1. The same reasoning gives ACC-H ≲ mean|a_pred,joints| + 0.145.
- **I2. Translation denoising is close to CD-neutral.** v8 moved translations up to 3 cm on 12 episodes and changed CD-O by −0.00165. Mild denoising should not materially hurt CD-O. The hard bound in §C.2 makes this auditable.
- **I3. Episodes 0 and 2 may hide large CD-O errors.** They are the same hoop object as Episode 1, whose pre-v6 CD-O was ≥ 139.8 cm. Old v6 overlays noted the original ring was "much too small and displaced" [H §3.1, v6 final review]. If Episode 0 or 2 shares that failure, a fix could exceed the whole 9.62 cm CARI4D gap alone. **Unverified: must be measured first (E4.0).**
- **I4. Same physical object in three episodes.** Each object appears in three episodes [H §6.3]. Mesh and scale disagreement within a triple flags a local error without hidden truth.

### B.4 Unknowns, and files the executor must inspect before coding

| Unknown | Why it matters | Where to look |
|---|---|---|
| Which of the 136 `mhr_pose` dimensions are angles (Euler, radians) and which are translations or other values | Unwrapping and units for smoothing | `track1_convert_refined.py`; the local MHR decoder used by reviews (`reconstruction/data/weights/track1_submission_mhr/mhr_model.pt`); parameter transform in `lib_mhr` |
| Exact `MHR_TABLE3_BODY_JOINT_INDICES` | Local ACC-H proxy must use the same joints | `metric_code/track_1/ACC-H.py` (grep the constant) |
| Whether NPZ `pose` arrays cover every video frame, and the `valid_input[T]` semantics | Smoother weights | Open one v8 NPZ with `allow_pickle=False` |
| Whether `track1_cdo_hand_penetration_review.py` decodes **candidate** humans or reuses control humans | v9 changes humans; the PEN check must use candidate humans | Read its loading code before use |
| Mesh origin vs centroid offset per episode (`‖s·c_mesh‖`) | Effect of rotation on translation; relevant to E2b | Compute from v8 GLB and NPZ |
| Hoop ring radius in Episodes 0 and 2 vs Episode 1's accepted metric hoop | I3 | v8 GLBs, `cdo_v6/hoop_geometry`, `cdo_v7/episode_decisions.md` |
| Current official rules on final ranking (one package vs best per metric) and remaining weekly quota | Tradeoff policy (§G.6) | Official site and Kaggle competition rules pages |

---

## C. Ranked experiments and pilot

### C.1 Ranked experiment table

The "Guess" column is an uncertain expectation, not a promise. Cost estimates require pilot measurement before extrapolation.

| Rank | ID | Hypothesis | Targeted failure | Episodes | Arrays / code touched | Parent | Mechanism | Evidence to seek | Cost (uncertain) | Guess | Abort criteria |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **E1 Human pinned smoothing** | ACC-H is jitter-dominated (I1) | Human joint jitter, never smoothed since v3 | All 30 | Array 0 only; new `track1_whittaker.py`, `track1_v9_temporal.py` | v8 | Robust Whittaker smoothing in parameter space; first scored frame pinned bit-exactly | Scored-triple joint acceleration magnitude falls ≥70%; displacement caps hold; local PEN policy holds; videos show no foot or hand "swimming" | CPU: seconds per episode. Decode and PEN checks in Docker: minutes to a few hours total | ACC-H 1.16 → 0.17–0.40; CD-H ±0.1; CD-O and ACC-O exactly unchanged | Reduction <40% at the displacement cap; Euler artifacts; PEN fails after reweighting on >6 episodes |
| 2 | **E2 Object translation smoothing** | ACC-O is jitter-dominated (I1, H §6.2) | Translation jitter; v8 was too conservative (SG7, 12 episodes, 3 cm cap) | All 30 | Array 4 only; same new modules | v8 (+E1 humans for PEN checks) | Robust Whittaker on `t` directly (what ACC-O scores), with contact-anchor reweighting | Aligned scored-triple `|a_t|` ≤ 0.10 cm on average; per-episode CD bound ≤ 2 cm; IoU change ≥ −0.002 | CPU plus a GPU PEN check | ACC-O 0.72 → 0.10–0.25; CD-O change roughly −0.05 to +0.05 | IoU drop > 0.005 on any episode; bound > 2 cm after capping |
| 3 | **E3 Per-episode CD-O risk audit** (diagnostic only) | Unweighted episode averaging makes a few bad episodes dominant | Unknown outliers | All 30 | None; new read-only `track1_v10_risk_audit.py` | v9 or v8 | Rank episodes by IoU, triple consistency, size priors, floaters and rotation jumps | Ranked list, with figures | CPU plus existing renders: hours | No score change by itself | — |
| 4 | **E4 Hoop Episodes 0 and 2** | Same hoop as Episode 1; may share the "too small/displaced" failure (I3) | Wrong ring geometry, scale or depth | 0, 2 | Arrays 3–7 for those episodes; existing hoop scripts parameterized by parent | v9 | Reuse Episode 1's accepted metric hoop and image-fitted orientation; bounded smooth depth; **redesigned evidence-based proximity gate** | E4.0: ring diameter × scale vs Episode 1; IoU gain ≥ 0.02; mask-edge agreement; hand checks | GPU: hours | If broken like Episode 1: up to several cm aggregate CD-O; otherwise about 0 | E4.0 shows diameters agree within 10% and IoU ≥ 0.60, so skip |
| 5 | **E5 Triple consistency and floater cleanup** | Same physical object should share mesh and scale; floaters inflate CD-O | Inconsistent scale or mesh; spurious area | Flagged by E3 | Arrays 5–7 (and 4 if re-depthed) for flagged episodes only | v9 | Replace an outlier mesh with the best-supported triple mesh; fit only scale and ray depth to masks; remove floaters | IoU gain; size consistency; hand gate; component stats | GPU renders: hours | 0–3 cm aggregate CD-O, highly uncertain | Fails the disjoint-frame IoU screen; hand gate fails |
| 6 | **E6 Asymmetric rotation flips** (v11, optional) | Flips of handles or asymmetric parts inflate CD-O | Tracker flips | Iron 6–8, pan 12–14, stool 24–26, cart 27–29, desk 9–11 (as flagged) | Array 3 | v10 | Symmetry-aware DP over flip hypotheses scored by per-frame IoU and edges (extend `track1_cdo_temporal_candidates.py`) | Fewer >45° jumps **with** IoU gain; hand gate | GPU: many hours | Uncertain; prior attempts had contact rejections [H §3.1 v5] | No IoU gain; contact failures |
| — | Deferred | — | — | — | — | — | 300-step reruns, blind depth, v2 human hybrid, first-frame alignment change, Track 2/3 | — | — | — | Prior negative evidence [H §7]; the alignment-coupling risk is untestable locally |

**E1 + E2 form v9 together. This is attributable.** By B.1(3) and B.1(4):

- With `f0` pinned, CD-H and ACC-H depend only on E1.
- CD-O and ACC-O depend only on E2. (CD-O also depends on rotation and mesh, which are unchanged.)
- Only PEN mixes the two, and §E.4 checks it on the combined candidate plus each single-factor candidate.
- Official package scores therefore give per-factor attribution for four of five metrics without spending two upload rounds. Both single-factor packages are still built locally for diagnostics.

### C.2 Mathematical guards that need no hidden data

1. **CD worst-case bound under translation-only change.** Every object surface point moves by the same vector Δt_f, and the aligned scale is s_A. So each directed nearest-neighbour mean changes by at most s_A·|Δt_f|:

   > `|ΔCD-O_f| ≤ 2·s_A·|Δt_f|`

   **Report** `B_O,e = mean_f∈scored 2|Δt_f|` per episode and `B_O = mean_e B_O,e`, assuming s_A ≈ 1. The prediction→reference term changes by at most the mean displacement, and the reference→prediction term by at most the max displacement. For pure translation, both equal |Δt_f|.
2. **CD-H bound under human change.**

   > `|ΔCD-H_f| ≤ 2·s_A·max_v |Δv_f|`

   over the decoded body vertices. Use `2 * max_v |delta_v|` instead: the all-vertex maximum bounds either host subset maximum, whereas an all-vertex mean need not bound its mean. Keep the unknown alignment multiplier explicit.
3. **ACC triangle bound** (I1): predicted aligned acceleration magnitude bounds the error from above, together with the inferred reference magnitude.
4. **Alignment invariance check:** for every episode, `mhr_pose[f0]`, `mhr_scales` and `mhr_shape` must be **bit-identical** to v8. That makes the s_A and alignment transform identical to v8 [S ACC-O.py:686].

### C.3 Pilot design (v9)

- **Pilot episodes, one per major object type:**

  | Episode | Object | Why chosen |
  |---:|---|---|
  | 1 | hoop | v6-fixed; few frames; thin, open mesh |
  | 4 | big bowl | 519 scored frames |
  | 7 | iron | Ablation history; 48 rotation jumps |
  | 11 | desk | Longest episode (803 scored frames) |
  | 13 | pan | Short (106 scored frames); low prior gain |
  | 24 | stool | Large prior gains |
  | 28 | cart | Short (147 scored frames); low prior gain |

  Exclude Episode 16 from the **pilot** only, because of its historical protection [H §9.2]. Derived smoothing (no expensive rerun) is in line with v8's authorized precedent, so include it in the all-30 run.
- **Controlled factors:**
  - λ grid for human and object: `{3, 10, 30, 100, 300, 1000}`.
  - Robust IRLS: on/off.
  - Pin on/off (human only; off is **diagnostic only, never packaged**).
  - Contact-anchor reweighting: on/off.
- **No mixing:** change one factor per comparison. Keep **no depth shifts, scale changes, or rotation edits** in v9.
- **Determinism:** the smoother is deterministic linear algebra. Any FPS512 subset uses the existing fixed seed; record it.
- **Frame splits:** no fitting to hidden data. For IoU, split each episode's scored frames into even and odd indices. Choose λ using even-frame proxies only, then report odd-frame proxies as a check on a **local** split.
- **Expansion rule to all 30:** all of the following must hold.
  1. On ≥ 6 of 7 pilot episodes, the chosen λ_H gives ≥ 70% reduction in scored-triple joint acceleration magnitude. Mean vertex displacement must be ≤ 1.0 cm, with p99 of the per-frame maximum vertex displacement ≤ 4.0 cm.
  2. The chosen λ_O gives aligned scored-triple translation acceleration ≤ 0.10 cm on average, with `B_O,e` ≤ 2.0 cm and odd-frame IoU change ≥ −0.002 for every pilot episode.
  3. The PEN policy (§E.4) passes on ≥ 6 of 7 after reweighting.
  4. Visual review of full videos for Episodes 7 and 11 finds no new swimming, foot skating or object lag.
  5. The pinned-frame bit-identity test passes.

  If the human criteria fail, ship **E2 only** in v9. If the object criteria fail, ship **E1 only**. If both fail, stop and report.
- **One global λ per stream** (λ_H, λ_O), chosen on the pilot and **frozen** before the all-30 run. Per-episode fallback is limited to anchor reweighting or parent retention (§E.5). Do not tune per episode, so the run cannot overfit local proxies.

---

## D. Implementation changes

### D.1 New pure-NumPy smoother: `reconstruction/scripts/track1_whittaker.py`

Runs in the host venv: NumPy only, no SciPy. Use the bandwidth-two Cholesky solve; verify against an independent dense oracle in tests.

```text
whittaker_smooth(y[T,D], w[T], lam: float, pin: Optional[int] = None,
                 robust_iters: int = 3, huber_k: float = 2.5) -> (x[T,D], info)
  objective: sum_f w_f ||x_f - y_f||^2 + lam * sum_f ||x_{f-1} - 2 x_f + x_{f+1}||^2
  solve:     (W + lam * D2^T D2) x = W y        # D2: (T-2)xT second-difference
  pin:       eliminate x[pin] = y[pin] exactly (substitute and solve the reduced system;
             do NOT use a 1e12 weight, because bit-identity is required)
  robust:    r_f = ||x_f - y_f|| (per-frame vector norm across D, after per-dim scale normalization);
             sigma = 1.4826 * MAD(r); w_f <- w0_f * min(1, huber_k*sigma / r_f); repeat
  info:      final weights, residual stats, effective cutoff, iterations
```

- **Properties to test** (§E.1):
  - Linear trajectories are reproduced exactly (null space of D2).
  - The pinned row is bit-identical.
  - With uniform weights, the frequency response is `H(ω) = 1 / (1 + λ(2 − 2cos ω)²)`.
  - Uniform infinite-length half-power periods: λ=10 → about13.81 frames, λ=100 → about24.70 frames, λ=1000 → about44.01 frames. Pinning, robust weights and finite boundaries change this response.
  - Frame-to-frame jitter at ω≈π is attenuated by `1/(1+16λ)`.
- **Weights** `w0_f`: 1.0 when `valid_input[f]` is true (or the field is absent), and **1e-3** when false, so invalid frames are interpolated.
- **Gaps:** NPZ arrays are full-video, so smooth the whole video. If any NPZ were scored-only (check B.4), split at frame-index gaps and smooth each run independently. Never difference across gaps.

### D.2 Driver: `reconstruction/scripts/track1_v9_temporal.py` (new; all flags below are **new work**)

```text
--root PATH                      default: repo reconstruction root
--parent-episodes DIR            REQUIRED (v8: .../contact_smoothing_v8/submission_episodes)
--parent-parquet FILE            REQUIRED (v8 parquet); SHA256 recorded and verified against H §3.3
--scored-frames FILE             default incident_reports/track1_scored_frames.json
--out NAME                       REQUIRED isolated experiment name, e.g. v9_temporal
--episodes N [N ...]             default all 30
--stream {human,object}          one stream per invocation (keeps factors separable)
--lam FLOAT                      smoothing strength
--robust/--no-robust             default robust
--pin-first-scored/--no-pin      human default pin; --no-pin writes diagnostics only, refuses to stage
--human-dir DIR                  object stream: which humans to use for PEN checks (E1 output or parent)
--anchor-report FILE             optional per-frame anchor list from the PEN check (§E.4)
--status                         read-only progress
--resume                         skip episodes with complete, hash-matched outputs
```

**Per-episode procedure:**

1. Load the parent NPZ with `allow_pickle=False`.
2. Record SHA256 for the NPZ and GLB.
3. Smooth the selected stream:
   - **Human:** smooth `pose[T,136]` in parameter space. First, run `np.unwrap` on every dimension B.4 identifies as an angle (radians). Also flag any dimension whose consecutive difference exceeds π.
   - **Object:** smooth `translation[T,3]` (metres) directly. **Do not convert it to a centroid** (B.1(4)).
4. Write a new NPZ that copies every other key byte-for-byte. Update only the target array.
5. Add a `report` JSON entry `{"v9_temporal": {...params, parent hashes...}}`, preserving existing keys.
6. Copy the GLB unchanged and verify its hash.

**Output layout:** everything below is new and isolated; nothing is overwritten.

```text
reconstruction/data/v2d_challenge/track_1/v9_temporal/
  parent_manifest.json                   # parent dir/parquet paths + SHA256 of all 60 files + parquet
  human/lam_<λ>/episode_000NNN/{episode_000NNN.npz, episode_000NNN_object.glb, smooth_info.json}
  object/lam_<λ>/episode_000NNN/{...}
  combined/candidate_episodes/{episode_000NNN.npz, episode_000NNN_object.glb}  # final staged pairs
  diagnostics/{motion_*.json, pen_*.json, iou_*.json, bounds_*.json}
reconstruction/incident_reports/track1_improvement/v9_temporal/
  execution_log.md, pilot_results.md, selection_review.json, package_validation.json,
  submission_results.md (only if submitted)
```

### D.3 Motion and bound diagnostics: `reconstruction/scripts/track1_v9_motion_report.py` (new; runs in Docker)

Runs in Docker because it needs torch to decode the local MHR model.

- **Human:** decode `pose`, `scales` and `shape` with the same local decoder used by existing hand reviews. Then:
  - Select `MHR_TABLE3_BODY_JOINT_INDICES`.
  - Compute the scored-triple joint acceleration magnitude (cm) using `_stretches` semantics.
  - Compute per-frame vertex displacement vs parent (mean, max, p99) and the CD-H bound from C.2(2).
- **Object:** compute the scored-triple `|a_t|` (cm) and the CD-O bound `B_O,e` from C.2(1).
- **Reuse:** the scored-gap-safe logic already in `track1_contact_smoothing.scored_motion_diagnostics` (it exists; extend, do not duplicate). Import the stretch definition to match the official `_stretches`.
- **Output:** one JSON per run, with input hashes and source fingerprints.

### D.4 Generic packer: `reconstruction/scripts/track1_package_from_parent.py` (new)

The existing campaign packer hardcodes v6 (B.1(8)).

```text
--parent-episodes DIR  --parent-parquet FILE  --candidate-episodes DIR
--allowed-arrays 0,4   # numeric array indices allowed to differ (H §10.5)
--expect-episodes 0-29 --commit FULL_40_CHAR_GITHUB_URL (omit = stage+validate only)
--out FILE.parquet
```

1. Validate all 30 pairs: finite arrays; positive scale; orthonormal rotations (`‖RᵀR−I‖ < 1e-6`, `det > 0`); mesh integrity and budget; scored-frame coverage.
2. Call the official `eval_reconstruction.py --episodes DIR --sample data/track_1_sample_submission.parquet --commit URL --out FILE` [H §10.4].
3. Align the result and the parent Parquet by `row_id`, then compare every value column.
   - **Fail** if any row outside `--allowed-arrays` differs.
   - **Fail** if any `mhr_pose` row at frame `f0` differs (alignment invariance).
4. Record changed-row counts per array and episode, SHA256 and row count (must be 740,780).
5. **Never upload.**

### D.5 Reuse of existing review tools (explicit directories only)

- `track1_cdo_full_image_review.py --episodes … --control-dir <v8> --candidate-dir <v9> --report-dir <diag>`: all-scored-frame IoU and common-support depth.
- `track1_cdo_hand_penetration_review.py --episodes … --sample-count 0 --control-dir … --candidate-dir … --report-dir …`: exact-mesh full-hand and FPS512 checks.
  - Both this checker and the video renderer previously assumed identical humans. Add explicit `--allow-human-change` support, independently decode the actual control and candidate, and test the decoding adapter. Enable strict framewise contact gating for this experiment.
- `track1_cdo_render_candidate.py --control-dir … --candidate-dir … --output-dir … --candidate-label "v9 temporal"`: full videos and mask sheets.
- These tools bind source fingerprints to caches [H §6.4]. If you edit them, invalidate old caches; never relabel them.

### D.6 v10 tooling (after v9)

- **`track1_v10_risk_audit.py`** (read-only). For each episode, using v9 pairs, compute:
  - All-scored-frame IoU (reuse D.5 output).
  - Scaled bounding-box extents and principal dimensions.
  - Surface area and connected components (area of the largest component as a fraction of the total).
  - Watertightness, and the origin–centroid offset `‖s·c‖`.
  - Triple-consistency z-scores: per-object median of scaled extents, with each episode's ratio to that median.
  - Rotation jumps above 45° on scored frames (reuse `track1_cdo_pose_audit.py` logic, parameterized by directory).
  - A human-relative size check: object height vs MHR human height in 3 frames where both are fully visible.

  Output a ranked CSV and Markdown table, plus montage images.
- **Hoop and geometry scripts** (`track1_cdo_hoop_*.py`, `track1_cdo_temporal_candidates.py`): add `--parent-dir` and `--out` arguments, required new work. They currently select iterations 4–7.

---

## E. Validation and promotion

### E.1 Synthetic and numerical tests (new `reconstruction/scripts/test_track1_whittaker.py`; host venv)

1. A linear trajectory and λ ∈ {1, 1e3} reproduce the input to `1e-9`.
2. The pinned row is **bit-identical** (`np.array_equal`), with and without robust iterations.
3. White-noise-on-sine input: acceleration magnitude falls ≥ 90% at λ=100; the sine amplitude at a 60-frame period is retained ≥ 95%.
4. A spike outlier (a single +10 cm frame) is suppressed with robust weighting: the neighbour displacement is < 0.5 cm, vs > 1 cm without robust weighting.
5. Gap semantics: the motion report never forms a triple across a missing frame index (mirror `_stretches`). This reuses the existing test pattern in `test_track1_contact_smoothing.py`.
6. Angle unwrap: an input wrapping from +π to −π gives no acceleration spike after unwrap and smoothing.
7. The packer rejects a candidate where one `object_scale` row changed under `--allowed-arrays 0,4`. It also rejects a changed `mhr_pose` row at `f0`.

Also run `test_track1_contact_smoothing.py` (existing, eight cases) [H §10.3].

### E.2 Real-data validation matrix (pilot, then all 30)

| Check | Scope | Tool | Pass condition |
|---|---|---|---|
| Alignment invariance | All 30 | Packer | `mhr_pose[f0]`, `mhr_scales`, `mhr_shape` bit-identical to v8 |
| Changed arrays only | All 30 | Packer row diff | Only arrays 0 and 4 differ; counts recorded |
| Human jitter reduction | All scored triples | D.3 | Pilot criterion (C.3) at frozen λ_H |
| Human displacement | All frames | D.3 | Mean ≤ 1.0 cm; p99 of per-frame max ≤ 4.0 cm; CD-H bound mean ≤ 2.0 cm |
| Object jitter | All scored triples | D.3 | Mean aligned `|a_t|` ≤ 0.10 cm (s_A≈1 assumption stated) |
| CD-O bound | All scored frames | D.3 | `B_O,e` ≤ 2.0 cm per episode; report `B_O` |
| Silhouette | All scored frames | `track1_cdo_full_image_review.py` | Per-episode IoU change ≥ −0.002 (odd frames reported separately) |
| PEN (local) | All scored frames, combined humans and object | Hand-penetration review | §E.4 |
| Visual | Full videos for 3 episodes (7, 11, plus the worst displacement); 6-frame sheets for all others | Render tool | No new swimming, skating, lag or interpenetration; reasons logged in `visual_reviews.json` |
| Integrity | All 30 | Packer | Finite values, orthonormal rotations, positive scale, mesh budget, 740,780 rows |

Evidence levels stay separate [H §5.3]. A local proxy is never reported as an official score.

### E.3 Tolerances and missing evidence

- Floating-point comparisons: `atol = 1e-9 m` for smoother tests; exact equality for the pinned or unchanged arrays.
- If a check cannot run (for example, a missing mask for some frames), record `null` and the reason. **Never write 0.** Treat a missing check as a failure for promotion unless the frame is outside the scored set.

### E.4 PEN policy (corrected direction preserved [H §5.2])

- **Method:** compute local full-hand and FPS512 mean penetration (predicted hands inside the actual submitted mesh, absolute generalized winding > 0.5, triangle distance) per scored frame. Do this for v8, the E1-only candidate, the E2-only candidate, and the combined v9.
- **Episode-level gate:**
  - The candidate full-hand mean must not increase in **any scored frame** beyond numerical tolerance.
  - The candidate FPS512 mean must not increase in **any scored frame** beyond numerical tolerance.
  - Report the equal-episode aggregate; per-frame non-increase is still required.

  These gates preserve the conservative v8 policy; local hand sets still differ from the host and do not certify official PEN.
- **Contact preservation:** on frames where the parent's unsigned hand–object gap is ≤ 1.5 cm (contact-like [H §6.1]), the candidate gap must not increase by more than 0.5 cm. This is the same as v8.
- **Repair loop (at most 3 rounds), per stream:**
  1. Frames violating per-frame non-increase (tolerance 1e-6 m), or the contact-gap limit, become **anchors**.
  2. Set `w_f = 1e4` toward the parent value there, plus a ±3-frame taper of 1e2, then re-solve.
  3. After three rounds, if the episode still fails, **retain the parent for that stream in that episode**. Log the reason. The other stream may still be promoted.
- Anchoring can create local acceleration spikes, so re-run D.3 after each round. Promotion uses the final round's numbers.

### E.5 Promotion rule per episode and stream

Promote the v9 stream for an episode only if:

1. All E.2 rows pass for that episode and stream.
2. The scored-triple acceleration magnitude reduction is ≥ 30%. (Below that, there is no meaningful gain; retain v8.)
3. The visual review raises no blocker.

Otherwise, retain v8 for that stream. Record every decision in `selection_review.json`, with hashes and reasons.

### E.6 v10 promotion (geometry), stricter because geometry changes CD-O directly

Promote a geometry change only if all of the following hold:

- (a) Odd-frame IoU improves by ≥ 0.02. Choose the candidate on even frames only.
- (b) Mask-edge chamfer, in image pixels, improves on both halves.
- (c) Triple-consistency or size-prior evidence supports the direction of change.
- (d) The hand gate passes, with the redesigned proximity rule for the hoop only (below).
- (e) Visual review of the full video passes.

Change **one factor at a time** per episode, in this order: mesh, then scale, then depth. Each factor needs its own ablation row. Never add a constant ray shift alone (v7 lesson).

**Redesigned hoop proximity rule (E4 only).** A wrong control should not define "contact."

- Define contact-like frames from **image evidence**: the dilated (5 px) hand mask overlaps the hoop mask in ≥ 1% of hand pixels.
- On those frames, require the predicted hand–hoop gap to be ≤ 3 cm.
- Keep penetration bounded absolutely: local mean ≤ 0.05 cm per frame.

Report both the old rule and the new rule. **Ask the user before promoting under the new rule** (Q3 in §H.0), because it changes an established gate.

---

## F. Resource plan and resume

| Stage | Where | VRAM | RAM | Disk | Runtime (guess; measure in pilot) |
|---|---|---|---|---|---|
| Whittaker fits, 30 episodes × 2 streams × 6 λ | Host venv, CPU | 0 | < 2 GiB | about 30 × 6 × 2 × 16 MB ≈ 6 GB worst case (NPZ copies; keep only the chosen λ after the pilot) | Minutes |
| Human decode and motion report | Docker `v2d_cari4d:blackwell` | < 4 GiB | < 8 GiB | Small | Minutes per λ |
| IoU review (all scored frames) | Docker, GPU | Per v8 history (one or two processes proven) | — | Mask sheets: hundreds of MB | Hours; measure on the pilot |
| PEN review (9,877 scored frames × 3 candidates) | Docker, GPU, fused winding | Measure; dense meshes up to 757,976 faces [H §6.4] | — | Caches: GB | Hours; measure per episode on the pilot (Episodes 1 and 11 bracket the cost) |
| Full videos (3 episodes) | Docker | Moderate | — | About 1–2 GB | Under 1 hour |

- **Concurrency:** at most two GPU subprocesses, as proven in v8. Start with one; add a second only after `nvidia-smi` shows peak memory < 11 GiB per process.
- **Run order:** sequential heavy jobs; fits run in parallel on CPU.
- **Resume:** every per-episode output directory has `done.json`, containing input hashes, the source fingerprint (SHA256 of the smoother, driver and checker files), parameters and output hashes. `--resume` skips only complete, hash-matched directories and recomputes anything partial. Use an exclusive lock file per experiment (pattern from `track1_contact_campaign.py`).
- **Interruption:** partial directories are recomputed, never patched. PEN caches are bound to the checker fingerprint, so do not reuse them after source edits [H §6.4].
- **Disk:** check `df -h` before each stage. There are 631 GiB free [H §8], so disk is not a constraint, and preserve pilot candidates and diagnostics for reproducibility.

---

## G. All-30 package, provenance, submission and rollback

1. **Parent binding:** `parent_manifest.json` records the v8 directory, the SHA256 of all 60 files, and the v8 Parquet SHA256 `bd73a931e9f9ec85b920884415c5dc2123d86e2d3d91395c7820ebf564fd3198`. Abort on mismatch.
2. **Staging:** `combined/candidate_episodes/` gets, per episode, v8 arrays with the promoted human and object streams. GLBs are copied byte-identical in v9.
3. **Validation:** run the D.4 packer without `--commit` (stage and validate only). Required:
   - 740,780 rows.
   - Only arrays 0 and 4 changed.
   - `f0` rows unchanged.
   - Changed-row counts reported per episode.
4. **Provenance:** commit only the new scripts, tests and reports (`git add` explicit paths; **never `git add .`** [H §9.2]). Push so the commit is publicly accessible, then pack with the full 40-character commit URL. Never reuse a historical URL.
5. **Official checks:**
   - `tools/preflight.py --track track_1`
   - `submit/v2d_submit.py NEW.parquet --track track_1 -m "v9 temporal: pinned human + object translation Whittaker" --dry-run`

   Record SHA256 and row count.
6. **Scoring-policy question (resolve before upload):** determine from the official rules whether the final standing uses **one selected package per competition** or the **best per metric**.
   - If per-metric best counts, a v9 that slightly worsens PEN cannot lower our displayed PEN (v1's 0.00213 remains).
   - If one package must serve all metrics, keep the conservative §E.4 PEN policy.
   - Either way, aim for a Pareto improvement over v8.
7. **Upload (only with user authorization):**
   - Pass `--wait --receipt reconstruction/data/v2d_challenge/track_1/submission_packages/track1_v9_temporal.kaggle-receipt.json`.
   - Use one upload guard file, and do not delete existing guards [H §10.4].
   - Poll exact IDs only. If one metric's upload is ambiguous, inspect remote history before any retry. Never re-upload all five.
8. **Rollback:** v1–v8 Parquets, receipts and source pairs stay immutable. If v9 regresses any metric beyond the §A.2 tradeoffs, v8 remains the parent for v10. Record the failure and its attribution (E1 vs E2 using B.1(3)/(4)) in `submission_results.md`.
9. **v10:** the same procedure with parent = v9 (or v8 if v9 is rejected). `--allowed-arrays` lists only the arrays the promoted geometry changes touch (for example `3,4,5,6,7` for Episodes 0 and 2). Report the changed rows **per episode**, and fail if any non-target episode changes.

---

## H. Ordered execution checklist and final deliverables

### H.0 Questions for the user, asked only when reached

1. **Before the first v9 upload:** "v9 passed all local gates [summary]. Upload v9 to all five Track 1 competitions? (This uses one of this week's submissions.)"
2. **Only if the official rules are ambiguous** about per-metric best vs single package: "Should PEN be held strictly non-increasing, or may we trade ≤ 0.0005 cm of PEN for large ACC gains?"
3. **Before promoting E4 under the redesigned hoop proximity rule:** confirm acceptance of that gate change.

Everything else is reversible and isolated. Proceed without asking.

### H.1 Phase 0: reconcile (read-only, about 30 minutes)

```bash
cd /home/alien51/video_to_data
git fetch && git status --short && git log -3 --oneline     # expect v2d-track1-blackwell, untracked rollout/*inventory.json only
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
docker ps --format '{{.Names}} {{.Status}}'; df -h .
sha256sum reconstruction/data/v2d_challenge/track_1/submission_packages/track1_contact_smoothing_v8.parquet  # must equal bd73a931…3198
ls reconstruction/data/v2d_challenge/track_1/contact_smoothing_v8/submission_episodes | wc -l             # expect 60
```

Then resolve every row of B.4 and write the answers to `incident_reports/track1_improvement/v9_temporal/execution_log.md`. In particular:

- Map the angle dimensions of `pose`.
- Record the ACC-H joint indices.
- Check NPZ coverage and `valid_input`.
- Check how the hand-review tool loads humans.

**Decision:** if NPZ arrays are not full-video, switch D.1 to per-run smoothing on scored stretches only. Gaps are unscored, so this remains correct.

### H.2 Phase 1: implement and test v9 tooling (about half a day)

1. Write `track1_whittaker.py` and `test_track1_whittaker.py`, then run the tests:

   ```bash
   reconstruction/.venv/bin/python reconstruction/scripts/test_track1_whittaker.py
   reconstruction/.venv/bin/python reconstruction/scripts/test_track1_contact_smoothing.py
   ```
2. Write `track1_v9_temporal.py`, `track1_v9_motion_report.py` and `track1_package_from_parent.py`, and add the packer tests (E.1 #7).
3. If needed, patch the hand-review tool to accept candidate humans (D.5) and add a test that changed humans change the evaluated hand points.
4. **Commit the tooling** (explicit paths). Its SHA becomes the producing commit if v9 is packaged from it.

### H.3 Phase 2: v9 pilot (Episodes 1, 4, 7, 11, 13, 24, 28)

1. Write `parent_manifest.json` (hash all v8 files).
2. Run the human stream for each λ in the grid, then the motion report in Docker. Use the D.3 container pattern [H §10.3]:

   ```bash
   docker run --rm --gpus all --user 1000:1000 -e LOGNAME=uid1000 -e HOME=/tmp \
     -v /home/alien51/video_to_data/reconstruction:/repo --entrypoint python v2d_cari4d:blackwell \
     /repo/scripts/track1_v9_motion_report.py --exp v9_temporal --stream human --episodes 1 4 7 11 13 24 28
   ```
3. Choose λ_H as the **largest** λ meeting the displacement caps (C.3 rule 1) on even frames. Freeze it in `pilot_results.md`.
4. Run the object stream for each λ (with `--human-dir` set to the frozen-λ_H outputs), then the motion report with the object bound. Choose λ_O as the largest λ meeting C.3 rule 2. Freeze it.
5. Run the IoU review (all scored frames) and the PEN review for the E1-only, E2-only and combined candidates. Apply the E.4 repair loop.
6. Render full videos for Episodes 7 and 11, and inspect them.
7. **Gate:** apply the C.3 expansion rule. Write `pilot_results.md` with tables of λ vs acceleration, displacement, bounds, IoU and PEN for every pilot episode. Commit the reports.

### H.4 Phase 3: v9 all 30

1. Run both streams at the frozen λ values for the remaining 23 episodes, including Episode 16 (derived-only change, no rerun).
2. Run the D.3 report, all-scored-frame IoU and PEN with repair, then 6-frame sheets for all episodes and full videos for the worst-displacement episode.
3. Apply E.5 per episode and stream. Write `selection_review.json` and `visual_reviews.json`.
4. Stage the pairs, then run the D.4 packer without a commit and validate (§G.3). Commit the reports.
5. Pack with the commit URL, run preflight, then run the dry run. **Ask H.0 Q1.** Upload only if authorized. Poll exact IDs, then write `submission_results.md`:
   - All five official scores, side by side with v8.
   - Attribution notes per B.1.
   - Whether I1's predictions held (ACC-O ≤ local |a| + 0.074).

### H.5 Phase 4: v10 audit and targeted geometry (after v9 is scored, or in parallel on CPU)

1. **Risk audit:** run `track1_v10_risk_audit.py` and produce a ranked table. Prioritize episodes with:
   - The lowest IoU.
   - Triple-consistency ratios outside 0.85–1.15.
   - Area outside the largest component above 2%.
   - Many asymmetric-object rotation jumps.
2. **E4.0, hoop pre-check (CPU):** compare Episode 0 and 2 ring diameter × scale, and IoU, against Episode 1's accepted metric hoop.
   - **Skip E4** if the diameters agree within 10% and IoU ≥ 0.60.
   - Otherwise, run the E4 candidates for 0 and 2 (one factor at a time) and apply E.6.
3. **E5, flagged triples:**
   - First, do floater cleanup only. Remove connected components below 2% of area **and** with < 1% projected overlap with the object mask on scored frames. Run IoU and hand checks.
   - Then, if still flagged, try triple mesh replacement with a scale and ray-depth fit (even frames), and validate on odd frames.
4. Assemble v10 from the v9 parent. The packer restricts changes to the targeted arrays and episodes. Commit, pack, dry-run, then ask the user before uploading.

### H.6 Phase 5 (optional, time permitting before 2026-11-01): v11 rotation flips (E6)

- Run only on asymmetric-object episodes flagged by the audit.
- Use DP over flip hypotheses, with per-frame IoU and edge cost plus a geodesic smoothness penalty. Promote only under E.6.
- In the unlimited final window (2026-11-01 to 2026-11-04, if the rules confirm it), use the remaining uploads for **already-validated** single-factor candidates. Do not use them for hidden-score search.

### H.7 Final deliverables (per round)

- Code: `track1_whittaker.py`, `track1_v9_temporal.py`, `track1_v9_motion_report.py`, `track1_package_from_parent.py`, `track1_v10_risk_audit.py`, tests, and any hand-review patch.
- Data, uncommitted and isolated: `reconstruction/data/v2d_challenge/track_1/v9_temporal/…` and `v10_geometry/…`.
- Reports, committed: `incident_reports/track1_improvement/v9_temporal/`, containing
  - `execution_log.md`
  - `pilot_results.md`
  - `selection_review.json`
  - `visual_reviews.json`
  - `package_validation.json`
  - `submission_results.md`
- v10 gets the equivalent report files.
- **Update the handoff file:** append v9/v10 rows to `CHALLENGE_STATUS_AND_LLM_HANDOFF.md` §3, covering scores, IDs, SHA256 and producing commit. Keep v1–v8 history intact [H §14].

---

## Appendix 1. Why not simply copy the leader's approach or rerun everything?

- **Full reruns:** a 300-step or full-clip refinement did not fix flips [H §7], and an all-30 rerun costs days on one laptop GPU.
- **Hidden-score probing:** blind depth shifts and repeated leaderboard queries are unreliable [H §7, §12.7], and probing wastes quota.
- **Human hybrids:** a v2 human hybrid changes the first-frame alignment and could move every object metric unpredictably (B.1(3)). Pinned smoothing gets the human temporal gain without that coupling.

## Appendix 2. Quick reference: ranks implied by target scores (2026-10-16 snapshot)

Rank = 1 + number of teams with a strictly lower score, as defined in [H §4].

| Metric | Value | Implied rank (of 26) |
|---|---:|---:|
| ACC-H | 0.30 | 19 |
| ACC-H | 0.20 | 16 |
| ACC-H | 0.15 | 10 |
| ACC-O | 0.25 | 17 |
| ACC-O | 0.15 | 14 |
| ACC-O | 0.10 | 11 |
| CD-O | 23.16 | 7 |
| CD-O | 23.04 | 5 |

Other teams will also move before the freeze, so treat these as orientation only.

## Executed local outcome

The corrected temporal pilot has now run:168 isolated fits and84 decoded human candidates. Both streams fail necessary expansion gates; no all30 temporal expansion or upload occurred. See `reconstruction/incident_reports/track1_improvement/v9_temporal/pilot_results.md` and `execution_log.md`. The all30 v8 geometry audit also disproves I3's missing-hoop premise: Episodes0,1,2 already share identical mesh/scale. Keep v8 as the submitted parent. The unverified assumptions and optional future experiments above are not completed results.
