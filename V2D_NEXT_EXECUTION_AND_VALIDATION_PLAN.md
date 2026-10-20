# V2D Next Execution and Validation Plan

**Review status:** corrections are required before execution. Read [the execution-laptop review](V2D_NEXT_EXECUTION_AND_VALIDATION_PLAN_REVIEW.md) alongside this proposal.

- **Branch:** `v2d-track1-blackwell`
- **Audited commit:** `cff4d10` ("Implement cross-track improvement tooling and planning handoff")
- **Audience:** the execution assistant on `/home/alien51/video_to_data` (WSL2, RTX 5090 Laptop, 24,463 MiB, 21,000 MiB guard)
- **Freeze:** 2026-11-04 22:00 UTC (17:00 EST)
- **Supersedes:** the ranking in the [historical cross-track plan](https://github.com/JavierMedel/video_to_data/blob/cff4d10d3267cdaf1b2f6a6c98be1500bdeb304e/V2D_CROSS_TRACK_IMPROVEMENT_PLAN.md). The superseded file was removed from the current tree and remains in Git history.
- **Inputs:** `V2D_CROSS_TRACK_IMPLEMENTATION_HANDOFF.md` §11 and `CROSS_TRACK_IMPLEMENTATION.md`.

Every statement is tagged so its evidence level stays visible:

- **[FACT]** Verified in source, a committed report or the public dataset during this planning pass.
- **[HYP]** A hypothesis that must be tested before anyone relies on it.
- **[OPEN]** An unresolved question, usually about the rules. The default is the conservative choice.
- **[PH]** A placeholder that the execution laptop must resolve locally.

No score gain in this document is a measurement.

---

## 0. Executive summary

1. **The implementation is sound enough to use.**
   - An independent read-only audit, plus my own spot checks, found **no critical or high-severity defects**.
   - It found one medium and five low or design issues (D1–D6, §1.3). All are small, local fixes; apply them before the first GPU job.
2. **Two previous assumptions are wrong. Each one changes the plan.**
   - **Track 2 Tier 2 ships no camera intrinsics or extrinsics.** [FACT, HF `tier_2_synthetic_noise/README.md`]
     - "Image-based refinement" therefore first needs per-episode camera self-calibration.
     - Neither HF tier publishes a Tier 3 reconstruction (`track_2/` contains only `tier_1_multiview_caption` and `tier_2_synthetic_noise`). Tier 3 means our own monocular reconstruction from the byte-identical exo video.
   - **ACC-H and ACC-O are errors against the reference acceleration**, not our own acceleration (`ACC-O.py` L261–282). [FACT]
     - The current λ rule is "strongest λ that passes the gates", and the gate is "own acceleration does not increase". That rule can over-smooth real motion such as grasps, lifts and placements.
     - λ must be chosen against a surrogate with known ground truth (§5.4).
3. **The handoff's "participant MHR decoder export" blocker can be solved.**
   - The official `_MHRBody` is a NumPy port of the Apache-2.0 TorchScript `mhr_model.pt` that we already hold.
   - Export its buffers into the NPZ contract, then prove parity on real v8 frames (§3.1, gate G2).
4. **New priority order:**

| Rank | Work | Why it ranks here |
|---:|---|---|
| 1 | Defect fixes + snapshot + rules record | CPU only |
| 2 | Track 2 Tier 1 per-episode checkpoint re-selection | Local metric equals the official one exactly; GPU cost is evaluation only |
| 3 | **Track 2 Tier 2 baseline-equivalent cohort** | Four empty boards; an inferred upper bound of about 15 GPU-h at 5M steps per episode (§1.2) |
| 4 | Track 1 v11: recentre, contact/PEN, λ by surrogate | Mostly CPU work |
| 5 | Tier 2 denoising/refinement arm | Only after surrogate evidence |
| 6 | Track 3 improvement profile pilots | |
| 7 | Tier 3 own reconstruction | **Blocked on rules Q9 (world anchor).** The packer maps rollouts into the released world with one rigid transform, and scoring has no initial alignment (§3.4). |

5. **Nothing in this plan uploads automatically.** Every upload passes gates G9–G12 (§7) and needs explicit user approval.

---

## 1. Current-state audit

### 1.1 What was inspected and how

| Source | Method | Scope |
|---|---|---|
| `reconstruction/scripts/cross_track/*.py` (track1, temporal, human, mhr_torch, policy, jobs, tier_inputs, track3, packaging, submission, diagnostics) and `run_cross_track_improvement.py` | Line-level read: independent code-review agent plus my own reads | Commit `cff4d10` |
| Official metric code: `metric_code/track_1/ACC-O.py`, `track_2/{AUC,SP-SR,MP-SR,MPPE}.py`, `track_3/*.py`, `config/tracks.json`, `v2dlb/dataset.py` | Read | Kit as committed |
| Existing Track 1 tooling: `mhr_export_utils.py`, `track1_contact_rotation.py`, `track1_mask_render.py`, `mesh_to_mhr_params.py` | Read | Committed |
| HF dataset `nvidia/video_to_data_challenge` `track_2/` tree and both READMEs | HF API, public | Live on planning day |
| Status docs: `TRACK2_STATUS.md`, `TRACK3_STATUS.md`, handoff, runbook | Read | Committed |

**Not inspectable from the planning machine:**

- large local data, checkpoints, rollouts and weights
- Kaggle standings
- the `kaggle` package version on the execution laptop
- TorchScript buffer names inside `mhr_model.pt`

Each of these becomes a local inspection task (§10).

### 1.2 Evidence ledger

| Category | Item | Status |
|---|---|---|
| **Real submissions** | Track 1 v8 | CD-H 15.43955, CD-O 23.48324, ACC-H 1.15894, ACC-O 0.71908, PEN 0.00322. v10 is held and needs re-audit. |
| | Track 2 Tier 1 short campaign (packet commit `fa96985`) | AUC 0.24466, SP-SR 0.10, MP-SR 0.03333, MPPE 53.8482 |
| | Track 3 | AUC 0.16307, SP-SR 0.05, MP-SR 0, RPE 22.28685, MPPE 20.82175, CD-O 0.57037 |
| | Track 2 Tier 2 / Tier 3 | **No submissions.** 8 boards empty. |
| **Passing code tests** | Host suite | 87 |
| | Torch CPU suite | 2 (synthetic, not real assets) |
| | Warm start | 1 (fake env, 4 CPU steps) |
| | Tier 1 recompute | Matches Kaggle within 1e-4 |
| **Unrun experiments** | Real MHR parity; evidence export; any object-fit or human-fit candidate; Tier 2/3 cohorts; Track 3 dev O/M/R/A; reward-profile pilot; warm-start GPU run; authenticated snapshot; real guarded upload | Not run |
| **Missing artifacts** | Participant MHR NPZ asset | Missing |
| | Track 1 object-fit evidence NPZs | Missing |
| | Tier 2 local data | [PH] Check if present |
| | Tier 3 reconstructions | Do not exist anywhere |
| | Rules JSON | Intentionally unresolved |

**[FACT] Measured cost anchor.**

- The Tier 1 campaign used 5,000,192 steps per episode with evaluation.
- Status timestamps show 5/30 evaluated at 2026-10-20 01:57 UTC and all 30 complete by about 15:34 UTC (11:34 Toronto).
- That is roughly 25 episodes in at most about 13.5 h, including packing and Kaggle scoring. The upper bound is about 30 min per episode, so a full 30-episode cohort costs at most about 15 GPU-h.
- **This is inferred from status prose, not a profiler.** Re-measure it (§6).

### 1.3 Defects and limitations found in `cff4d10`

Severity reflects the effect on correctness or results. None of these silently corrupts a submitted packet, because every path is gated.

| ID | Severity | Location | Problem | Fix |
|---|---|---|---|---|
| **D1** | Medium | `jobs.py` L39–45 (`run_jobs`), `policy.py` ~L150 (`training_job`) | Existing `state_*.safetensors` in `output_dir` are adopted or resumed without checking which run made them. A job pointed at a baseline directory relabels the baseline policy as `TRAINED` for the new arm, or appends checkpoints into the baseline lineage. A warm-start arm over a previous fresh run silently resumes and drops `initial_policy_path`. | Write `output_dir/.run_identity.json` (job hash, arm, parent checkpoint SHA) on first launch. Resume only if it matches. Otherwise require the directory to be absent or empty. Refuse any path under `*/baseline/`, `*/policies_short/` or a submitted-packet root. Add tests for all three cases. |
| **D2** | Low (lowers yield) | `track1.py` L113–117 | The PEN cutting-plane loop **replaces** `constraints` each iteration instead of **accumulating** them, so iteration *k* forgets the half-spaces from *k−1*. This can oscillate and use up the 5 iterations. | Keep a deduplicated union `active += new` across iterations (key: frame, rounded normal, rounded bound). |
| **D3** | Low (lowers yield) | `track1.py` L124–131 | The final re-fit after per-frame rigid repair passes only `fixed` (box pins). Previously enforced half-spaces are dropped, so other frames can penetrate again and the gate rejects a λ that could have passed. | `smooth(..., active + fixed)` |
| **D4** | Low, fails closed | `submission.py` L45–47 | `competition_submissions(slug, page_number=..., page_size=100)` may not match the installed `kaggle>=2` signature, which may use `page_token`. `competition_leaderboard_view(slug, page_size=100)` is equally unverified. On failure, snapshot and quota abort and the submission stalls. | Inspect `inspect.signature` locally (§10, step 3). Branch on the parameter name, paginate via `next_page_token` if present, and add a mocked test for each style. |
| **D5** | Low | `track3.py` L168 (`dev_jobs`) | `jobs.json` is overwritten without the `new_run` guard that `plan_jobs` uses. | Call `new_run(output, [manifest], {...})` before `save`. |
| **D6** | Design | `track1.py` selection loop; `gate()` | Choosing the **largest passing λ** with the check `acceleration_nonincrease` (own acceleration) optimises the wrong quantity. ACC-O measures `‖a_pred − a_ref‖`. | Select λ by GCV on residuals **and** cap it by the surrogate study (§5.4). Keep the gate as a safety check, not the objective. |
| **L1** | Limitation | `tier_inputs.py` `condition_positions` / `condition_source` | Each per-frame correction is clamped to the high-frequency noise scale `median‖Δ²x‖/√6`. That removes white jitter but cannot remove random-walk or low-frequency drift, which the folder name `synthetic_random_walk` suggests is present [HYP]. | Not a bug. Treat conditioning as a jitter filter only; drift needs observation (§4.6). |

**Checked and cleared:**

- **`condition_source` loops over `b['poses'].shape[1]`.** `prepare_track2_sources.py` L120 writes `poses` as `reshape(-1, 77, 3)`, so the per-joint loop is correct. [FACT]
- **Quaternion order.**
  - The HF Tier 2 parquet stores `[x, y, z, qw, qx, qy, qz]`, w-first.
  - `prepare_track2_sources.continuous_object_poses` handles wxyz, and `poses.npy` holds matrices.
  - The frame-reference NPZ writes `as_quat()` in xyzw under the key `pose_xyzw`, which the packer expects. [FACT]

**Verified correct by the audit** (no action needed):

- **Track 1 maths:** recentring maths, the stretch split, the acceleration definition and PEN local frame/half-space signs. Variable ordering in `kron(H, I)` matches.
- **MHR torch port:** Euler order, `exp2` scales, correctives, the /100 and FLIP, and root decomposition.
- **Policy and training:** `policy.select` ordering and its Tier-1-only scope; curriculum scaling; warm start (actor only, mutually exclusive with resume).
- **Submission:** 23 slugs, freeze time, receipt and manifest exclusivity.
- **Track 3 mesh:** `_align_rigid` direction.

---

## 2. Corrected priorities

### 2.1 Ranking rule

Rank work by expected benefit per scarce resource. GPU hours before the freeze are the scarcest. Each item's benefit is classed as one of:

- **exact:** the local metric equals the official one
- **proxy:** a local measurement correlates with the official metric
- **entry:** the item fills an empty board
- **speculative**

Baselines stay frozen: v8 Track 1, the Tier 1 short-campaign packet and the Track 3 packet are never modified in place.

| # | Work package | Benefit class | Resource | Depends on | Notes |
|---:|---|---|---|---|---|
| WP0 | Fix D1–D5; record the rules JSON; authenticated snapshot | Enables everything | CPU, about ½ day | none | D6 lands with WP3 |
| WP1 | Tier 1 per-episode checkpoint re-selection (`checkpoint-inventory` → `checkpoint-select`) | **Exact**: same trajectory-derived metric as the packer | GPU evaluation only for missing rollouts | WP0 (D1) | Cheapest certain improvement. Episodes whose best checkpoint is earlier than the last one improve SP-SR, MP-SR and AUC directly. |
| WP2 | Tier 2 baseline-equivalent cohort: Tier 2 inputs as released, existing retarget and training, 5,000,192 steps | **Entry** on 4 boards | At most about 15 GPU-h (§1.2) | WP0, G5, G6 | Value depends on how the overall standing treats empty boards [OPEN Q6]. It is still the most likely large gain per GPU-h. |
| WP3 | Track 1 v11: MHR exporter, parity, evidence adapter, object fit (recentre, contact, PEN, λ by surrogate) | Proxy for ACC-O and PEN; CD-O held neutral by gate | Mostly CPU plus short GPU mask renders | WP0, G2, G3 | Human refit (ACC-H) is a separate, later arm (WP3b) |
| WP4 | Tier 2 noise characterisation, then a refinement arm (camera self-calibration plus silhouette/keypoint refinement) | Speculative until §5.6 passes | CPU first, then about 1–3 GPU-h pilot | WP2 retarget assets | Only replaces the WP2 packet if a surrogate shows a gain |
| WP5 | Tier 1 stronger training: object-focused profile and/or warm start on the weakest episodes | Exact | Measured from pilot | WP1 | Warm start only within Tier 1 |
| WP6 | Track 3 improvement-profile pilot on public dev episodes, then evaluation episodes | Proxy (dev) | Measured from pilot | WP0 (D5) | Dev/eval separation (G8) |
| WP7 | Tier 3 own monocular reconstruction | Entry on 4 boards | Large and unknown | §5.7 feasibility, rules Q3/Q4 | Defer unless the §5.7 cheap path exists |

**Removed or demoted from the old plan:**

- **Gain numbers.** The predicted gains ("CD-O −x%", "ACC −y%") are deleted as unsupported.
- **The original 290-GPU-hour schedule.** Replaced by §6.
- **Treating Tier 2 like a temporal-smoothing problem.** Demoted because of L1 and the missing cameras.

### 2.2 Rules questions (default = conservative)

Record answers, with the source quote and URL, in `reconstruction/data/cross_track/rules/rules_v1.json` [PH: path], following the runbook's rules contract. Do not contact the organisers on the user's behalf; list the question for the user instead.

| ID | Question | Default until there is a quoted answer |
|---|---|---|
| Q1 | Is the weekly submission window rolling 7 days or calendar-based? Where does a calendar week start? | `unresolved`, so the upload gate refuses |
| Q2 | Is there an unlimited period before the freeze? | `false` |
| Q3 | May Tier 1 clean trajectories be used **in any way** when producing Tier 2/3 submissions, including as tuning ground truth? | **No**. Isolation as in the handoff. |
| Q4 | May Tier 2 perturbed trajectories be used as a prior for Tier 3? | **No**. Tier 3 starts from video only. |
| Q5 | May a Tier 2/3 policy be warm-started from a Tier 1 policy? | **No** (inherits Q3) |
| Q6 | How is the overall ranking aggregated across boards, and how is a missing board treated? | Assume a missing board is the worst outcome, so WP2 stays ranked high |
| Q7 | May Track 3 public dev episodes be used to tune hyper-parameters (not as meshes)? | Yes for tuning, per the FAQ intent; dev scans never substitute for meshes |
| Q8 | Is estimating camera intrinsics and extrinsics from the released video plus noisy poses "a method" allowed for Tier 2/3? | Yes ("by whatever method"), because it uses only that tier's own released input |
| Q9 | For Tier 3, which world-frame anchor may register a video-only reconstruction to the released Track 2 world (needed because of §3.4)? Candidates: the released `ground_plane/<sequence>.json`, the `support_surface` stage, or the first-frame pose of a released tier. | **None.** Tier 3 stays blocked (WP7) until there is a quoted answer. |

---

## 3. Input acquisition and export tasks

All new artifacts go in **new versioned directories** and carry SHA-256 values for every parent. Units are metres unless stated otherwise.

### 3.1 Participant MHR body asset: `mhr_body_asset_v1.npz`

**Source:** `reconstruction/data/weights/track1_submission_mhr/mhr_model.pt` (TorchScript, Apache-2.0). The SAM3D copy at `checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt` must have the same SHA, or the difference must be recorded.

**Contract**, exactly what `ACC-O._MHRBody.__init__` reads (L486–505) [FACT]:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `joint_parents` | [127] int | −1 for root |
| `joint_offsets` | [127,3] | Local translation offsets, **cm** (the output is divided by 100) |
| `joint_prerotations` | [127,4] | Quaternion. **[PH] Order must be confirmed against `_quat_to_mat`** (read its body before writing). |
| `parameter_transform` | [127·7=889, 204] | Maps model params (pose 136 + scales 68) to joint params (t3, euler3, log2-scale1) |
| `inverse_bind_pose` | [127,8] | t3 + quat4 + s1 |
| `base_shape` | [V·3] | Rest vertices, cm |
| `shape_vectors` | [45, V·3] | Identity blendshapes |
| `skin_dense` | [V,127] | LBS weights. Rows must sum to 1 (check). |
| `corr_w1_idx` / `corr_w1_val` / `corr_w1_shape` | COO [2,nnz] / [nnz] / [2] | Pose-corrective layer 1. Input = 6D rotation features of joints 2..126. |
| `corr_w2_data` / `corr_w2_indices` / `corr_w2_indptr` / `corr_w2_shape` | CSR | Pose-corrective layer 2 → V·3 |
| `role_alignment`, `role_body`, `role_left_hand`, `role_right_hand` | int index arrays | **Host roles are hidden.** See below. |

**Extraction steps** (Torch container `v2d_cari4d:blackwell`, CPU, read-only mount, output to a new directory):

1. **Inspect, read-only:**
   - Dump `named_buffers()`, `named_parameters()`, `named_modules()` and every tensor attribute (name, shape, dtype, sparse layout) to `mhr_torchscript_inventory.json`.
   - Dump `model.code` and each sub-module's `.code` to text.
   - Expected names are unknown [PH], so map them by **shape**: 127, 889×204, V×127, and so on.
2. **Map** each contract key to a TorchScript tensor:
   - Convert sparse COO/CSR as needed.
   - Convert quaternions to the order `_quat_to_mat` expects.
   - Record the mapping in `mapping.json`.
3. **Roles (participant proxies):**
   - `left_hand` / `right_hand`: the vertex indices from `modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz` (`vertex_indices`, shape [2, N]; confirm which row is left).
   - `body`: all vertices.
   - `alignment`: all vertices minus hand vertices.
   - Write `role_source = "participant_proxy_v1"` into a sidecar JSON.
   - **[HYP]** Proxies make local CD-H and PEN diagnostics *approximate*. They never claim the host value.
4. **Expression:** `_MHRBody` has no expression input, and `mesh_to_mhr_params.py` calls the TorchScript with 72 zero expression coefficients. The parity test therefore runs the TorchScript with zero expressions.
5. **Units and axes hazard** [FACT]. Three wrappers use three conventions:
   - official: `/100` and `FLIP = [1,−1,−1]`, giving metres
   - `mesh_to_mhr_params.py`: `diag(1,−1,−1) @ MHR(...)/100`
   - `track1_contact_rotation.py`: divides `MHR(...).run` output by **1000**, so that wrapper returns mm

   The parity test must compare in metres after each wrapper's own conversion and assert all three agree.

**Gate G2 (parity):**

- **Frames:** at least 100 real frames sampled across at least 10 v8 episodes (`pose`, `scales`, `shape` from the v8 bundles). Use a stratified sample that includes extreme hand poses.
- **Bound:** maximum vertex error ≤ 1e-5 m, and maximum joint error ≤ 1e-5 m, between:
  - official `_MHRBody(asset)`
  - TorchScript run directly
  - `cross_track/mhr_torch.py` built from the same asset
- **Output:** `decoder_parity.json` (`human.py` already writes this format).
- **Failure:** the human refit (WP3b) stays blocked. The object fit (WP3) still proceeds, because it only needs hand vertices, and those come from the TorchScript directly.

### 3.2 Track 1 object-fit evidence NPZ (one per episode)

**Contract** [FACT, `track1.py` L83–100]:

| Key | Shape | Rule |
|---|---|---|
| `parent_npz_sha256`, `parent_mesh_sha256` | scalar str | SHA of the exact v8 `episode_XXXXXX.npz` and `_object.glb` |
| `coordinate_frame` | scalar str | `'prediction_world_meters'` |
| `frames` | [F] int | Scored frame indices from `reconstruction/incident_reports/track1_scored_frames.json` |
| `masks` | [F,H,W] bool | Object mask at full camera resolution |
| `intrinsics` | [3,3] or [F,3,3] | From `camera_calibration(load_edex(export), 0)` |
| `world_to_camera` | [F,4,4] | **Identity** for Track 1, as `track1_contact_rotation.py` already asserts |
| `hand_points` | [F,N,3] m | Decoded MHR hand-surface vertices in prediction world |
| `hand_normals` | [F,N,3] unit | Vertex normals of the decoded full mesh at the same indices |

**Adapter:** new file `reconstruction/scripts/cross_track/track1_evidence.py`. It reuses functions verbatim from `track1_contact_rotation.py` L36–75:

- `load_edex`, `camera_calibration`, `read_mask(export, 'object', 0, f'{frame:06d}')`
- `MHR(model_path, 'cuda'|'cpu', 4, 'float32').run(pose, r_[scales, shape])`, divided by 1000, then `hand_ids` from the spec

Export path pattern [FACT]:

- `data/<improvement_pilot_v1|improvement_rollout_v1>/episode_XXXXXX/run/export/episode_XXXXXX`
- Pilot episodes 6, 18 and 28 use `improvement_pilot_v1`.

Normals: compute with `trimesh.Trimesh(verts, faces).vertex_normals`. Faces come from the TorchScript inventory [PH: buffer name] or the hand spec if it carries faces.

**Split and convention checks:**

- **Index alignment.** `len(bundle['pose']) == number of export frames`. Every scored frame has a mask file. Mask `H,W` equals the image size in the edex intrinsics, so the principal point lies inside the image.
- **Reprojection sanity.** Render v8 object silhouettes on 20 frames and compute IoU against the masks. The median must equal what `image_choices.npz` (`ious[:,0]`) recorded for the parent. A mismatch means a frame or axis error, and the episode stops.
- **Hand sanity.** The hand points' centroid projects inside the human mask on ≥ 95% of frames where the human is visible.

**Episodes:** pilot 6, 18 and 28 first (§5.3), then all 30.

### 3.3 Track 2 Tier 2 inputs

- **Download:** HF `nvidia/video_to_data_challenge`, path `track_2/tier_2_synthetic_noise/`, at revision `5f68335f3acc802033d1e80728c1633197521de8`.
  - [FACT] That is HF `main` at planning time, and it is the revision `prepare_track3_campaign.py` already pins. Both tiers exist at it.
  - Required: `data/`, `videos/`, `mesh/`, `ground_plane/`, `support_surface/`, `meta/`.
  - Store it under `reconstruction/data/v2d_challenge/track_2/tier_2_synthetic_noise/`. That is the dataset root the Tier 1 campaign passes to the packer (`run_track2_tier1_campaign.py` L237), and `v2dlb/dataset.py` L29–42 finds the folder under that name.
- **Facts** (HF README):
  - 30 episodes, 22,990 frames, 30 fps
  - human `pose` [231] = 77 rotation vectors
  - object `pose` `[x,y,z,qw,qx,qy,qz]` in the OpenCV world
  - zero pose when `visible == false`
  - exo video 1536×1152, frame-exact
  - **no intrinsics or extrinsics**
- **Checks:**
  - Frame counts per episode equal Tier 1 counts (same timeline) — a structural check that uses no Tier 1 values.
  - Mesh SHA equals Tier 1 mesh SHA (the README says the mesh is the original).
  - Video SHA equals the Tier 1 video SHA (byte-identical, per the README).
- **Steps:** `prepare-tier --tier 2`, then native retarget, then `tier-inventory --tier 2`. **Episode 16 must be included explicitly.**

### 3.4 Track 2 Tier 3 inputs

- **No published reconstruction** [FACT]. The input is the same exo MP4s. Any Tier 3 source must come from our own pipeline, with `source_manifest.json` listing the video SHAs that `tier_inputs.py` L22–30 validates.
- **The packer cannot find Tier 3 input on its own.** `v2dlb/dataset.py` L29–42 looks for `track_2/cari4d_reconstruction` or `track_2/tier_3_cari4d`. Neither is in the participant release [FACT]. Tier 3 packing therefore **must** pass `--frame-references <dir>` with one `episode_XXXXXX.npz` per episode (`pack_submission.py` L132–160).
- **World-frame constraint, which decides feasibility** [FACT, `pack_submission.py` L95–130 and L380–390]:
  - The packer maps the rollout into the frame-reference world with **one rigid transform**, anchored on frame 0 of the evaluator reference vs the frame reference.
  - It checks that the two references differ by that rigid transform on all frames.
  - Track 2 scoring then has **no initial-pose alignment**.
  - So a Tier 3 reconstruction must be expressed in the *released* source OpenCV world. `tier_inputs.validate_root` already enforces this: for Tier 3 it requires `coordinate_frame == 'released_track2_opencv_world'` ("independently established"). A monocular video does not determine that world (camera-to-world gauge), unless something rule-permitted anchors it. See §5.7 and Q9.

### 3.5 Track 3 public-development bundles

- **Location:** `reconstruction/data/v2d_challenge/track_3/public/`, dataset revision `5f68335f…`. It contains 20 dev episodes with references and scans.
- **Use:** `track3 dev-jobs` needs **exactly four** public dev episodes with `split: "development"`.
  - Each has arms `O`, `M`, `R`, `A`, each a prepared native `reference` plus `reference_sha256` (runbook §"Public development decomposition").
  - Generated jobs use 10,000,384 steps, so 16 jobs per decomposition. **Measure Sharpa throughput before launching** (§6).
- **Pick the four dev episodes** for object diversity: at least one rigid small object (cup or spoon), one articulated or thin object, and one two-object episode.
- **Split check:** dev episode IDs ∩ scored evaluation IDs = ∅ (assert in the manifest builder). Dev scans never enter evaluation packets (G8).

---

## 4. Implementation changes

Every change bumps the method or configuration string, so `new_run` fingerprints change and old runs are never resumed under new code. Each change gets focused tests in the existing `reconstruction/scripts/test_cross_track_improvement.py` unless noted.

### 4.1 D1: job output identity (`jobs.py`, `policy.py`)

- **Current:** any `state_*` in `output_dir` is adopted or resumed (§1.3).
- **Proposed:**
  - `training_job` adds `run_identity = sha256(json(track, episode, reference_sha, arm, parent_sha, steps, reward_profile, overrides))` to the job.
  - `run_jobs`:
    1. If `output_dir/.run_identity.json` exists, it must equal the job's identity. Otherwise raise `Output directory belongs to another run`.
    2. If the file does not exist but `state_*` or `policy_*` files exist, raise.
    3. Otherwise create the directory and write the identity file before launching.
  - `training_job` refuses outputs whose resolved path contains `/baseline/` or `/policies_short/`, or that lie under any directory containing a `kaggle_receipt.json`.
- **Integration:** the identity is written **inside the output directory**. Training containers mount it read-write already. Nothing else changes.
- **Compatibility risk:** an existing unfinished job without the identity file will refuse to resume. That is intended; resume it through an explicit `arm: resume` job.
- **Tests:**
  - fresh arm into a baseline-like directory → raises
  - fresh arm into a directory with foreign checkpoints → raises
  - matching identity → resumes
  - warm-start arm over a fresh run's directory → raises

### 4.2 D2 and D3: PEN cutting planes accumulate (`track1.py` `object_candidate`)

- **Proposed:**
  - Keep a dict `active[(frame, tuple(np.round(n, 6)), round(bound, 7))] = (frame, n, bound)`.
  - Each loop iteration adds the new constraints. `smooth(...)` receives `list(active.values())`.
  - The final re-fit receives `list(active.values()) + fixed`.
  - Raise the iteration cap from 5 to 10. With accumulation, constraints only grow, so the QP is monotonically more restricted and terminates.
- **Configuration:** `method: 'recenter_huber_contact_pen_v2'`.
- **Test:** a synthetic unit cube moving along x, with hand points arranged so that two frames alternately penetrate under replacement. v1 behaviour ends with depth > 0; v2 must end at depth 0.

### 4.3 D4: Kaggle client compatibility (`submission.py` `snapshot`)

- **Proposed:** a helper `_page(api, slug, token_or_number)` that inspects `inspect.signature(api.competition_submissions)`:
  - `page_token` → loop on the returned `next_page_token` (or the attribute the client exposes [PH: confirm in §10 step 3])
  - `page_number` → current behaviour

  `competition_leaderboard_view` likewise passes `page_size` only if it is accepted.
- **Tests:** two fake APIs, one per style. The existing fake (`test_cross_track_improvement.py` L176) stays.

### 4.4 D5: `track3.dev_jobs` new-run guard

`new_run(output, [manifest], {'manifest_sha256': sha(manifest)})` before `save`. Test: a second call into the same output with a different manifest raises.

### 4.5 D6: λ selection by GCV, capped by the surrogate (`temporal.py`, `track1.py`)

- **Current:** the largest λ in `[3, 10, 30, 100, 300]` that passes all gates wins.
- **Proposed:**
  1. **`temporal.gcv_lambda(values, frames, lambdas, weights)`** returns λ̂ minimising `GCV(λ) = n·RSS(λ) / (n − tr H_λ)²` for the unconstrained weighted Whittaker smoother on each stretch.
     - `tr H_λ` is computed exactly from a sparse banded factorisation. A stretch of ≤ 1,100 frames is cheap.
     - Pooled across stretches.
  2. **The candidate set** is λ ≤ min(λ̂·2, λ_cap), where λ_cap comes from the §5.4 surrogate study and is written to a committed JSON `track1_lambda_cap_v1.json` with its evidence hash. Among those, choose the largest λ that passes the gates.
  3. Add review fields `gcv_lambda`, `lambda_cap` and `removed_acc_fraction`, where `removed_acc_fraction` = mean‖a_removed‖ / mean‖a_baseline‖. This is a diagnostic: values above 0.9 suggest real motion is being erased.
- **Integration:** `object-fit --lambda-policy {max_passing,gcv_capped}` (new flag), default `gcv_capped`. `max_passing` stays available for an A/B.

### 4.6 New: `cross_track/mhr_asset.py` (exporter + parity)

- **Subcommands** (added to the `run_cross_track_improvement.py` choices and module map):
  - `mhr-inventory`: TorchScript inventory to JSON (CPU)
  - `mhr-export --model --hand-spec --mapping --output`
  - `mhr-parity --asset --model --bundles <v8 dir> --frames 100 --output`
- **Behaviour:** §3.1. The export fails if any contract key is unmapped, shapes disagree, `skin_dense` rows do not sum to 1 within 1e-6, or `parameter_transform` is not 889×204.
- **Integration:** `human-fit --body-asset` and `diagnostics` (`body_asset`) consume the NPZ unchanged.
- **Risk:** the TorchScript may store the parameter transform or correctives under a different factorisation, for example a dense W1 or concatenated 6D features. The mapping must reproduce the *official* computation. Parity (G2) is the arbiter, not the mapping's plausibility.

### 4.7 New: `cross_track/track1_evidence.py` (`track1-evidence` subcommand)

- **Inputs:** `--parent <v8 dir> --episode N --export-root <PH> --scored <json> --model <mhr_model.pt> --hand-spec <npz> --output <dir>`
- **Behaviour:** §3.2. Writes `evidence_episode_XXXXXX.npz` plus `evidence_review.json`, which holds the reprojection IoU vs `image_choices.npz`, hand-in-human-mask rate and all SHAs.
- **Device:** the CPU path is acceptable for MHR decoding of the hand subset. The GPU is only needed for optional reprojection renders, under the GPU lock.
- **Reuse:** everything comes from `track1_contact_rotation.py` and `mhr_export_utils.py`. No new geometry code.

### 4.8 New: `cross_track/tier2_noise.py` (CPU, Tier 2 data only)

It characterises the perturbation **without Tier 1 values** (Q3). It writes per-episode and pooled numbers:

1. **Object.**
   - Acceleration-norm distribution and per-axis power spectrum of the position.
   - The same for the rotation (log-map increments).
   - Increment autocorrelation: white jitter gives a negative lag-1 autocorrelation of first differences (about −0.5); random walk gives about 0.
2. **Resting segments.**
   - Define frames where the object is far from both hands (> 0.15 m from SOMA-FK wrist joints; [PH: SOMA FK entry point in `robotic_grounding/.../retarget/read_soma.py`]) and its pose changes slowly.
   - Over each segment, measure the drift of the object's **signed height above the released Tier 2 `ground_plane`** and of its horizontal position.
   - A physically resting object has zero true drift, so the measured drift is the noise's low-frequency component. This is rule-safe because the plane belongs to the Tier 2 release.
3. **Human.**
   - Foot-joint height above the ground plane during stance (low foot speed) shows root drift.
   - Bone-length consistency is preserved by design (identity is unchanged), so skip it.
4. **Classification:** each episode is labelled `white`, `random_walk`, `mixed` or `unclear`, with numbers.

**Uses:**

- decides whether WP4 is worth GPU time
- sets λ and noise scales for any Tier 2 conditioning
- adds **resting-height and foot-contact constraints** as a drift remedy that needs no camera (§4.9a)

### 4.9 Tier 2 refinement options (only after §5.6 evidence)

**(a) Constraint-based drift removal, no camera.**

Smooth the object position with:

- equality soft-constraints holding a constant height over each resting segment
- a constant pose over each resting segment
- hand-attachment continuity over contact windows, using SOMA hand joints

Do the same for the human root using stance-foot constraints. Reuse `temporal.smooth` (supports targets, weights and constraints) and `smooth_rotations`.

New function: `tier_inputs.condition_source(..., method='constraints_v1', noise_report=...)`. It writes a **new** source directory, as the existing conditioning does.

**(b) Camera self-calibration plus image refinement** (GPU; heavier).

1. **Per episode, estimate one static camera (K, R, t).** The exo camera is static [FACT, README].
   - Correspondences: the object mesh silhouette, rendered with the noisy Tier 2 poses, against SAM2 object masks, plus 2D human keypoints (ViTPose/WiLoR from existing images) against SOMA-FK joints.
   - Robust fit over hundreds of frames, with Huber loss.
   - Zero-mean noise averages out. Random-walk noise biases it [HYP]. Mitigate by fitting the camera only on resting segments after (a).
2. **Refine per-frame object poses** with FoundationPose tracking (`v2d_track3_foundation_pose:blackwell`), the exact mesh and the estimated K, initialised from the (a)-conditioned pose.
3. **Validate** by held-out reprojection: fit the camera on even frames, measure mask IoU on odd frames.

Write output as a new source directory. `prepare-tier` and the packer frame references follow the existing contract (`coordinate_frame='released_track2_world'`).

### 4.10 Reuse without change

- `checkpoint-inventory` / `checkpoint-select` / `pack-selected` (WP1)
- `prepare-tier` / `tier-inventory` / `plan-jobs` / `jobs` (WP2), after D1
- `assemble-track1` (WP3)
- `track3 mesh` / `dev-jobs` (WP6), after D5
- `submission snapshot` / `submit`, after D4

Warm start already enforces same-tier lineage (`policy.py` checks `/track2/tier{N}/` in the parent's parquet path), which satisfies Q5 by construction [FACT].

---

## 5. Small validation experiments

Each pilot states its baseline, primary metric, feasibility checks, required observations and **minimum gain before expansion**. Run CPU or read-only checks first.

### 5.1 CPU and code checks (P0)

- **Run:** the existing 87-test host suite, plus new tests for D1–D6 and the new modules.
- **Pass:** all green; `git diff --check` clean; producing commit pushed.
- **Then:** the torch CPU suite and warm-start test (§8.2–8.3 of the handoff) re-run unchanged.

### 5.2 MHR parity pilot (CPU container)

- **Run:** `mhr-inventory`, then `mhr-export`, then `mhr-parity` on 100 frames across 10 v8 episodes.
- **Baseline:** none; this is an equivalence test.
- **Primary metric:** maximum vertex error in metres.
- **Pass (G2):** ≤ 1e-5 m on all three decoders, plus a unit/axis agreement report.
- **Observation to record:** time per 100 frames on CPU (for scheduling).

### 5.3 Track 1 object-fit pilot (episodes 6, 18, 28)

- **Baseline:** v8 bundles for the same episodes.
- **Steps:** export evidence (§4.7, G3), then `object-fit` twice per episode: `--lambda-policy max_passing` (v1-like control) and `gcv_capped` (v2).
- **Primary local metrics** (no ground truth exists, so these are proxies):
  - `pen_depth_proxy`: the official `_episode_penetration` on proxy hand roles
  - mask IoU change
  - mean and p99 displacement
  - `removed_acc_fraction`
- **Feasibility:** recentre neutrality (`verify_recenter`) is exact. Candidates pass `assemble-track1` identity and anchor checks for these episodes.
- **Expansion criterion:** in at least 2 of 3 episodes, a λ > 0 passes the gate. PEN proxy is 0 on those. The surrogate (§5.4) predicts an ACC-O reduction for the chosen λ at that episode's estimated noise level. **Otherwise stop WP3.** Track 1 stays at v8; record why.

### 5.4 Surrogate λ study (CPU, Track 1)

**Purpose:** a ground-truth-backed choice of λ_cap and validation of GCV, because ACC-O is measured against a reference acceleration we cannot see.

**Clean corpus** (no Track 1 hidden data):

- Track 3 public dev object reference trajectories (mocap)
- Track 2 Tier 1 released object trajectories, used here as generic clean motion for a *Track 1* tool; not affected by Q3, which concerns Tier 2/3 isolation

Resample to the Track 1 frame rate [PH: read from Track 1 metadata or edex timestamps].

**Noise model:**

- Estimate it from v8's own translations (our predictions, rule-safe): a high-pass residual spectrum and its autocorrelation.
- Inject it at 0.5×, 1× and 2× the estimated level.

**Procedure:**

- For λ in `{1, 3, 10, 30, 100, 300, 1000}`, compute the official `acceleration_error_per_frame` mean against the clean trajectory.
- Record the oracle λ* and the GCV choice per (trajectory, noise level).

**Pass:**

- GCV-capped λ is within 10% of the oracle's ACC error on the median trajectory.
- GCV-capped λ is never worse than unsmoothed on more than 5% of trajectories.
- λ_cap = the largest λ whose 90th-percentile ACC error is ≤ 1.05× the oracle's.

**Output:** `track1_lambda_cap_v1.json` with corpus hashes.

### 5.5 Tier 1 checkpoint re-selection (WP1)

1. **`checkpoint-inventory`** over all preserved Tier 1 actor checkpoints and existing results.
2. **Determinism pilot:**
   - Pick 3 episodes with at least 3 checkpoints each.
   - Run `checkpoint-select --run-evaluations` on a 3-episode candidates file. The tool requires the full roster, so if it refuses, run inventory evaluations per episode [PH: confirm whether a subset mode exists; if not, add a `--episodes` filter that only skips `select`'s completeness check in pilot mode and writes `LOCAL_SUBSET_DIAGNOSTIC`].
   - Re-evaluate one checkpoint twice. The trajectory SHA or the metrics must be identical. **If evaluations are not deterministic, stop.** Selection would be overfitting to noise; record the variance instead.
3. **Full:** all 30 episodes, with missing rollouts evaluated on GPU.
4. **Minimum gain to pack:**
   - The selection's local mean has SP-SR ≥ 0.10 and MP-SR ≥ 0.0333 (not worse).
   - AUC improves by ≥ 0.005, **or** SP-SR or MP-SR strictly improves.
   - Otherwise keep the existing packet.

### 5.6 Tier 2 pilots

**(a) Baseline-equivalent single-episode end-to-end (GPU).**

- **Steps:** `prepare-tier --tier 2`, native retarget, then G5 retarget validity. Then one 5,000,192-step fresh job on one median-length episode, then deterministic evaluation.
- **Record:** compile time, steady steps/s, evaluation time, peak VRAM and host RSS.
- **Pass:**
  - G5 and G6 pass.
  - The rollout is valid for the packer.
  - The own-reference diagnostic (scored against the Tier 2 input, `align=False`) is not catastrophically worse than the same episode's Tier 1 own-reference diagnostic (MPPE within 2×). This compares two of our own diagnostics, not clean truth.
- **Expand:** the remaining 29 episodes, including episode 16.

**(b) Noise characterisation** (§4.8, CPU, can run during (a)).

- **Decision rule:**
  - Pooled `white`: WP4 is low value; at most apply the existing noise-limited conditioning as a jitter-only arm.
  - `random_walk` or `mixed`: run (c).

**(c) Constraint-based conditioning (§4.9a), CPU.**

- **Proxies** (rule-safe):
  - resting-height standard deviation drops by ≥ 50%
  - stance-foot height standard deviation drops by ≥ 30%
  - contact-window hand–object offset variance does not increase
  - mean correction ≤ 3× the estimated drift scale
- **If they pass:** retarget plus a 3-episode training A/B against (a) on the same episodes. The own-reference diagnostics cannot show truth, so promotion needs a **second Tier 2 upload as the measurement**, only after the WP2 entry exists and only within quota (G9–G12).

**(d) Camera self-calibration (§4.9b).**

- **Run:** only if (c) passes and GPU time remains after WP2 and WP3.
- **Pass:** held-out mask IoU ≥ 0.6 median and reprojection consistency across even/odd splits.

### 5.7 Tier 3 feasibility (read-only first)

1. **World anchor (rules + inspect).**
   - §3.4 establishes [FACT] that a Tier 3 frame reference must be in `released_track2_world` and that scoring does not align initial poses.
   - Without a rule-permitted world anchor (Q9), an arbitrary-gauge reconstruction moves *every* scored frame by the gauge error, so Tier 3 can at best place on the board, not score well.
   - **Stop WP7 while Q9 is unresolved**, unless the user explicitly decides that a valid low-scoring entry is worth the GPU time.
   - Also confirm by reading `metric_code/track_2/*.py` L200–260 that no other alignment step applies to Tier 3.
2. **If Q9 permits an anchor**, do a one-episode reconstruction. The released ground plane alone fixes only the up axis and height; yaw and horizontal translation still need a permitted anchor.
   - camera from video (§4.9b without the Tier 2 prior)
   - object via FoundationPose with the known mesh
   - human via SAM3D-Body MHR output, converted to SOMA-X by fitting SOMA-X joints to MHR joints [PH: SOMA-X forward-model availability in `robotic_grounding`]
   - then anchor registration, retarget validity (G5) and a frame-reference NPZ that passes `input_frame_reference` validation

   Record GPU-hours per episode. **Continue only if 30 × that cost fits the §6 budget after WP1–WP3.**

### 5.8 Track 3 dev pilot (WP6)

- **Prerequisite:** measure Sharpa throughput with one short dev job (≤ 1,048,576 steps; it must be divisible by 1,024 and cover the 11 curriculum stages × 1,024 worlds).
- **Then:** one `dev-jobs` decomposition (16 jobs × 10,000,384 steps) **only if** the measured time ≤ 25% of the remaining GPU budget.
- **Otherwise:** a reduced 2-arm comparison (baseline vs `object_focused` reward profile) on 2 dev episodes at the same budget.
- **Minimum gain:** ≥ 15% paired improvement on dev MPPE or RPE without SP-SR regression (the runbook's mesh-method threshold, reused).

---

## 6. Measured scheduling

### 6.1 Resources and concurrency

- **GPU:** one RTX 5090 Laptop, 24,463 MiB, with a 21,000 MiB device-wide guard and the shared lock `reconstruction/data/track3/pilot/episode_000038/pilot.lock`.
  - Past G1 policy runs used about 12.3–12.7 GiB [FACT, handoff §3], so **two concurrent policy jobs would exceed the guard. Run them one at a time.**
  - Short GPU tasks (mask renders, FoundationPose pilots) run only between policy jobs, under the same lock.
- **CPU work runs alongside GPU training:** MHR export and parity, evidence export (CPU decode), surrogate study, Tier 2 noise characterisation, tests.
  - [PH] Measure with `nproc` and `free -g`. Cap CPU containers at `--cpus 4 --memory 8g`, as the handoff tests do, so training keeps its host threads.
- **Disk** [PH: `df -h /home/alien51`]:
  - Tier 2 download: [PH: sum of the HF tree sizes]
  - each policy cohort: [PH: measure `du -sh` of one Tier 1 episode directory and multiply by 30]
  - Keep ≥ 50 GB free; refuse to start a cohort below that.

### 6.2 Throughput measurement (required before forecasting)

Every pilot job records the following in its `execution.json`:

- `compile_s`: launch to first training step logged
- `steps_per_s`: steady state, median over the last 50% of log intervals
- `eval_s`: deterministic evaluation duration
- `peak_vram_mib`: from 2 s `nvidia-smi --query-gpu=memory.used --format=csv` polling; the Track 3 pilot already writes `gpu.csv` this way
- `peak_host_rss_gib`

**Forecast:** `job_h = (compile_s + steps/steps_per_s + eval_s)/3600`; `cohort_h = Σ job_h × 1.15` (15% contingency for restarts and lock waits). The ~30 min per episode Tier 1 figure (§1.2) is only a starting assumption until the WP2 pilot measures it.

### 6.3 Queue (GPU, in order; CPU work in parallel)

| Order | GPU job | Gate before start | Expected size |
|---:|---|---|---|
| 1 | WP1 determinism pilot (3 episodes, evaluation only) | G1 | Evaluation minutes [PH measure] |
| 2 | WP2 Tier 2 single-episode pilot (train + evaluation) | G1, G5 on all 30 episodes | 1 job |
| 3 | WP1 full missing-rollout evaluations | Determinism passed | [PH: count missing from inventory] × `eval_s` |
| 4 | WP2 remaining 29 Tier 2 episodes | Pilot passed G6 | 29 jobs, measured |
| 5 | WP3 mask renders / reprojection checks | G2 passed (for hands), G3 | Short |
| 6 | WP5 or WP6 pilots, by remaining budget | §5.5 result, §5.8 throughput | Measured |
| 7 | WP4 (c) A/B, 3 episodes | §5.6(c) proxies passed | 3 jobs |

**Deadline buffers:**

- The last training job **finishes** by freeze − 48 h.
- All packing and official dry runs finish by freeze − 24 h.
- Uploads happen by freeze − 12 h, leaving time to poll receipts.
- At D−5, stop starting any WP whose measured cost would end after freeze − 48 h.

### 6.4 Heartbeat, interruption and resume

- **Heartbeat review every 2 h** while GPU jobs run. Check:
  - the last log step is advancing
  - VRAM is under the guard
  - no `STOPPED_FOR_REVIEW` state
  - disk free space
- **Resume:**
  - `jobs` re-runs are idempotent. Completed stages are skipped by checkpoint SHA.
  - An interrupted job resumes from its newest `state_*` **only** when the D1 identity matches.
  - Evaluations are keyed by checkpoint SHA and re-used if the trajectory SHA matches.
- **WSL or Docker restart:** verify the lock file is free (no live PID), then re-invoke the same `jobs --execute` command. Never start a second runner.

---

## 7. Explicit gates

A failing gate leaves the candidate unpromoted. Rollback means "keep the parent packet". Nothing is deleted or overwritten.

| Gate | Pass criteria | On failure | Blocks |
|---|---|---|---|
| **G1 Code** | All existing plus new tests pass; `git diff --check` clean; producing commit pushed; working tree clean except data | Fix before any GPU job | Everything |
| **G2 MHR parity** | ≤ 1e-5 m on ≥ 100 real v8 frames (vertices and joints), all decoders; units and axes agree | Human refit blocked; the object fit can still use the TorchScript hand path | WP3b, CD-H/PEN diagnostics with the NPZ asset |
| **G3 Track 1 evidence** | §3.2 checks: index alignment, reprojection IoU equals the recorded parent IoU (±0.01 median), hands-in-human-mask ≥ 95% | Episode excluded from WP3 | That episode's candidate |
| **G4 Track 1 candidate** | Existing `gate()` checks; D6 λ rule; packed neutrality (human bytes unchanged unless a separate human review passed); `assemble-track1` identity and anchor checks; official packer accepts | Parent v8 episode stays | Packet promotion (default ≥ 25 promoted, `--minimum-promoted`) |
| **G5 Retarget validity** (before any training) | Existing native retarget verification passes for **all 30** episodes including 16; no NaN; frame count equals source; joint limits respected; visible-frame coverage unchanged; support stage present | No training for that tier | WP2, WP4, WP7 |
| **G6 Job integrity** | D1 identity; target steps reached; actor checkpoint present; resolved config shows the intended reward profile and collision flags | Mark `STOPPED_FOR_REVIEW` | Evaluation and selection |
| **G7 Evaluation determinism** | Same checkpoint evaluated twice gives an identical trajectory SHA, or metric difference ≤ 1e-9 | Per-episode selection disabled; use final checkpoints | WP1 selection |
| **G8 Track 3 separation** | Dev IDs ∩ evaluation IDs = ∅; no dev scan in evaluation mesh directories (hash scan); `split: development` on every dev record | Abort the dev campaign | WP6 promotion |
| **G9 Rules** | Q1/Q2 recorded with quote, URL and capture date; other Q defaults noted | Upload blocked | All uploads |
| **G10 Snapshot** | Authenticated, ≤ 24 h old, `history_complete`, includes pending and failed attempts | Re-snapshot | Uploads |
| **G11 Packet** | 30/30 (or all scored) episodes; official `--dry-run` passes; producing commit SHA recorded and pushed; packet SHA in manifest | No upload | Uploads |
| **G12 Human approval** | The user explicitly approves the exact packet SHA and track; no existing manifest without a receipt | No upload | Uploads |

**Blocked while an input is absent:**

- **No participant MHR NPZ:** human refit and NPZ-based diagnostics are blocked.
- **No Tier 2 data:** WP2 and WP4 are blocked.
- **No Q9 answer:** WP7 is blocked.
- **No D4 fix:** snapshot and uploads are blocked.
- **No public dev references:** WP6 promotion is blocked.

---

## 8. Selection and packing

- **Complete coverage:** every packet covers the full scored roster. Unchanged episodes come from the parent with byte-identical content, verified by SHA. No partial packets.
- **Track 2 Tier 1:**
  - `pack-selected --selection <selection.json> --dataset-root reconstruction/data/v2d_challenge --commit <full SHA> --output <new version>`.
  - Selection uses `policy.rank` = (SP-SR, MP-SR, AUC, −MPPE) on the **trajectory-derived** metrics, never evaluator JSON summaries (handoff §8.4 warning).
- **Track 2 Tier 2:**
  - Use **final checkpoints**. Per-episode selection is restricted to Tier 1 by design. Own-reference selection cannot be validated without clean truth (Q3).
  - Always pass `--frame-references robotic_grounding/data/track2/tier2/<version>/frame_references` (written by `tier-inventory`), so the packer converts with the same continuous, visibility-interpolated poses that were retargeted. The packer's default (the released Tier 2 folder) holds zero poses on invisible frames [FACT, HF README].
  - The packer's clean cross-check line is informational only and must never steer choices.
- **Track 1:** `assemble-track1 --parent <v8> --parent-packet <v8 packet> --reviews <reviews> --scored <scored json> --output <v11> --commit <SHA>`. Serialized-artifact validation: re-load the packed NPZ and GLB, re-run `verify_recenter` and the official metric functions on the packed bytes.
- **Track 3:**
  - The packet comes only from evaluation-episode policies; the mesh packet is separate (`track3 mesh` → CD-O board).
  - Development results justify a method; they never enter the evaluation packet.

---

## 9. Submission readiness

1. **Rules record (G9).** Store it in the rules JSON with quotes. While Q1 is unresolved, `submission.quota` refuses by design. That is correct; do not bypass it.
2. **Fresh snapshot (G10)** with the D4-fixed client. Read own best and own latest for each of the 23 slugs and the rank-1/3/5/10 targets.
3. **Producing code committed and pushed;** record the full SHA in the packet (G11).
4. **Official dry run** through `submission submit … ` without `--execute` (gate check), then with `--execute`, which runs `--dry-run` first and then uploads.
5. **Exclusive receipt recovery:**
   - Receipt exists with the matching packet SHA: **poll only**.
   - Manifest exists without a receipt: **stop and ask the user**. Never retry automatically.
6. **No automatic duplicate baseline upload.** A packet whose local metrics equal the current submission is not uploaded.
7. **Upload order**, after G9–G12, each needing user approval:
   1. Tier 2 entry (fills empty boards)
   2. Tier 1 re-selection (if §5.5 minimum gain is met)
   3. Track 1 v11 (if G4 passes on ≥ 25 episodes)
   4. Track 3 (if §5.8 gains are met)

   One upload per track per decision; re-snapshot before each.

---

## 10. Exact first steps on the execution laptop

Run from `/home/alien51/video_to_data`. Labels: **[INSPECT]** is read-only, **[MODIFY]** edits code or data, **[EVALUATE]** uses the GPU for evaluation, **[TRAIN]**, **[PACK]**, **[UPLOAD]**. `<…>` marks a placeholder to resolve locally.

1. **[INSPECT] Sync and baseline the tests.**
   ```bash
   cd /home/alien51/video_to_data
   git fetch origin && git status --short | head
   git checkout v2d-track1-blackwell && git pull --ff-only
   git log -1 --format=%H
   reconstruction/.venv/bin/python -m pytest -q reconstruction/scripts/test_cross_track_improvement.py
   ```
2. **[INSPECT] Resources.**
   ```bash
   nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
   nproc; free -g; df -h /home/alien51
   docker images --format '{{.Repository}}:{{.Tag}}' | grep -E 'v2d_cari4d|flash_chord|v2d_track3|v2d_sam2'
   ls -l reconstruction/data/track3/pilot/episode_000038/pilot.lock 2>/dev/null
   ```
3. **[INSPECT] Kaggle client signatures (for D4).**
   ```bash
   reconstruction/.venv/bin/python - <<'PY'
   import inspect, importlib.metadata as m
   from kaggle.api.kaggle_api_extended import KaggleApi as K
   print(m.version('kaggle'))
   print(inspect.signature(K.competition_submissions))
   print(inspect.signature(K.competition_leaderboard_view))
   PY
   ```
4. **[INSPECT] MHR TorchScript inventory (for §3.1).** The container is read-only and writes to stdout.
   ```bash
   mkdir -p reconstruction/data/cross_track/mhr_asset_v1
   docker run --rm --cpus 4 --memory 8g -v "$PWD":/repo:ro -w /repo --entrypoint python v2d_cari4d:blackwell -c "
   import torch, json
   m = torch.jit.load('reconstruction/data/weights/track1_submission_mhr/mhr_model.pt', map_location='cpu')
   rows = [(n, list(t.shape), str(t.dtype), str(t.layout)) for n, t in list(m.named_buffers()) + list(m.named_parameters())]
   print(json.dumps({'tensors': rows, 'modules': [n for n, _ in m.named_modules()]}, indent=1))
   " > reconstruction/data/cross_track/mhr_asset_v1/torchscript_inventory.json
   docker run --rm --cpus 2 --memory 4g -v "$PWD":/repo:ro -w /repo --entrypoint python v2d_cari4d:blackwell -c "
   import torch; m = torch.jit.load('reconstruction/data/weights/track1_submission_mhr/mhr_model.pt', map_location='cpu'); print(m.code)
   " > reconstruction/data/cross_track/mhr_asset_v1/torchscript_code.txt
   sha256sum reconstruction/data/weights/track1_submission_mhr/mhr_model.pt checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt 2>/dev/null
   ```
5. **[INSPECT] Locate Tier 1 policy and result roots (for WP1).**
   ```bash
   find robotic_grounding/data/track2/tier1 -maxdepth 2 -type d | sort | head -60
   find robotic_grounding/data/track2/tier1 -name 'policy_*.safetensors' | wc -l
   ```
6. **[MODIFY] Implement D1–D5** (§4.1–4.4) with tests, run G1, then commit and push. Commit message: `Fix cross-track job identity, PEN constraint accumulation, Kaggle pagination and dev-jobs guard`.
7. **[INSPECT, network] Authenticated snapshot** (read-only API):
   ```bash
   reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py submission snapshot \
     --output reconstruction/data/cross_track/snapshots/$(date -u +%Y%m%dT%H%MZ)
   ```
8. **[INSPECT] Tier 1 checkpoint inventory.**
   ```bash
   reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py checkpoint-inventory \
     --policy-roots <policies_short root> <baseline root> \
     --existing-results <results_short root> \
     --output reconstruction/data/cross_track/tier1_selection_v1/candidates.json
   reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py checkpoint-select \
     --candidates reconstruction/data/cross_track/tier1_selection_v1/candidates.json \
     --reference-root reconstruction/data/v2d_challenge/track_2/tier_1_multiview_caption \
     --output reconstruction/data/cross_track/tier1_selection_v1/report
   ```
   Without `--run-evaluations` this reports missing rollouts and does not use the GPU. Then run the §5.5 determinism pilot, adding `--run-evaluations`: **[EVALUATE]**.
9. **[INSPECT, network] Tier 2 download** (data only, pinned revision):
   ```bash
   reconstruction/.venv/bin/python - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download('nvidia/video_to_data_challenge', repo_type='dataset',
                     revision='5f68335f3acc802033d1e80728c1633197521de8',
                     allow_patterns=['track_2/README.md', 'track_2/tier_2_synthetic_noise/*'],
                     local_dir='reconstruction/data/v2d_challenge', max_workers=4)
   PY
   find reconstruction/data/v2d_challenge/track_2/tier_2_synthetic_noise -type f | wc -l
   ```
10. **[MODIFY data] Prepare Tier 2 sources** (new version root; all 30 episodes by default, including 16). `--dataset` must end with the configured `input_path` `track_2/tier_2_synthetic_noise` (`tier_inputs.validate_root`).
    ```bash
    reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py prepare-tier \
      --tier 2 --dataset reconstruction/data/v2d_challenge/track_2/tier_2_synthetic_noise \
      --output robotic_grounding/data/track2/tier2/v1/source_episodes
    ```
    **[GPU, Docker] Retarget.** `retarget_track2_tier1.py` defaults `--episodes` to every episode **except 16**, so pass all 30 explicitly. Inspect why 16 was excluded for Tier 1 (`TRACK2_STATUS.md`, retarget reports) before trusting its Tier 2 output.
    ```bash
    reconstruction/.venv/bin/python robotic_grounding/scripts/retarget_track2_tier1.py \
      --tier 2 --data-root robotic_grounding/data/track2/tier2/v1 \
      --reports robotic_grounding/incident_reports/track2_tier2/retarget_v1 \
      --episodes $(seq 0 29)
    reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py tier-inventory \
      --data-root robotic_grounding/data/track2/tier2/v1 --tier 2 \
      --output robotic_grounding/incident_reports/track2_tier2/inventory_v1.json
    ```
    The inventory writes packer frame references to `robotic_grounding/data/track2/tier2/v1/frame_references/`. G5 must pass before step 11.
11. **[TRAIN] Tier 2 single-episode pilot.** Write `spec_tier2_pilot.json` following the runbook's typed example:
    - `"track": "track_2_tier2"`
    - `"target_steps": 5000192`
    - `"arm": "fresh"`
    - `"reward_profile": "baseline"`
    - one `episodes[]` item whose `reference` and `reference_sha256` are copied from that episode's row in `inventory_v1.json`, and a **new, empty** `output_dir` under `robotic_grounding/data/track2/tier2/policies_v1/episode_XXXXXX`

    Then:
    ```bash
    reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py plan-jobs \
      --specification spec_tier2_pilot.json --output robotic_grounding/incident_reports/track2_tier2/pilot_v1
    reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py jobs \
      --manifest robotic_grounding/incident_reports/track2_tier2/pilot_v1/jobs.json \
      --report robotic_grounding/incident_reports/track2_tier2/pilot_v1/run
    # Validated only; launch after D1 is merged and G5 passed:
    reconstruction/.venv/bin/python reconstruction/scripts/run_cross_track_improvement.py jobs \
      --manifest robotic_grounding/incident_reports/track2_tier2/pilot_v1/jobs.json \
      --report robotic_grounding/incident_reports/track2_tier2/pilot_v1/run --execute
    ```
    Record the §6.2 throughput fields, then re-forecast the 29-episode cohort.
12. **[MODIFY, CPU in parallel with 11]** Implement §4.6 (MHR exporter and parity), then §4.7 (evidence adapter), §4.8 (Tier 2 noise report) and §4.5 (GCV λ). Run §5.2, §5.4 and §5.6(b).
13. **[PACK] / [UPLOAD]** happen only after the §7 gates, and **[UPLOAD]** only with explicit user approval of the packet SHA.

**What to report back to the planning side** after steps 1–12: the D1–D5 commit SHA; the Kaggle signatures; the TorchScript inventory and parity numbers; the WP1 missing-rollout count and determinism result; the WP2 pilot throughput fields and G5 result; the Tier 2 noise classification table; the §5.4 λ_cap; and the snapshot summary (own best and rank targets per slug).
