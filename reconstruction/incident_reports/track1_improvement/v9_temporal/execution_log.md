# v9 temporal / v10 audit execution log

User authorized correcting and implementing `V2D_NEXT_IMPROVEMENT_PLAN.md`. This execution implements the local temporal experiment and decision gates; no new upload is made.

## Reconciliation

- Parent is complete submitted v8; Parquet SHA256 `bd73a931e9f9ec85b920884415c5dc2123d86e2d3d91395c7820ebf564fd3198`. All60 NPZ/GLB hashes match its committed validation manifest before and after the pilot.
- Docker Desktop was stopped at entry. The existing desktop application was started; Docker29.8.2 and the existing Blackwell image became available. No host ML packages were installed and no image was rebuilt.
- Host lacks graph engines for trimesh connected components. The read-only geometry audit therefore ran in the existing container with its existing graph dependencies.
- Full-video arrays and `valid_input` are read directly from each parent. All candidate metadata and non-target arrays remain identical; provenance lives in sidecars instead of mutating the NPZ report field.
- The model exposes249 parameter names. Root translation parameters0:3 use cm. Pose3:130 are angular-named/scaled angular controls, but only112 passed the numerical +2pi test at the probe pose. Internal length controls130:136 are not unwrapped. The complete decoded unwrap-only equivalence check over all seven pilot clips stays below0.0002 cm maximum vertex movement. Model/source fingerprints are recorded.
- ACC-H joints come from the released `MHR_TABLE3_BODY_JOINT_INDICES`. Diagnostics reproduce contiguous stretches, repeated endpoint acceleration values and equal weighting of episodes; all magnitudes remain unaligned local proxies, not official errors.

## Implemented tools

- NumPy banded Whittaker fit, exact pin elimination, finite robust weighting and sidecar parameter/input hashes.
- Isolated plain/robust human/object candidates for seven pilot episodes and six strengths each, plus strict unchanged-field and first-scored-row checks.
- GPU decoded human displacement/motion reports; explicit per-unit-alignment conservative distance bounds.
- Generic parent-based temporal package validator with complete-selection guard, field/packed-row whitelists, actual first-scored-row checks against the parent Parquet, genuine-code provenance requirements, no upload capability.
- Hand checker can explicitly decode each candidate's own human, with a strict per-frame full-hand/FPS512 non-increase option and <=0.5 cm parent-contact-like gap increase. Cache reuse binds checker/helper/decoder-model/eager-evaluator/hand-spec source fingerprints.
- Renderer explicitly decodes each panel's actual human. Its short diagnostic option is separately labeled and cannot be reported as a full-video check.
- Parent-hash-bound all30 geometry audit with cached actual-v8 image checks, welded connected-component diagnostics, extents, centroid offsets and gap-safe rotation jumps.

## Actual commands and results

From workspace root, host fits:

```bash
reconstruction/.venv/bin/python reconstruction/scripts/track1_v9_temporal.py --stream human
reconstruction/.venv/bin/python reconstruction/scripts/track1_v9_temporal.py --stream object
reconstruction/.venv/bin/python reconstruction/scripts/track1_v9_temporal.py --stream human --robust
reconstruction/.venv/bin/python reconstruction/scripts/track1_v9_temporal.py --stream object --robust
```

GPU diagnostics use the existing `v2d_cari4d:blackwell` image, reconstruction mounted at `/repo`, user1000:1000, `LOGNAME=uid1000`, `HOME=/tmp`. The model probe uses `track1_v9_motion_report.py --probe`; motion diagnostics run once plain and once with `--robust`. Seven episodes × six strengths × two modes × two streams gives168 fits and84 decoded human candidates. The fit runner's hash-checked resume was exercised successfully on completed plain human outputs.

`track1_v9_pilot_review.py` completes with `PILOT_REJECTED_NO_EXPANSION`. Neither stream meets necessary gates. No lambda is frozen for expansion. No all30 temporal campaign, accepted v9 package, official preflight/dry-run, quota use or new leaderboard version is claimed.

The changed-human hand checker completed a sampled three-frame Episode1 smoke test using the actual candidate pose. Both means were zero on those samples; that does not establish full-clip interaction quality. The renderer decoded both short diagnostic clips (four frames each) and correctly records changed human arrays, each panel's actual geometry and `full_video_review:false`. A started full render was stopped after80 logged frames once the candidate was rejected; partial temporary files remain preserved and are not passed video evidence.

## Expansion/geometry decisions

All30 v8 meshes were audited, with image evidence hash-matched to the exact submitted pair. Hoops0,1,2 have identical GLB hashes and scale/extents. Episodes0 and2 have IoU0.7288 and0.6721, so the proposed missing-metric-hoop replacement is skipped under its own precheck. Label-group extents do not exceed the plan's15% inconsistency flag (the largest group ratio is approximately1.111). Mesh connected components are not automatically floaters: removal requires projected visibility/occlusion evidence that this audit does not provide. No component is removed, and no unverified cleanup is staged. Low-IoU rollers and iron rotations remain the supported next diagnostic directions.

The optional v11 rotation experiment is not launched: its stated preceding geometry evidence/promotion gates are not established. The current outcome is retained v8 plus reproducible rejected-pilot evidence, not an invented improvement.
