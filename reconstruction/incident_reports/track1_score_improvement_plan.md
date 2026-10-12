# Track 1 score audit and improvement priorities

## Baseline

First authorized version scored COMPLETE on five competitions: CD-H 15.11851 cm, CD-O 152.72099 cm, ACC-H 1.62217, ACC-O 7.10147, PEN 0.00213 cm. No further upload is authorized.

## Verified findings

- Read-only audit covered all 30 NPZ/GLB pairs and their saved reconstruction bundles. Submitted object translations match each saved refined bundle. No translation packing mismatch found.
- Official human converter mean vertex errors range from 0.4566 to 1.3829 mm. These local conversion errors are much smaller than the 15.12 cm held-out CD-H score. Improving the converter is a lower priority.
- Object mask agreement is poor: median across episodes of sampled projected bounding-box IoU is 0.118. Episodes 1, 5, 6, 14, 18 and 23 have zero median overlap. This uses nine or fewer scored-frame samples per episode and projected mesh bounding boxes; it is a diagnostic, not exact silhouette IoU or a held-out metric.
- Episode 6 (iron): submitted object depth ranges 2.86–27.28 m while predicted human root depth ranges 3.14–3.74 m. Median root-to-object distance is 10.49 m. Initial tracker object depth reaches 29.28 m.
- Episode 28 (laptop cart): initial depth reaches 13.54 m; submitted depth reaches 8.39 m while human depth ranges 3.19–3.65 m.
- Episodes 6, 18 and 28 report FoundationPose reliability fraction 1.0, but tracker selected-IoU medians are respectively 0, 0 and 0.0113. Existing reliability flags cannot be trusted as image-agreement certificates.
- Episode 18 raw network prediction has negative object depth in some frames. This is the raw prediction, not a claim that the submitted refined pose has negative depth.
- All-video-frame object translation second-difference magnitudes increase from 4.35 to 33.23 cm for Episode 6 and from 3.11 to 14.17 cm for Episode 28 when comparing tracker initialization to submitted predictions. These are temporal diagnostic proxies, not Kaggle acceleration errors against hidden reference trajectories.
- The speed-oriented first-entry recovery used one requested refinement step and a batch size of 64 for newly resumed episodes. This is substantially less refinement than the pipeline default of 300 steps.
- A very small penetration score does not establish accurate contact: separated human and object surfaces can also produce nearly zero penetration.

## Recommended order

1. **Repair object initialization on a small pilot: Episodes 6, 18 and 28.** Diagnose failure frames using overlays, object masks, camera intrinsics and human-aligned depth. Calibrate mesh scale from observed image/depth evidence. Reject behind-camera, inconsistent-depth and poorly reprojecting candidates; reacquire after failure rather than declaring them reliable. Do not simply replace all predictions with raw tracker poses: the tracker itself has catastrophic failures.
2. **Refine only after initialization is plausible.** Run enough silhouette/contact/temporal optimization iterations to establish convergence, verify sampled image alignment improves, and preserve all baseline artifacts in separate candidate directories. Increasing refinement alone may not recover a 29 m initialization under strong priors.
3. **Handle trajectory outliers and jitter.** Use image agreement and tracking confidence to identify bad frames; use motion-aware interpolation/smoothing and rotation-aware processing. Validate observed motion so real movement is preserved. Compare acceleration proxies and mask alignment before choosing a candidate.
4. **Improve human tracking.** Inspect human overlays and depth/root consistency in failed frames. Reduce pose/root jitter and detector failures. Keep shared identity and evaluate converter error independently from reconstruction error.
5. **Validate a candidate before another upload.** Use the official packer and dry-run, compare local evidence against the immutable first-entry baseline, and request explicit authorization for a second leaderboard version. Hidden held-out scores cannot be predicted from these local proxies.

## Constraints

Episode 16 remains protected from reruns without explicit approval. Original submission arrays, meshes, checkpoints and the scored Parquet remain preserved. No GPU reconstruction rerun or new submission was performed for this audit.

Detailed diagnostics: `incident_reports/track1_score_audit.json`. Audit implementation: `scripts/track1_score_audit.py`. Scored-frame metadata was derived from the official sample, not held-out labels.
