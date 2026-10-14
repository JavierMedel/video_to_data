# Fourth Track 1 submission review

Local diagnostics only; lower motion/penetration is better, higher IoU is better. Official CD-O remains unmeasured until upload.

| Episode | Selection | IoU v3 → candidate | Translation acceleration cm | Surface acceleration cm | Signed penetration cm |
|---|---|---|---|---|---|
| 1 | RETAIN_V3 | 0.0005 → 0.5480 | 1.842 → 0.560 | 3.218 → 0.717 | 0.00000 → 0.04666 |
| 2 | PROMOTE | 0.0113 → 0.6880 | 2.097 → 0.795 | 3.597 → 0.917 | 0.00000 → 0.00091 |
| 12 | PROMOTE | 0.0658 → 0.7301 | 4.195 → 1.192 | 4.255 → 2.759 | 0.00000 → 0.00448 |
| 14 | PROMOTE | 0.0682 → 0.7435 | 3.609 → 1.087 | 3.781 → 2.509 | 0.00000 → 0.00015 |
| 19 | PROMOTE | 0.4524 → 0.4557 | 1.077 → 0.415 | 7.505 → 1.841 | 0.03455 → 0.00025 |
| 20 | PROMOTE | 0.4254 → 0.4333 | 0.887 → 0.472 | 3.255 → 1.523 | 0.09762 → 0.06687 |

All 30 human parameter arrays must match v3 exactly when packing. The 25 unselected episode pairs reuse v3.

- Episode 1: Retain v3. Metric-ring fitting and body-aware ambiguity selection improve silhouette and motion, but signed penetration remains 0.04666 cm, above the 0.01 cm gate. Reviewed overlays also retain ambiguity during stepping through the hoop.
- Episode 2: Promote the correct hoop mesh with continuously transported plane orientation. Reviewed multi-frame overlays support much better alignment, zero large rotation jumps and lower surface motion. Signed penetration passes. Unsigned median hand distance worsens from 16.77 to 33.97 cm and remains an unresolved 3D diagnostic; no hidden-reference score is inferred from it.
- Episode 12: Promote the mesh-scored temporal pose sequence. Reviewed overlays and exact NPZ/GLB reconstruction video show improved pan placement; surface and translation motion improve, signed penetration passes and human parameters match v3 exactly. Reject the held-out depth adjustment.
- Episode 14: Promote the mesh-scored temporal sequence with the held-out -0.05m correction. Reviewed overlays retain the pan handle and improve alignment; motion improves and signed penetration passes.
- Episode 19: Promote the smoother mesh-scored orientation sequence and held-out -0.10m correction. Reviewed overlays preserve roller alignment while signed penetration and motion proxies improve.
- Episode 20: Promote the temporal sequence with the fitted -0.08m ray correction. Exact final artifact overlays preserve/improve alignment; both held-out and overall signed penetration improve, with lower motion and comparable hand proximity.

Full-clip/300-step/rotation-only refinement tests were completed and not selected; they retain 71–72 large rotation jumps on Episode 12.

Video: `data/v2d_challenge/track_1/cdo_v4_experiments/review_videos/episode_000012/episode_000012_v3_vs_v4.mp4` (405 frames, 30fps, 1536x656). Matching close-ups: `episode_000012_v3_vs_v4_object_closeups.mp4` (405 frames, 30fps, 1024x464).

## Official result

| Metric | Previous v3 | New v4 | Change | Submission ID |
|---|---:|---:|---:|---:|
| cd_h_cm | 15.43955 | 15.43955 | +0.00% | 57196175 |
| cd_o_cm | 34.15571 | 28.45924 | -16.68% | 57196176 |
| acc_h_cm | 1.15894 | 1.15894 | +0.00% | 57196178 |
| acc_o_cm | 1.10054 | 0.88239 | -19.82% | 57196179 |
| interpenetration_cm | 0.00318 | 0.00331 | +4.09% | 57196182 |

CD-O and ACC-O improve; human scores are unchanged; official penetration worsens slightly despite the local proxy gate. See `submission_results.md`.
