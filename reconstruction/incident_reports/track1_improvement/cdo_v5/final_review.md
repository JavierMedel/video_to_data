# Fifth Track 1 iteration review

All comparisons use immutable scored v4 as control. Human parameters are preserved exactly. Local hand penetration uses the released containment direction, all anatomical hand vertices and a labelled local FPS512 selection; host vertex IDs and alignment scale remain unavailable.

| Episode | Decision | IoU v4 → candidate | Surface acceleration cm | Large rotation jumps | Penetration review |
|---|---|---|---|---|---|
| 23 | PROMOTE | 0.7628 → 0.8004 | 2.531 → 1.522 | 104 → 30 | All scored frames, full hand + FPS512 |
| 18 | RETAIN_V4 | 0.4121 → 0.4695 | 5.575 → 1.474 | 48 → 4 | Rejected in screening/held-out check |
| 21 | PROMOTE | 0.7253 → 0.7709 | 1.796 → 1.176 | 45 → 18 | All scored frames, full hand + FPS512 |
| 16 | PROMOTE | 0.7353 → 0.7370 | 3.655 → 1.536 | 28 → 6 | All scored frames, full hand + FPS512 |
| 13 | RETAIN_V4 | 0.6652 → 0.6969 | 3.478 → 1.675 | 33 → 2 | Rejected in screening/held-out check |

- Episode 23: Promote temporal object poses with the held-out +0.03m ray correction and 0.9 scale factor. Exact overlays support improved placement; rotation jumps decrease from 104 to 30. Full-hand and FPS512 penetration both decrease across all 399 scored frames, and the contact-like proximity guard passes.
- Episode 18: Retain v4. The initial candidate increases hand penetration; the temporal-only correction passes sampled penetration but fails the held-out contact-proximity check. Visual alignment improvement alone is insufficient for promotion.
- Episode 21: Promote temporal poses with the held-out -0.03m correction and 0.9 scale factor. Exact overlays improve alignment; large jumps decrease from 45 to 18. Both penetration checks improve over all 347 scored frames and proximity passes. The initial zero-offset candidate was rejected for creating a hand gap.
- Episode 16: Promote temporal poses and translation smoothing without the geometry/depth probe. Reviewed overlays preserve block alignment; jumps decrease from 28 to 6. Both penetration checks improve over all 240 scored frames and hand proximity is stable.
- Episode 13: Retain v4. Temporal correction improves silhouette and motion but its hand penetration increases; none of the bounded contact-preserving options meets all fitting/held-out conditions.

Four numerical parity tests pass against the released PEN implementation, including the actual dense meshes. Five temporal-selection tests pass. Source and evidence hashes are recorded in the decisions.

## Official results

| Metric | Previous v4 | New v5 | Change | Submission ID |
|---|---:|---:|---:|---:|
| cd_h_cm | 15.43955 | 15.43955 | +0.00% | 57197586 |
| cd_o_cm | 28.45924 | 28.14623 | -1.10% | 57197588 |
| acc_h_cm | 1.15894 | 1.15894 | +0.00% | 57197591 |
| acc_o_cm | 0.88239 | 0.86183 | -2.33% | 57197592 |
| interpenetration_cm | 0.00331 | 0.00325 | -1.81% | 57197593 |

All three object metrics improve; human scores are unchanged. Exact scores and limitations are in `submission_results.md`.
