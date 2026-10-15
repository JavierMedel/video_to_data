# Episode 7 ablation findings

The fixed 2x2 ablation is complete. All four branches were reviewed on every one of the 597 scored frames. Motion diagnostics cover all 777 video frames. Five focused tests passed. Human arrays, object rotations, scale and mesh remain exactly equal to submitted v6. Combined translations reproduce submitted v7 within 1 micrometre.

| Branch | Translation acceleration proxy (cm) | Visible IoU | Cached depth error (cm) | Full-hand penetration (cm) | Penetration / proximity gates |
|---|---:|---:|---:|---:|---|
| v6 control | 1.765503 | 0.705783 | 9.549499 | 0.0017015244 | pass / pass |
| Smoothing only | 0.611534 | 0.705158 | 9.552169 | 0.0017581934 | fail / pass |
| Depth only | 1.773966 | 0.707713 | 5.823755 | 0.0012849410 | pass / pass |
| Combined | 0.615009 | 0.706794 | 5.687881 | 0.0012797692 | pass / pass |

All changed branches pass the image alignment, meaningful-gain and depth checks. The unchanged control has no meaningful gain by definition. Depth numbers use common visible pixels for each candidate against v6; support differs between pairs. Full-hand penetration is an unaligned diagnostic, not official PEN. The separate FPS512 penetration check also rejects smoothing alone and passes the two depth-shifted branches.

Smoothing alone reduces the translation acceleration proxy by about 65.4%, with essentially unchanged image/depth agreement. It increases mean full-hand penetration by about 3.3%, so it is not eligible for promotion under the current guard. Its grasp-proximity check passes.

The +0.07 m ray shift accounts for most of the improvement against cached monocular depth. It also lowers the local penetration diagnostic, allowing the combined branch to pass where smoothing alone fails. Depth-only slightly worsens translation and surface acceleration. These observations separate the local effects; they do not establish which component caused v7's official CD-O regression.

Six matching source-video mask frames per changed branch were visually inspected. Gross projected alignment remains comparable, and differences are subtle at that scale. These overlays cannot establish accurate depth or grasp. See `visual_review.json` for image fingerprints and scope.

Keep submitted v6 as the reconstruction control. No branch was promoted or uploaded. A next local experiment should constrain smoothing by hand interaction during fitting rather than moving the entire object deeper to make a smoothed candidate pass. Object rotations still contain the control's 48 jumps above 45 degrees; they were deliberately held fixed in this experiment. New official controlled scores would be necessary to attribute the hidden-reference reconstruction regression.

Detailed gate outputs and artifact hashes are in `comparison.json`; the reproducible command is in `execution_plan.md`. Cached hand control reuse verifies artifact hashes, frame lists, evaluation scope and hand/sample counts. Vertex-level penetration deltas are unavailable when using the scalar control cache and are reported as null.
