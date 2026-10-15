# Episode 1 local review

Three v4 hoop candidates were reassessed against v5. All failed the corrected non-increasing hand penetration screen. A constant depth correction selected on 22 frames failed 11 held-out frames; it was rejected.

The final proposal starts from the image-fitted circular hoop geometry. It keeps the metric mesh, scale and rotations, and changes translation along camera rays within a fixed +/-18 cm search bound. Actual offsets are -9.06 to +8.19 cm. A conservative tube envelope around the actual mesh constrains full predicted hands on every video frame. Dynamic programming followed by bounded least squares penalizes offset changes and 3-D acceleration. This is direct geometric fitting, not a learned model or a held-out generalization claim. No hidden ground truth is used.

Actual-mesh generalized-winding and triangle-distance checks on all 174 scored frames find zero penetration for both full 4636-vertex anatomical hands and local FPS512, for control and candidate. Image overlap is evaluated on all 174 frames: mean IoU 0.629370. The old control mean on the original 33-frame screen was 0.000545. Local motion proxies improve: translation acceleration 1.84232 -> 0.49368 cm and surface acceleration 3.21777 -> 0.58338 cm; rotation jumps over 45 degrees fall from nine to zero. These are diagnostics, not official scores.

The control hoop is far from predicted hands, so there are no control-contact-like frames. The proximity guard is therefore inapplicable; it does not establish grasp accuracy. Exact host hand-role IDs and hidden alignment scale are unavailable. Human pose, shape and scale arrays are unchanged.

Six mask-overlay comparisons inspected: frames 151, 240, 325, 373, 426 and 461. The new ring follows observed hoop size and location, while the old ring is much too small and displaced. Also inspected the exact reconstruction close-up at frame 334. Both full videos decoded successfully: 668 frames at 30 fps each. Visual review accepted with the stated contact limitations. Five temporal-selection tests and syntax/diff checks passed. Official package validation and all five leaderboard scores are complete; see submission_results.md.
