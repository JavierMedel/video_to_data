# Fifth Track 1 iteration

1. Audit all 24 remaining episodes against immutable scored v4 and rank rotation failures. Complete: selected 23, 18, 21, 16, 13.
2. Diagnose prior penetration regression. Complete: official PEN checks hand vertices inside the object; the v4 gate checked the opposite containment direction. Implement generalized-winding-number hand penetration with the actual submitted human decoder. Keep body penetration as an additional diagnostic.
3. Generate isolated temporal pose/translation/depth candidates for the five selected episodes, preserving all human arrays. Running.
4. Compare old/new hand-inside-object depth, visible alignment, motion and visual evidence. State hand-role subset and alignment-scale limits explicitly. Select only supported changes.
5. Commit producing code, build a 30-episode package using v4 as control, verify original hashes and human identity, run official preflight and dry-run, and issue one controlled submission across all five metrics.
6. Save exact receipts and compare official results against v4: CD-H 15.43955, CD-O 28.45924, ACC-H 1.15894, ACC-O 0.88239, PEN 0.00331. Target baseline CD-O 23.16246 remains unmet.

All new artifacts use `cdo_v5_experiments`; v3/v4 data and packages are preserved.

The first hand-depth screen rejected four original candidates. Added a control-contact-like proximity check (nearest-vertex upper bound, not a grasp label), and fit ray corrections on 22 frames with a single check on 11 disjoint frames. Current corrected candidates: Episode 23 +0.03m, Episode 21 -0.03m. Episode 18 correction failed the held-out contact check and remains unpromoted pending final review. Final all-scored-frame checks are running.

All held-out contact-preserving probes are complete. Provisional candidates: 23 (+0.03m ray correction), 21 (-0.03m ray correction), 16 (temporal selection without the geometry/depth probe). Episodes 18 and 13 failed held-out/contact eligibility and retain v4. Every scored frame is now being checked for all three provisional candidates; promotion remains conditional.

Final selection: promote 16, 21, 23 after all scored-frame checks (240, 347, 399 frames respectively). Retain 13 and 18. Nine tests, official pack validation, authenticated preflight and quota dry-run passed. Producing code 1c3a1efca3c049c8cff229f25163254e518558d3 is pushed. A single v5 version was accepted on all five leaderboards; scores are pending in the exclusive receipt. Exact comparison videos contain 563 frames each at 30fps and are validated.

All five v5 submissions completed. Official scores: CD-H 15.43955, CD-O 28.14623, ACC-H 1.15894, ACC-O 0.86183, PEN 0.00325. All object metrics improve and human metrics are unchanged. No current-batch steps remain; see `submission_results.md` for IDs and comparison.
