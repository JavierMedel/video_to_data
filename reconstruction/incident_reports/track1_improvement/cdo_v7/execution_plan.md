# All-episode depth and geometry refinement

1. Freeze v6 as control and audit all 30 episodes using 33 scored samples each: actual-mesh image overlap, common-pixel cached depth, motion, and artifact hashes.
2. Rank remaining failures and fit bounded mesh-scale, ray-depth and temporal translation candidates. Select on fitting frames and check disjoint frames. Keep human parameters exact.
3. Screen promising candidates using corrected hand-inside-object penetration and proximity. Reject regressions; validate every scored frame for promotions and inspect comparison images/videos.
4. Package all accepted candidates with retained v6 episodes, run official validation, submit one version, record all five scores, and commit/push code and reports.

Cached monocular depth is diagnostic, not hidden ground truth. Hoop-specific fitting remains confined to Episode 1. Full hand diagnostics are not exact official PEN because host role indices are unavailable.
