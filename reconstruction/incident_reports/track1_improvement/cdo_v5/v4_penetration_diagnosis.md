# v4 penetration regression diagnosis

All five promoted v4 episodes increase the corrected hand-inside-object diagnostic on the 33 sampled scored frames. This was missed by the opposite-direction body proxy.

| Episode | v3 full-hand depth cm | v4 full-hand depth cm | Change |
|---|---:|---:|---:|
| 2 | 0.00000000 | 0.00141966 | +0.00141966 |
| 12 | 0.00000000 | 0.00038873 | +0.00038873 |
| 14 | 0.00000000 | 0.00014535 | +0.00014535 |
| 19 | 0.00000479 | 0.00083248 | +0.00082769 |
| 20 | 0.00100129 | 0.00192114 | +0.00091985 |

The unaligned full-hand diagnostic change, divided across the 30 equally weighted episodes, is +0.00012338 cm. The official observed change was +0.00013000 cm. The shared direction and similar magnitude support the containment-direction mismatch as an explanation. Different sampled frames, host hand IDs and hidden positive alignment scales prevent treating this as exact attribution.
