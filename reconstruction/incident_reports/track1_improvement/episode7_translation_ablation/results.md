# Episode 7 translation ablation

Control: submitted v6. Only object translations vary. All diagnostics below are local proxies.

| Branch | Ray shift (m) | Translation acceleration (cm) | Surface acceleration (cm) | Image IoU | Depth error (cm) | Hand / proximity |
|---|---:|---:|---:|---:|---:|---|
| control | +0.00 | 1.765503 | 3.466227 | 0.705783 | 9.549498558044434 | True / True |
| smoothing_only | +0.00 | 0.611534 | 2.665002 | 0.705158 | 9.552168846130371 | False / True |
| depth_only | +0.07 | 1.773966 | 3.470335 | 0.707713 | 5.823755264282227 | True / True |
| combined | +0.07 | 0.615009 | 2.666447 | 0.706794 | 5.687880516052246 | True / True |

State: LOCAL_VALIDATION_COMPLETE. Source hashes preserved. Combined reproduces v7 within 1 micrometre.

No candidate is automatically promoted. Visual review and any official score remain separate from these local checks.

The unchanged control intentionally fails the meaningful-gain screen. Hand checks reuse the hash-matched v6 control summary; pointwise penetration deltas are unavailable with this cache. Depth comparisons use common pixels separately for each branch versus v6, so their support can differ.
