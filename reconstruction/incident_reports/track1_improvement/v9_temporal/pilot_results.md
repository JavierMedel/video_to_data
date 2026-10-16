# v9 temporal pilot results

State: **PILOT_REJECTED_NO_EXPANSION**. No new submission or changed package. Immutable v8 remains current.

Seven episodes × six lambdas × two robust modes × two streams =168 isolated fits;84 human candidates were decoded and compared to the actual parent. All60 parent pair hashes and the scored v8 Parquet hash match.

| Mode | Lambda | Human passes /7 (need6) | Worst human p99 max displacement cm | Object mean acceleration cm, unaligned | Worst object bound cm per unit alignment scale |
|---|---:|---:|---:|---:|---:|
| plain | 3 | 0/7 | 115.506 | 0.27607 | 1.647 |
| plain | 10 | 0/7 | 134.611 | 0.19108 | 1.930 |
| plain | 30 | 0/7 | 152.529 | 0.13508 | 2.568 |
| plain | 100 | 0/7 | 162.375 | 0.09306 | 3.664 |
| plain | 300 | 0/7 | 166.294 | 0.06744 | 5.116 |
| plain | 1000 | 0/7 | 169.787 | 0.04823 | 7.243 |
| robust | 3 | 0/7 | 151.128 | 0.20067 | 2.389 |
| robust | 10 | 0/7 | 159.068 | 0.14375 | 3.122 |
| robust | 30 | 0/7 | 163.291 | 0.10695 | 3.918 |
| robust | 100 | 0/7 | 167.313 | 0.07772 | 4.985 |
| robust | 300 | 0/7 | 169.606 | 0.05796 | 6.500 |
| robust | 1000 | 0/7 | 174.140 | 0.04151 | 9.017 |

## Decision

No tested human configuration passes even one pilot episode under the combined gain/displacement conditions. No object configuration simultaneously reaches 0.10 cm unaligned acceleration and a 2 cm per-unit-alignment bound on every episode. Parameter-space smoothing is not accepted merely because its motion proxy decreases.

The independent unwrap-only diagnostic changes decoded vertices by less than 0.0002 cm across the pilot: the observed large deformations come from smoothing these parameter trajectories, not the periodic branch normalization alone. There is no claim about hidden-reference score improvement.

## Deferred checks

Full-scored-frame contact, image, video promotion and anchor-repair expansion were not launched for candidates already failing necessary displacement/motion gates. A sampled changed-human checker smoke test and two decoded four-frame diagnostic clips (full scene and object crop) verify the upgraded tools; they do not promote those candidates. No actual official package is built without complete passing selection evidence.

## Geometry precheck

The independent all30 v8 geometry audit is in `../v10_geometry/risk_audit.md`. Hoops 0, 1, 2 already share exactly the same submitted GLB hash, scale and extents; Episodes 0 and 2 IoU are 0.7288 and 0.6721. The proposed missing-metric-hoop premise is false for the current artifacts, so its replacement experiment is skipped. Low-IoU rollers and remaining iron rotations remain diagnostic leads. Disconnected mesh components alone are not evidence of removable floaters.

The initial full-clip diagnostic render was stopped after80 logged frames because the candidate had failed promotion. Its partial temporary files remain local and are not completed-video evidence. The short diagnostic clips record DIAGNOSTIC_PASS and full_video_review:false. Only their frame2 full-scene preview was visually inspected; this is tool verification, not candidate acceptance.
