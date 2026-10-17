# v10 contact-constrained rotation review

State: LOCAL_VALIDATION_COMPLETE. Promoted: [18]. Pending: [].

| Episode | Decision | Mean IoU gain | Reasons |
|---:|---|---:|---|
| 7 | RETAIN_V8 | 0.01151 | mean_iou_gain_at_least_002; independent_sampled_hand_check; visual_review |
| 18 | PROMOTE | 0.02304 | Six object crops and two full-scene source/actual-mesh comparisons inspected; small orientation changes remain comparable with modest outline gains and no new gross displacement. Inherited geometry mismatch remains. Numerical image/depth/contact/motion gates independently required. |
| 20 | RETAIN_V8 | 0.01539 | mean_iou_gain_at_least_002; independent_sampled_hand_check; visual_review |

All unpromoted episodes retain the exact v8 pair. Whole-package human/translation/scale/mesh arrays are unchanged. A candidate is not eligible merely because the general image-review tool says PASS: this campaign additionally requires0.02 mean IoU gain, framewise contact preservation, a motion guard, independent sampled hand review and explicit visual evidence.
