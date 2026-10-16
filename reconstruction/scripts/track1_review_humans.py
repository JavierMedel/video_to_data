"""Decode the actual bundle for each review panel, including changed humans."""
import numpy as np


def humans_identical(before, after):
    return all(np.array_equal(before[k], after[k]) for k in ('pose', 'scales', 'shape'))


def decode_bundle_human(model, bundle, frames, tensor):
    """tensor adapter keeps this helper independently testable without Torch."""
    return model.run(tensor(bundle['pose'][frames]),
                     tensor(np.r_[bundle['scales'], bundle['shape']][None]))


def framewise_contact_pass(control, candidate, control_gap, candidate_gap, tolerance_cm=1e-8):
    """A better episode average cannot conceal a worse individual contact frame."""
    full_pass = True
    for key in ('per_frame_full_hand_depth_cm', 'per_frame_fps512_depth_cm'):
        before, after = np.asarray(control[key]), np.asarray(candidate[key])
        if before.shape != after.shape or before.ndim != 1 or not np.isfinite([before, after]).all():
            raise ValueError('Invalid per-frame contact arrays')
        full_pass &= bool(np.all(after <= before+tolerance_cm))
    control_gap, candidate_gap = np.asarray(control_gap), np.asarray(candidate_gap)
    if control_gap.shape != before.shape or candidate_gap.shape != before.shape or not np.isfinite([control_gap,candidate_gap]).all():
        raise ValueError('Invalid gap arrays')
    active = control_gap <= 1.5
    gap_pass = bool(np.all(candidate_gap[active] <= control_gap[active]+.5+tolerance_cm))
    return full_pass, gap_pass
