"""Discrete, contact-constrained trajectory smoothing without GPU dependencies."""
import numpy as np


def scored_motion_diagnostics(rotations, translations, vertices, frames, max_vertices=1024):
    """Measure proxies only on consecutive original-video scored triples."""
    rotations, translations, vertices = [np.asarray(x, dtype=float) for x in (rotations, translations, vertices)]
    frames = np.asarray(frames)
    if rotations.shape != (len(translations), 3, 3) or translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError('Invalid trajectory shapes')
    if frames.ndim != 1 or frames.dtype.kind not in 'iu' or (np.diff(frames) <= 0).any() or (frames < 0).any() or (frames >= len(translations)).any():
        raise ValueError('Scored frames must be unique, increasing original indices')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or max_vertices <= 0 or any(not np.isfinite(x).all() for x in (rotations, translations, vertices)):
        raise ValueError('Invalid geometry')
    consecutive = np.diff(frames) == 1
    indices = frames[2:][consecutive[:-1] & consecutive[1:]] - 2
    result = {'consecutive_scored_triples': len(indices), 'translation_acceleration_cm_proxy': None,
              'sampled_vertex_acceleration_cm_proxy': None}
    if not len(indices): return result
    sampled = vertices[::max(1, int(np.ceil(len(vertices)/max_vertices)))]
    acceleration = np.diff(translations, n=2, axis=0)[indices]
    rotation_acceleration = np.einsum('tij,pj->tpi', np.diff(rotations, n=2, axis=0)[indices], sampled)
    result.update(translation_acceleration_cm_proxy=float(np.linalg.norm(acceleration, axis=1).mean()*100),
                  sampled_vertex_acceleration_cm_proxy=float(np.linalg.norm(rotation_acceleration+acceleration[:, None], axis=2).mean()*100),
                  sampled_vertex_count=len(sampled))
    return result


def summarize_selected_contacts(full, fps, proximity, done, path, frames):
    """Summarize exact precomputed frame/option checks after finite selection.

    Selection creates no new geometry. These are reused fit evaluations, not
    independent hand rerenders; callers must verify artifact/input hashes.
    """
    full, fps, proximity = [np.asarray(x, dtype=float) for x in (full, fps, proximity)]
    done, path, frames = np.asarray(done, dtype=bool), np.asarray(path), np.asarray(frames)
    if full.ndim != 2 or any(x.shape != full.shape for x in (fps, proximity, done)) or not done.all():
        raise ValueError('Contact evaluations must be complete and shape matched')
    if any(not np.isfinite(x).all() or (x < 0).any() for x in (full, fps, proximity)):
        raise ValueError('Invalid contact evaluations')
    if path.shape != (len(full),) or path.dtype.kind not in 'iu' or (path < 0).any() or (path >= full.shape[1]).any():
        raise ValueError('Invalid selected options')
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu' or (frames < 0).any() or (frames >= len(full)).any() or len(np.unique(frames)) != len(frames):
        raise ValueError('Invalid evaluation frames')
    summaries, gaps = {}, {}
    for label, choices in [('control', np.zeros(len(frames), dtype=int)), ('candidate', path[frames])]:
        summaries[label] = {'mean_full_hand_depth_cm_unaligned': float(full[frames, choices].mean()),
                            'mean_fps512_depth_cm_unaligned': float(fps[frames, choices].mean()),
                            'per_frame_full_hand_depth_cm': full[frames, choices].tolist(),
                            'max_hand_vertex_depth_cm': None, 'penetrating_hand_vertex_fraction': None}
        gaps[label] = proximity[frames, choices]
    active = gaps['control'] <= 1.5
    increase = gaps['candidate'][active] - gaps['control'][active]
    penetration_pass = all(summaries['candidate'][k] <= summaries['control'][k]+1e-8
                           for k in ('mean_full_hand_depth_cm_unaligned', 'mean_fps512_depth_cm_unaligned'))
    proximity_pass = not len(increase) or (np.median(increase) <= 1. and np.quantile(increase, .95) <= 2.)
    return {'summary': summaries, 'hand_penetration_gate_pass': bool(penetration_pass),
            'grasp_proximity_gate_pass': bool(proximity_pass),
            'grasp_proximity': {'control_contact_like_frames': int(active.sum()), 'control_threshold_cm': 1.5,
                'median_gap_increase_cm': float(np.median(increase)) if len(increase) else None,
                'p95_gap_increase_cm': float(np.quantile(increase, .95)) if len(increase) else None,
                'control_nearest_vertex_cm': gaps['control'].tolist(), 'candidate_nearest_vertex_cm': gaps['candidate'].tolist(),
                'scope': 'Unsigned nearest mesh-vertex proxy on local FPS512 points; not a grasp label.'}}


def feasible_options(options, full_depth, fps_depth, proximity, max_displacement_m=.03):
    """Control is always feasible; require framewise non-increasing penetration.

    Arrays have shape [frames, options, ...], with unchanged control at index 0.
    Proximity may grow at most 0.5 cm on control-contact-like frames.
    """
    options = np.asarray(options, dtype=np.float64)
    full_depth, fps_depth, proximity = [np.asarray(x, dtype=float) for x in (full_depth, fps_depth, proximity)]
    if options.ndim != 3 or options.shape[2] != 3 or not np.isfinite(options).all():
        raise ValueError('Expected finite T x K x 3 translations')
    if any(x.shape != options.shape[:2] or not np.isfinite(x).all() or (x < 0).any() for x in (full_depth, fps_depth, proximity)):
        raise ValueError('Expected finite nonnegative T x K hand diagnostics')
    if max_displacement_m <= 0:
        raise ValueError('Displacement bound must be positive')
    displacement = np.linalg.norm(options - options[:, :1], axis=2)
    allowed = (full_depth <= full_depth[:, :1] + 1e-10) & (fps_depth <= fps_depth[:, :1] + 1e-10)
    allowed &= (proximity[:, :1] > 1.5) | (proximity <= proximity[:, :1] + .5)
    allowed &= displacement <= max_displacement_m
    allowed &= options[:, :, 2] > .1
    allowed[:, 0] = True
    # Avoid changing the endpoints of this isolated finite video.
    allowed[:2, 1:] = False
    allowed[-2:, 1:] = False
    return allowed


def objective(options, rotations, centroid, path, fidelity_weight=.001):
    values = options[np.arange(len(options)), path]
    centers = np.einsum('tij,j->ti', rotations, centroid) + values
    return float(np.square(np.diff(centers, n=2, axis=0)).sum()
                 + fidelity_weight * np.square(values - options[:, 0]).sum())


def select_safe_path(options, allowed, rotations, centroid, fidelity_weight=.001):
    """Globally minimize squared surface acceleration over feasible options.

    With fixed rotations, the option-dependent part of mean squared surface
    acceleration equals the acceleration of the transformed mesh centroid.
    The remaining rotational variance is constant across paths. A two-state
    history makes this second-order objective solvable by dynamic programming.
    """
    options = np.asarray(options, dtype=np.float64)
    allowed = np.asarray(allowed, dtype=bool)
    rotations, centroid = np.asarray(rotations, dtype=float), np.asarray(centroid, dtype=float)
    if options.ndim != 3 or options.shape[2] != 3 or len(options) < 3 or not np.isfinite(options).all():
        raise ValueError('Expected at least three finite T x K x 3 translations')
    frames, count = options.shape[:2]
    if allowed.shape != (frames, count) or not allowed.any(axis=1).all():
        raise ValueError('Every frame needs a feasible option')
    if rotations.shape != (frames, 3, 3) or centroid.shape != (3,) or not np.isfinite(rotations).all() or not np.isfinite(centroid).all() or fidelity_weight < 0:
        raise ValueError('Invalid fixed geometry or fidelity weight')
    centers = options + np.einsum('tij,j->ti', rotations, centroid)[:, None]
    unary = fidelity_weight * np.square(options - options[:, :1]).sum(axis=2)
    cost = unary[0, :, None] + unary[1, None, :]
    cost[~(allowed[0, :, None] & allowed[1, None, :])] = np.inf
    parents = np.zeros((frames, count, count), dtype=np.int64)
    for t in range(2, frames):
        new = np.full((count, count), np.inf)
        for previous in range(count):
            for current in np.flatnonzero(allowed[t]):
                acceleration = centers[t, current] - 2 * centers[t-1, previous] + centers[t-2]
                choices = cost[:, previous] + np.square(acceleration).sum(axis=1) + unary[t, current]
                selected = int(np.argmin(choices))
                new[previous, current] = choices[selected]
                parents[t, previous, current] = selected
        cost = new
    previous, current = np.unravel_index(np.argmin(cost), cost.shape)
    if not np.isfinite(cost[previous, current]):
        raise ValueError('No feasible trajectory')
    path = np.zeros(frames, dtype=np.int64)
    path[-2:] = previous, current
    for t in range(frames-1, 1, -1):
        path[t-2] = parents[t, path[t-1], path[t]]
    return path, float(cost[previous, current])
