"""NumPy-only weighted Whittaker fits and exact pinned-row band solves."""
import numpy as np


def whittaker_smooth(y, w, lam, pin=None, robust_iters=0, huber_k=2.5):
    original = np.asarray(y)
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(w, dtype=np.float64)
    if y.ndim != 2 or len(y) < 3 or base.shape != (len(y),):
        raise ValueError('Expected T>=3 by D values and T weights')
    if not np.isfinite(y).all() or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Values must be finite and weights positive')
    if not np.isfinite(lam) or lam < 0 or robust_iters < 0 or huber_k <= 0:
        raise ValueError('Invalid smoothing parameters')
    if pin is not None and (not isinstance(pin, (int, np.integer)) or not 0 <= pin < len(y)):
        raise ValueError('Invalid pinned frame')
    n = len(y)
    # D2.T @ D2 has bandwidth two. This avoids a dense cubic solve.
    diagonal = np.zeros(n); off1 = np.zeros(n-1); off2 = np.ones(n-2)*lam
    diagonal[:-2] += lam; diagonal[1:-1] += 4*lam; diagonal[2:] += lam
    off1[:-1] -= 2*lam; off1[1:] -= 2*lam
    weights = base.copy()
    scale = np.maximum(np.median(np.abs(np.diff(y, axis=0)), axis=0)*1.4826, 1e-6)
    for iteration in range(robust_iters+1):
        a = diagonal+weights; b = off1.copy(); c = off2.copy(); rhs = weights[:, None]*y
        if pin is not None:
            for distance, band in ((1, b), (2, c)):
                for neighbor in (pin-distance, pin+distance):
                    if 0 <= neighbor < n:
                        index = min(pin, neighbor)
                        rhs[neighbor] -= band[index]*y[pin]
                        band[index] = 0.
            a[pin] = 1.; rhs[pin] = y[pin]
        l0 = np.zeros(n); l1 = np.zeros(n); l2 = np.zeros(n)
        for i in range(n):
            if i >= 2: l2[i] = c[i-2]/l0[i-2]
            if i >= 1: l1[i] = (b[i-1]-l2[i]*l1[i-1])/l0[i-1]
            l0[i] = np.sqrt(a[i]-l1[i]**2-l2[i]**2)
        x = rhs.copy()
        for i in range(n):
            if i >= 1: x[i] -= l1[i]*x[i-1]
            if i >= 2: x[i] -= l2[i]*x[i-2]
            x[i] /= l0[i]
        for i in range(n-1, -1, -1):
            if i+1 < n: x[i] -= l1[i+1]*x[i+1]
            if i+2 < n: x[i] -= l2[i+2]*x[i+2]
            x[i] /= l0[i]
        if pin is not None: x[pin] = y[pin]
        residual = np.sqrt(np.mean(((x-y)/scale)**2, axis=1))
        sigma = max(1.4826*np.median(np.abs(residual-np.median(residual))),
                    np.median(residual)*.1, 1e-8)
        if iteration < robust_iters:
            weights = base*np.minimum(1., huber_k*sigma/np.maximum(residual, 1e-12))
            weights = np.maximum(weights, base*1e-6)
    x = x.astype(original.dtype if original.dtype.kind == 'f' else np.float64)
    if pin is not None: x[pin] = original[pin]
    return x, {'lambda': float(lam), 'robust_iterations': robust_iters,
               'min_weight': float(weights.min()), 'max_weight': float(weights.max()),
               'pinned_frame': int(pin) if pin is not None else None}


def scored_acceleration_magnitude(points, frames):
    """Released ACC inclusion/endpoint replication; no alignment or GT claim.

    Points are in metres, full-video indexed; return unaligned centimetres.
    Average returned per-episode values equally to mirror leaderboard weighting.
    """
    points = np.asarray(points, dtype=np.float64); frames = np.asarray(frames)
    if frames.ndim != 1 or frames.dtype.kind not in 'iu' or not len(frames) or (np.diff(frames) <= 0).any():
        raise ValueError('Frames must be unique increasing integer indices')
    if frames[0] < 0 or frames[-1] >= len(points) or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError('Invalid points/frame coverage')
    values = []
    for run in np.split(frames, np.flatnonzero(np.diff(frames) != 1)+1):
        if len(run) < 3: raise ValueError('Scored stretch shorter than three frames')
        magnitude = np.linalg.norm(np.diff(points[run], n=2, axis=0), axis=-1)
        if magnitude.ndim > 1: magnitude = magnitude.mean(axis=tuple(range(1, magnitude.ndim)))
        values.extend(np.r_[magnitude[0], magnitude, magnitude[-1]])
    return float(np.mean(values)*100.)
