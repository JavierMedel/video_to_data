"""Gap-aware robust QP and SO(3) trajectory fits. All positions are metres."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def stretches(frames, length):
    frames = np.asarray(frames)
    if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu'
            or frames[0] < 0 or frames[-1] >= length or np.any(np.diff(frames) <= 0)):
        raise ValueError('Expected increasing, unique original frame indices')
    return np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1)


def acceleration(points, frames):
    """Own acceleration magnitude in cm, with official endpoint replication."""
    points = np.asarray(points, dtype=float)
    values = []
    for run in stretches(frames, len(points)):
        if len(run) < 3:
            raise ValueError('Scored stretches require at least three frames')
        a = np.linalg.norm(np.diff(points[run], n=2, axis=0), axis=-1)
        if a.ndim > 1:
            a = a.mean(axis=tuple(range(1, a.ndim)))
        values.extend(np.r_[a[0], a, a[-1]])
    return float(np.mean(values) * 100)


def qp(hessian, rhs, matrix=None, lower=None, upper=None, *, tol=1e-8, iterations=5000):
    """Convex quadratic solver: .5*x'Hx-rhs'x subject to lower<=Ax<=upper.

    ADMM splits linear constraints from a sparse positive-definite quadratic.
    Reject non-convergence instead of exporting an infeasible approximation.
    """
    hessian = sparse.csc_matrix(hessian)
    rhs = np.asarray(rhs, dtype=float)
    if matrix is None or matrix.shape[0] == 0:
        return splu(hessian).solve(rhs)
    matrix = sparse.csc_matrix(matrix)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if matrix.shape[1] != len(rhs) or lower.shape != (matrix.shape[0],) or upper.shape != lower.shape:
        raise ValueError('Invalid constraint shapes')
    if np.any(lower > upper) or not np.isfinite(rhs).all():
        raise ValueError('Invalid constraint bounds or objective')
    rho = max(1., float(np.median(hessian.diagonal())))
    factor = splu(hessian + rho * (matrix.T @ matrix))
    x = splu(hessian).solve(rhs)
    z = np.clip(matrix @ x, lower, upper)
    u = np.zeros_like(z)
    for _ in range(iterations):
        x = factor.solve(rhs + rho * (matrix.T @ (z - u)))
        ax = matrix @ x
        previous = z.copy()
        z = np.clip(ax + u, lower, upper)
        u += ax - z
        primal = np.max(np.abs(ax - z))
        dual = np.max(np.abs(rho * (matrix.T @ (z - previous))))
        if primal <= tol and dual <= tol:
            return x
    raise ValueError(f'Constrained QP did not converge (primal={primal:g}, dual={dual:g})')


def smooth(values, frames, lam, weights=None, contact_target=None, contact_weights=None,
           constraints=None, robust_iters=3, huber_m=.01, pin_first=True):
    """Smooth each scored stretch; leave all unscored frames untouched.

    constraints: (original_frame, normal[D], upper_bound) entries for n @ x <= b.
    Contact targets and weights are supplied by observed contact windows.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or not np.isfinite(lam) or lam < 0:
        raise ValueError('Expected finite T by D values and nonnegative lambda')
    n, width = values.shape
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    cw = np.zeros(n) if contact_weights is None else np.asarray(contact_weights, dtype=float)
    target = values if contact_target is None else np.asarray(contact_target, dtype=float)
    if (weights.shape != (n,) or cw.shape != (n,) or target.shape != values.shape
            or not all(np.isfinite(a).all() for a in (weights, cw, target))
            or np.any(weights <= 0) or np.any(cw < 0) or huber_m <= 0):
        raise ValueError('Invalid evidence/contact weights')
    result = values.copy()
    for run in stretches(frames, n):
        size = len(run)
        if size < 3:
            continue
        difference = sparse.diags([np.ones(size-2), -2*np.ones(size-2), np.ones(size-2)],
                                  [0, 1, 2], shape=(size-2, size), format='csc')
        rows, lows, highs = [], [], []
        lookup = {int(f): i for i, f in enumerate(run)}
        if pin_first and run[0] == frames[0]:
            for dimension in range(width):
                row = sparse.csr_matrix(([1.], ([0], [dimension])), shape=(1, size*width))
                rows.append(row); lows.append(values[run[0], dimension]); highs.append(lows[-1])
        for frame, normal, bound in constraints or []:
            if frame not in lookup:
                continue
            normal = np.asarray(normal, dtype=float)
            if normal.shape != (width,) or not np.isfinite(normal).all() or not np.isfinite(bound):
                raise ValueError('Invalid half-space')
            rows.append(sparse.csr_matrix((normal, (np.zeros(width), lookup[frame]*width+np.arange(width))),
                                          shape=(1, size*width)))
            lows.append(-np.inf); highs.append(float(bound))
        matrix = sparse.vstack(rows, format='csc') if rows else None
        w = weights[run].copy()
        for iteration in range(robust_iters + 1):
            h = sparse.diags(w + cw[run]) + lam * (difference.T @ difference)
            rhs = w[:, None]*values[run] + cw[run, None]*target[run]
            x = qp(sparse.kron(h, sparse.eye(width)), rhs.ravel(), matrix,
                   np.asarray(lows), np.asarray(highs)).reshape(size, width)
            residual = np.linalg.norm(x-values[run], axis=1)
            if iteration < robust_iters:
                w = weights[run] * np.minimum(1., huber_m / np.maximum(residual, 1e-12))
        result[run] = x
    if pin_first:
        result[frames[0]] = values[frames[0]]
    return result


def smooth_rotations(rotations, frames, lam, *, max_nfev=100):
    """Fit in SO(3), with spatial angular increments and a pinned first frame."""
    rotations = np.asarray(rotations, dtype=float)
    if rotations.shape != (len(rotations), 3, 3) or lam < 0 or not np.isfinite(rotations).all():
        raise ValueError('Invalid rotations/lambda')
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6)):
        raise ValueError('Rotations must lie in SO(3)')
    out = rotations.copy()
    for run in stretches(frames, len(rotations)):
        if len(run) < 3:
            continue
        base = rotations[run]
        pinned = int(run[0] == frames[0])
        def residual(x):
            delta = np.zeros((len(run), 3)); delta[pinned:] = x.reshape(-1, 3)
            r = Rotation.from_rotvec(delta).as_matrix() @ base
            velocity = Rotation.from_matrix(r[1:] @ r[:-1].transpose(0, 2, 1)).as_rotvec()
            return np.r_[delta.ravel(), np.sqrt(lam)*np.diff(velocity, axis=0).ravel()]
        # Each residual block touches at most three neighbouring frames.
        pattern = sparse.lil_matrix((len(run)*3+(len(run)-2)*3, (len(run)-pinned)*3))
        for i in range(len(run)):
            if i >= pinned: pattern[3*i:3*i+3, 3*(i-pinned):3*(i-pinned+1)] = 1
        for i in range(len(run)-2):
            for j in range(max(pinned, i), i+3):
                pattern[3*len(run)+3*i:3*len(run)+3*i+3, 3*(j-pinned):3*(j-pinned+1)] = 1
        fit = least_squares(residual, np.zeros((len(run)-pinned)*3), jac_sparsity=pattern.tocsr(), max_nfev=max_nfev)
        if not fit.success:
            raise ValueError('SO(3) fit did not converge')
        delta = np.zeros((len(run), 3)); delta[pinned:] = fit.x.reshape(-1, 3)
        out[run] = Rotation.from_rotvec(delta).as_matrix() @ base
    out[frames[0]] = rotations[frames[0]]
    return out


def gcv_lambda(values, frames, lambdas, weights=None):
    """Weighted linear prediction GCV; NOT a hidden-reference ACC estimator.

    Exact effective degrees of freedom, a fixed first scored anchor, and pooled
    independent scored stretches. Trace solves are blocked to bound memory.
    Contact constraints and Huber reweighting are deliberately not approximated.
    """
    values = np.asarray(values, float)
    if values.ndim != 2 or not np.isfinite(values).all(): raise ValueError('Require finite T by D values')
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(values),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Invalid GCV weights')
    runs = stretches(frames, len(values)); scores = []
    for lam in sorted(set(lambdas)):
        if not np.isfinite(lam) or lam <= 0: raise ValueError('GCV requires positive finite lambdas')
        rss, trace, observations = 0., 0., 0
        for run in runs:
            n = len(run)
            if n < 3: continue
            d = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n), format='csc')
            h = (sparse.diags(weights[run])+lam*(d.T@d)).tocsc()
            pin = int(run[0] == frames[0])
            factor = splu(h[pin:, pin:])
            rhs = weights[run, None]*values[run]
            rhs = rhs[pin:]-h[pin:, :pin]@values[run[:pin]]
            fitted = np.vstack([values[run[:pin]], factor.solve(rhs)])
            diagonal = np.zeros(n-pin)
            for start in range(0, n-pin, 32):
                ids = np.arange(start, min(start+32, n-pin))
                unit = np.zeros((n-pin, len(ids))); unit[ids, np.arange(len(ids))] = 1.
                inverse_columns = factor.solve(unit)
                diagonal[ids] = inverse_columns[ids, np.arange(len(ids))]
            df = pin+float(diagonal@weights[run[pin:]])
            rss += float(np.sum(weights[run, None]*(fitted-values[run])**2))
            trace += df*values.shape[1]; observations += n*values.shape[1]
        if not observations: raise ValueError('GCV needs a scored stretch of at least three frames')
        denominator = observations-trace
        score = observations*rss/denominator**2 if denominator > 1e-9 else float('inf')
        if not np.isfinite(score): raise ValueError('GCV has no residual degrees of freedom')
        scores.append({'lambda': float(lam), 'gcv': score, 'weighted_rss': rss,
                       'effective_df': trace, 'observations': observations})
    if not scores: raise ValueError('Empty GCV lambda grid')
    return {'lambda': min(scores, key=lambda row: (row['gcv'], row['lambda']))['lambda'],
            'scores': scores, 'scope': 'Linear weighted prediction residual; constrained ACC benefit needs held-out evidence'}
