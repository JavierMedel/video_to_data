"""Fit a metric circular ring to pinhole rays; never use background depth."""
import numpy as np


def fit_circle_rays(rays, radius_m, initial_center, initial_normal, *, focal_pixels=1., max_evaluations=80):
    """Return center, unoriented plane normal, and robust radial image residual.

The radius must come from the input metric mesh. Rays originate at the camera
and have Z=1. Two mirrored plane initializations should be fitted independently.
    """
    from scipy.optimize import least_squares
    rays = np.asarray(rays, dtype=np.float64)
    center = np.asarray(initial_center, dtype=np.float64)
    normal = np.asarray(initial_normal, dtype=np.float64)
    if rays.ndim != 2 or rays.shape[1] != 3 or len(rays) < 12 or radius_m <= 0 or not np.isfinite(rays).all():
        raise ValueError('Expected finite [N,3] rays, N>=12, and positive radius')
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal *= -1.
    theta = np.clip(np.arccos(np.clip(normal[2], -1, 1)), 1e-4, np.pi/2-1e-4)
    phi = np.arctan2(normal[1], normal[0])

    def normal_from(x):
        return np.array([np.sin(x[3])*np.cos(x[4]), np.sin(x[3])*np.sin(x[4]), np.cos(x[3])])

    def residual(x):
        n = normal_from(x)
        denominator = rays @ n
        safe = np.where(abs(denominator) > 1e-5, denominator, np.where(denominator >= 0, 1e-5, -1e-5))
        distance = (n @ x[:3]) / safe
        points = rays * distance[:, None]
        return (np.linalg.norm(points - x[:3], axis=1) - radius_m) * focal_pixels / x[2]

    x0 = np.r_[center, theta, phi]
    lower = [-10., -10., radius_m + .05, 1e-5, -4*np.pi]
    upper = [10., 10., 15., np.pi/2-1e-5, 4*np.pi]
    fitted = least_squares(residual, x0, bounds=(lower, upper), loss='soft_l1', f_scale=2., max_nfev=max_evaluations)
    return fitted.x[:3], normal_from(fitted.x), float(np.median(abs(residual(fitted.x))))


def frame_from_normal(normal):
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent = np.array([0., 1., 0.])
    if abs(tangent @ normal) > .9:
        tangent = np.array([1., 0., 0.])
    tangent -= (tangent @ normal) * normal
    tangent /= np.linalg.norm(tangent)
    return np.stack([normal, tangent, np.cross(normal, tangent)], axis=1)
