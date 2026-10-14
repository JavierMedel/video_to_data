"""Select mesh-scored rotation hypotheses jointly across a video.

Hypotheses must be scored using the actual object geometry. An alternative
orientation is not assumed to be a physical symmetry (e.g. a pan has a handle).
"""
import numpy as np


def project_rotations(matrices):
    """Project finite 3x3 matrices onto SO(3), including reflection correction."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError('Expected finite [...,3,3] matrices')
    u, _, vh = np.linalg.svd(matrices)
    sign = np.ones(matrices.shape[:-2] + (3,))
    sign[..., -1] = np.linalg.det(u @ vh)
    return (u * sign[..., None, :]) @ vh


def rotation_hypotheses(predicted, tracked):
    """Include both observed stages and a short-window initialization candidate."""
    predicted, tracked = map(project_rotations, (predicted, tracked))
    if predicted.shape != tracked.shape or predicted.ndim != 3:
        raise ValueError('Expected matching [T,3,3] rotation sequences')
    averaged = np.stack([predicted[max(0, t-3):t+4].mean(axis=0) for t in range(len(predicted))])
    averaged = project_rotations(averaged)
    # These are hypotheses, not declared symmetry transforms. Silhouette
    # evidence is evaluated separately for all four using the entire mesh.
    flips = np.array([np.eye(3), np.diag([1, -1, -1]),
                      np.diag([-1, 1, -1]), np.diag([-1, -1, 1])])
    return np.concatenate([source[:, None] @ flips[None] for source in (predicted, tracked, averaged)], axis=1)


def circular_hoop_rotations(rotations, *, smoothing_radius=2):
    """Transport a frame along the unoriented normal of a circular hoop.

Only for a circular hoop with its thin axis on local X; it is unsuitable for
asymmetric handles. The caller must still check actual-mesh image alignment.
    """
    rotations = project_rotations(rotations)
    normals = rotations[:, :, 0].copy()
    for t in range(1, len(normals)):
        if normals[t] @ normals[t-1] < 0:
            normals[t] *= -1.
    normals = np.stack([normals[max(0, t-smoothing_radius):t+smoothing_radius+1].mean(axis=0) for t in range(len(normals))])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    output = np.empty_like(rotations)
    tangent = rotations[0, :, 1].copy()
    for t, normal in enumerate(normals):
        tangent = tangent - (tangent @ normal) * normal
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.eye(3)[np.argmin(abs(normal))]
            tangent -= (tangent @ normal) * normal
        tangent /= np.linalg.norm(tangent)
        output[t] = np.stack([normal, tangent, np.cross(normal, tangent)], axis=1)
    return output


def select_temporal_rotations(hypotheses, observation_cost, *, motion_weight=0.04, motion_scale_degrees=20., translations=None, translation_weight=0.04, translation_scale_m=.05):
    """Find the exact minimum-cost path with a quadratic angular step penalty."""
    hypotheses = np.asarray(hypotheses, dtype=np.float64)
    costs = np.asarray(observation_cost, dtype=np.float64)
    if hypotheses.ndim != 4 or hypotheses.shape[-2:] != (3, 3) or costs.shape != hypotheses.shape[:2] or not costs.size:
        raise ValueError('Expected rotations [T,K,3,3] and nonempty costs [T,K]')
    if not np.isfinite(hypotheses).all() or not np.isfinite(costs).all():
        raise ValueError('Hypotheses and costs must be finite')
    if motion_weight < 0 or motion_scale_degrees <= 0:
        raise ValueError('Invalid motion penalty')
    if translations is not None:
        translations = np.asarray(translations, dtype=np.float64)
        if translations.shape != hypotheses.shape[:2] + (3,) or not np.isfinite(translations).all() or translation_weight < 0 or translation_scale_m <= 0:
            raise ValueError('Expected finite translations [T,K,3] and valid translation penalty')
    if not np.allclose(hypotheses @ hypotheses.swapaxes(-1, -2), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(hypotheses), 1., atol=1e-4):
        raise ValueError('Hypotheses must be proper rotations')
    count, choices = costs.shape
    previous = costs[0].copy()
    parents = np.zeros((count, choices), dtype=np.int32)
    for t in range(1, count):
        # trace(R_a.T R_b) equals the Frobenius inner product.
        cosine = (np.einsum('aij,bij->ab', hypotheses[t-1], hypotheses[t]) - 1.) / 2.
        angles = np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))
        transition = previous[:, None] + motion_weight * (angles / motion_scale_degrees) ** 2
        if translations is not None:
            displacement = translations[t-1, :, None] - translations[t, None]
            transition += translation_weight * np.sum(displacement**2, axis=-1) / translation_scale_m**2
        parents[t] = transition.argmin(axis=0)
        previous = costs[t] + transition[parents[t], np.arange(choices)]
    path = np.empty(count, dtype=np.int32)
    path[-1] = previous.argmin()
    objective = float(previous[path[-1]])
    for t in range(count-1, 0, -1):
        path[t-1] = parents[t, path[t]]
    return hypotheses[np.arange(count), path], path, objective
