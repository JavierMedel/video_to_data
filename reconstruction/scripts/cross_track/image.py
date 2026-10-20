"""CPU silhouette rasterization for evidence gates; no tracking model imports."""
import numpy as np


def silhouette(vertices, faces, rotation, translation, scale, intrinsics, world_to_camera, shape):
    intrinsics, world_to_camera = np.asarray(intrinsics), np.asarray(world_to_camera)
    if (intrinsics.shape != (3, 3) or world_to_camera.shape != (4, 4)
            or not all(np.isfinite(x).all() for x in (vertices, rotation, translation, intrinsics, world_to_camera))
            or not np.isfinite(scale) or scale <= 0
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or min(shape) <= 0
            or not np.allclose(world_to_camera[3], [0, 0, 0, 1])
            or not np.allclose(world_to_camera[:3, :3]@world_to_camera[:3, :3].T, np.eye(3), atol=1e-5)):
        raise ValueError('Invalid camera calibration or geometry')
    world = scale*np.asarray(vertices)@np.asarray(rotation).T + translation
    camera = world@world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    if np.any(camera[:, 2] <= 1e-5):
        raise ValueError('Mesh crosses/behind near plane; use the established clipped GPU renderer')
    projection = camera@intrinsics.T
    xy = projection[:, :2]/projection[:, 2:]
    height, width = shape
    image = np.zeros(shape, bool)
    for triangle in xy[np.asarray(faces, int)]:
        a, b, c = triangle
        if abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])) < 1e-12: continue
        lo = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), [width-1, height-1])
        if np.any(lo > hi): continue
        x, y = np.meshgrid(np.arange(lo[0], hi[0]+1)+.5, np.arange(lo[1], hi[1]+1)+.5)
        def edge(u, v): return (x-u[0])*(v[1]-u[1])-(y-u[1])*(v[0]-u[0])
        e0, e1, e2 = edge(a, b), edge(b, c), edge(c, a)
        image[lo[1]:hi[1]+1, lo[0]:hi[0]+1] |= ((e0 >= 0)&(e1 >= 0)&(e2 >= 0)) | ((e0 <= 0)&(e1 <= 0)&(e2 <= 0))
    return image


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape or not b.any(): raise ValueError('Missing/nonmatching observed mask')
    return float(np.count_nonzero(a & b)/max(1, np.count_nonzero(a | b)))
