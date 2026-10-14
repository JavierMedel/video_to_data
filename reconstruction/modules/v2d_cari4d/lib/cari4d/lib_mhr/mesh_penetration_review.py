"""Hand-inside-object depth using the released Track 1 winding convention.

This implements the containment direction and geometry calculation, not the
hidden hand-role selection or human alignment scale of the leaderboard.
"""
import numpy as np
import torch


class MeshPenetrationReview:
    def __init__(self, vertices, faces, *, device='cuda', point_chunk=128, face_chunk=32768):
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError('Expected finite nonempty vertices [V,3]')
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError('Expected valid nonempty triangles [F,3]')
        if point_chunk <= 0 or face_chunk <= 0:
            raise ValueError('Chunk sizes must be positive')
        self.device = torch.device(device)
        self.point_chunk, self.face_chunk = point_chunk, face_chunk
        triangles = vertices[faces]
        valid = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=-1) > 0.
        self.triangles = torch.as_tensor(triangles[valid], device=device, dtype=torch.float64)
        self.triangles_f32 = self.triangles.float()
        self.lower = torch.as_tensor(vertices.min(axis=0), device=device)
        self.upper = torch.as_tensor(vertices.max(axis=0), device=device)

    @torch.inference_mode()
    def winding_numbers(self, query, *, dtype=torch.float64):
        query = torch.as_tensor(query, device=self.device, dtype=dtype)
        triangles = self.triangles if dtype == torch.float64 else self.triangles_f32
        out = torch.zeros(len(query), device=self.device, dtype=torch.float64)
        for start in range(0, len(query), self.point_chunk):
            q = query[start:start+self.point_chunk, None]
            total = torch.zeros(len(q), device=self.device, dtype=torch.float64)
            for begin in range(0, len(triangles), self.face_chunk):
                tri = triangles[begin:begin+self.face_chunk]
                a, b, c = (tri[None, :, k] - q for k in range(3))
                la, lb, lc = (torch.linalg.vector_norm(x, dim=-1) for x in (a, b, c))
                numerator = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
                denominator = la*lb*lc + (a*b).sum(dim=-1)*lc + (b*c).sum(dim=-1)*la + (c*a).sum(dim=-1)*lb
                total += torch.atan2(numerator, denominator).sum(dim=-1, dtype=torch.float64)
            out[start:start+len(q)] = total / (2.*np.pi)
        return out

    @torch.inference_mode()
    def depths(self, points):
        points = torch.as_tensor(points, device=self.device, dtype=torch.float64)
        if points.shape[-1:] != (3,) or not torch.isfinite(points).all():
            raise ValueError('Expected finite points [...,3]')
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        depths = torch.zeros(len(flat), device=self.device, dtype=torch.float64)
        near = torch.nonzero(((flat >= self.lower) & (flat <= self.upper)).all(dim=-1)).flatten()
        if not len(near) or not len(self.triangles):
            return depths.reshape(shape)
        winding = self.winding_numbers(flat[near], dtype=torch.float32)
        # Recompute near the official |w| > 0.5 boundary in float64. The wide
        # 0.01 band also covers open meshes, cancellation and surface rounding.
        uncertain = ((winding.abs()-.5).abs() <= .01) | ~torch.isfinite(winding)
        if uncertain.any():
            winding[uncertain] = self.winding_numbers(flat[near[uncertain]])
        inside = near[winding.abs() > .5]
        if len(inside):
            from kaolin.metrics.trianglemesh import point_to_mesh_distance
            triangles = self.triangles_f32[None].contiguous()
            for start in range(0, len(inside), self.point_chunk):
                ids = inside[start:start+self.point_chunk]
                distance2, _, _ = point_to_mesh_distance(flat[ids].float()[None].contiguous(), triangles)
                depths[ids] = distance2[0].clamp_min(0.).sqrt().double()
        return depths.reshape(shape)
