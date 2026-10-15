"""Opt-in kernel fusion for the unchanged generalized-winding calculation.

The containment threshold, float64 accumulation, uncertain-boundary fallback,
AABB screen and triangle-distance calculation remain those of the base class.
This fuses the point/face arithmetic to avoid materializing its intermediates.
"""
import numpy as np
import torch

from .mesh_penetration_review import MeshPenetrationReview


def _angle_sum(query, triangles):
    a, b, c = (triangles[None, :, k] - query[:, None] for k in range(3))
    la, lb, lc = (torch.linalg.vector_norm(x, dim=-1) for x in (a, b, c))
    numerator = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
    denominator = la*lb*lc + (a*b).sum(dim=-1)*lc + (b*c).sum(dim=-1)*la + (c*a).sum(dim=-1)*lb
    return torch.atan2(numerator, denominator).sum(dim=-1, dtype=torch.float64)


_fused_angle_sum = torch.compile(_angle_sum, fullgraph=True, dynamic=True,
                                  options={'use_fast_math': False, 'cuda.use_fast_math': False})


class FusedMeshPenetrationReview(MeshPenetrationReview):
    @torch.inference_mode()
    def winding_numbers(self, query, *, dtype=torch.float64):
        query = torch.as_tensor(query, device=self.device, dtype=dtype)
        triangles = self.triangles if dtype == torch.float64 else self.triangles_f32
        out = torch.zeros(len(query), device=self.device, dtype=torch.float64)
        for start in range(0, len(query), self.point_chunk):
            q = query[start:start+self.point_chunk]
            total = torch.zeros(len(q), device=self.device, dtype=torch.float64)
            for begin in range(0, len(triangles), self.face_chunk):
                total += _fused_angle_sum(q, triangles[begin:begin+self.face_chunk])
            out[start:start+len(q)] = total / (2.*np.pi)
        return out
