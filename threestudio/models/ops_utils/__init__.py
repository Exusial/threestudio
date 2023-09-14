import numpy 
import torch
import trimesh

import threestudio
from threestudio.models.mesh import Mesh
from threestudio.utils.typing import *
import cutils

def dual_contouring_undc(points, threshold=15.0, resolution=128):
    print(type(points))
    int_grid = torch.where(points > threshold, 1, 0).int().unsqueeze(-1)
    float_grid = torch.zeros_like(points).unsqueeze(-1).repeat(1,1,1,3) + 0.5
    vertices, faces = cutils.dual_contouring_undc(int_grid.detach().cpu().contiguous().numpy(), float_grid.detach().cpu().contiguous().numpy())
    print(vertices.shape, vertices.min(), vertices.max())
    vertices = vertices / (resolution - 1)
    return vertices, faces

