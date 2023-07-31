import numpy as np
import torch
from torch import nn 
import nvdiffrast.torch as dr
import smplx
import trimesh

def convert_sdf_to_alpha(sdf_guide):
    beta = 2e-3
    alpha = 1.0 / beta
    v = -1. * sdf_guide * alpha
    sg = alpha * torch.sigmoid(v)
    soft_guide = torch.log(torch.exp(sg) - 1)
    density_guide = torch.clamp(soft_guide, min=0.0)
    return density_guide

def rotate_x(rad):
    return np.array([
        [1., 0., 0.],
        [0., np.cos(rad), -np.sin(rad)],
        [0., np.sin(rad), np.cos(rad)]
    ])

def rotate_y(rad):
    return np.array([
        [np.cos(rad), 0., np.sin(rad)],
        [0., 1., 0.],
        [-np.sin(rad), 0., np.cos(rad)]
    ])

def save_smpl_to_obj(model_folder, out_dir="smpl.obj", model_type='smplx', ext='npz', gender='neutral', 
    num_betas=10, num_expression_coeffs=10, use_face_contour=False, sample_shape=False, sample_expression=True, bbox=None
    ,rotate=None):
    #print("model_type: ", model_type)
    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    betas, expression = None, None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    else:
        betas = torch.zeros([1, model.num_betas], dtype=torch.float32)
        betas[:, 1] = -0.5
    if sample_expression:
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)
    a_pose = torch.zeros_like(model.body_pose).reshape(1,-1,3)
    a_pose[:,12,2] = -0.8
    a_pose[:,13,2] = 0.8
    output = model(body_pose=a_pose, betas=betas, expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    if bbox is None:
        vertices = (vertices - vertices.min()) 
        vertices = vertices / vertices.max()
    else:
        vertices = vertices - (vertices.max() + vertices.min()) / 2
        vertices = vertices / vertices.max()
    vertices = rotate_x(np.pi/2).dot(rotate_y(np.pi / 2).dot(vertices.T)).T
    tri_mesh = trimesh.Trimesh(vertices, model.faces)
    tri_mesh = tri_mesh.simplify_quadratic_decimation(1000)  # 1000是目标面数
    tri_mesh.export(out_dir)

if __name__ == '__main__':
    save_smpl_to_obj(model_folder="/home/zjp/zjp/threestudio", out_dir="/home/zjp/zjp/threestudio/smpl.obj", gender='neutral')