import numpy as np
import os
import cv2
import torch
from torch import nn 
import nvdiffrast.torch as dr
import smplx
import trimesh
import pyrender
os.environ['PYOPENGL_PLATFORM'] = 'egl'

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
def rotate_z(rad):
    return np.array([
        [np.cos(rad), -np.sin(rad), 0.],
        [np.sin(rad), np.cos(rad), 0.],
        [0., 0., 1.]
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
        # vertices = vertices - (vertices.max() + vertices.min()) / 2
        # vertices = vertices - vertices.min()
        vertices = vertices - vertices.mean()
        vertices = vertices / vertices.max() / 1.6
        vertices = vertices
    vertices = rotate_x(np.pi/2).dot(rotate_y(np.pi / 2).dot(vertices.T)).T
    tri_mesh = trimesh.Trimesh(vertices, model.faces)
    #tri_mesh = tri_mesh.simplify_quadratic_decimation(1000)  # 1000是目标面数
    tri_mesh.export(out_dir)

def zoom_bbox_in_apos():
    # headshot, ordered by XYZ
    head_bbox = [-0.25,-0.3,0,0.25,0.3,0.4]
    # leftarm
    lefthand_bbox = [0.2,0.5,-0.2,0.5,1.0,0.5]
    # rightarm
    # righthand_bbox = [0.2,0.5,0.2,0.5,1.0,0.5]
    # # leftleg
    # leftleg_bbox = [-0.2,0.5,-0.2,0.5,1.0,0.5]
    # # rightleg
    # rightleg_bbox = [-0.2,0.5,0.2,0.5,1.0,0.5]
    return head_bbox

def check_bbox(c2w, camera_position=None):
    fmesh = trimesh.load("smpl.obj")
    mesh = pyrender.Mesh.from_trimesh(fmesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    s = np.sqrt(2)/2
    camera_pose = c2w.numpy()
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=1.0,
        innerConeAngle=np.pi/16.0,
        outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(400, 400)
    color, depth = r.render(scene)
    # biproject to get the pixel position
    head_bbox = torch.tensor(zoom_bbox_in_apos()).reshape(-1, 3)
    head_points = (head_bbox - c2w[:3, 3]) @ c2w[:3,:3]
    f = 200 / np.tan(np.pi / 6.0)
    head_points = head_points / head_points[:, 2:] * f + 200
    head_points = torch.sort(head_points, 1)[0]
    color = color.copy()
    print(color.dtype)
    color[head_points[0, 0].int(), head_points[0, 1].int()] = [255, 0, 0]
    color[head_points[1, 0].int(), head_points[1, 1].int()] = [255, 0, 0]
    cv2.imwrite("out.png", color)


import cv2
def draw_bodypose(canvas: np.ndarray, keypoints) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape
    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        if k1_index-1 >= len(keypoints) or k2_index -1>= len(keypoints):
            continue
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]]) * float(W)
        X = np.array([keypoint1[1], keypoint2[1]]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, 
                            color
                           #[int(float(c) * 0.6) for c in color]
                           )

    canvas = (canvas * 0.6).astype(np.uint8)

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint[0],keypoint[1]
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas
import math
def draw_poses(poses, H, W, draw_body=True, draw_hand=True, draw_face=True):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    # for pose in poses:
    pose = poses
    if draw_body:
            canvas = draw_bodypose(canvas, pose)

        # if draw_hand:
        #     canvas = draw_handpose(canvas, pose.left_hand)
        #     canvas = draw_handpose(canvas, pose.right_hand)

        # if draw_face:
        #     canvas = draw_facepose(canvas, pose.face)

    return canvas

if __name__ == '__main__':
    save_smpl_to_obj(model_folder="/home/zjp/zjp/threestudio", out_dir="/home/zjp/zjp/threestudio/smpl.obj", gender='neutral')