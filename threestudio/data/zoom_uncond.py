import bisect
import math
import random
from dataclasses import dataclass, field
import trimesh
import os 
import pyrender
from pyrender.constants import RenderFlags

import numpy as np
import cv2
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

from threestudio.utils.smpl_utils import zoom_bbox_in_apos, check_bbox,draw_poses,rotate_x,rotate_y,rotate_z

import smplx
import joblib

@dataclass
class RandomCameraDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    batch_size: int = 1
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    focus_mode: int = 0
    progressive_radius: float = 1.0
    max_steps: int = 30000
    focus_camera_distance: float = 0.6
    smpl_dir: str = "smpl.obj"

class RandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.head_bbox = torch.tensor(zoom_bbox_in_apos())
        #CHANGE
        smpl_model = smplx.create("/home/ldy/ldy/smplx_openpose/wiki/assets/SMPLX_OpenPose_mapping/models",model_type="smplx")
        smpl_data = joblib.load("/home/penghy/diffusion/avatars/sketchhuman/extern/PyMAF-X/output/anime/output.pkl")
        pose = torch.tensor(smpl_data["pose"][0].reshape(1,72)[:,3:66])
        betas = torch.tensor(smpl_data["betas"][0]).reshape(1, 10)
        smpl_mesh = smpl_model(betas=betas, body_pose=pose, return_verts=True)
        joints = smpl_mesh.joints.squeeze()
        mapping = [55, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59]
        openpose_joints = joints[mapping]
        openpose_joints = rotate_x(np.pi/2).dot(rotate_y(-np.pi / 2).dot(openpose_joints.detach().cpu().numpy().T)).T
        openpose_joints = torch.from_numpy(openpose_joints).float()
        self.openpose_joints_homogeneous = torch.cat([openpose_joints, torch.ones_like(openpose_joints[:, :1])], dim=-1)
        
        self.smpl_mesh = trimesh.load(self.cfg.smpl_dir)
        self.colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
            # 'purple': np.array([0.5, 0.5, 0.7]),
            'purple': np.array([0.55, 0.4, 0.9]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }
        self.init_pyrender(self.smpl_mesh)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        if self.height != self.heights[size_ind]:
            self.renderer.delete()
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=self.widths[size_ind],
                viewport_height=self.heights[size_ind],
                point_size=1.0
            )
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.cfg.batch_size)
                * (self.cfg.elevation_range[1] - self.cfg.elevation_range[0])
                + self.cfg.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.cfg.elevation_range[0] + 90.0) / 180.0,
                (self.cfg.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.cfg.batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                torch.rand(self.cfg.batch_size) + torch.arange(self.cfg.batch_size)
            ) / self.cfg.batch_size * (
                self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0]
            ) + self.cfg.azimuth_range[
                0
            ]
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.cfg.batch_size)
                * (self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0])
                + self.cfg.azimuth_range[0]
            )
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.cfg.batch_size)
            * (self.cfg.camera_distance_range[1] - self.cfg.camera_distance_range[0])
            + self.cfg.camera_distance_range[0]
        )

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(self.cfg.batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.cfg.batch_size, 3) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.cfg.batch_size, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.cfg.batch_size)
            * (self.cfg.fovy_range[1] - self.cfg.fovy_range[0])
            + self.cfg.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.cfg.batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.cfg.batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.cfg.batch_size) * math.pi - 2 * math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.cfg.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.cfg.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        # print(self.directions_unit_focal)

        # get focus mode part directions
        focus = self.focus_mode_camera_position(self.smpl_mesh)
        focus_rays = self.generate_directions_map(focus, focal_length)
        # rays_d_head = self.project2pixel(c2w, self.head_bbox)
        # rays_d_head[:, :, :, :2] = rays_d_head[:, :, :, :2] / focal_length[:, None, None, None]
        # print(rays_d_head[0,:10])
        # print(directions[0,:10])
        # exit()
        # rays_o_head, rays_d_head = get_rays(rays_d_head, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        smpl_map = self.generate_view_smpl_map(c2w, fovy)
        openpose_map = self.generate_view_smplx_openpose(c2w, fovy)
        batch =  {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "rendered_smpl": smpl_map,
            "rendered_openpose": openpose_map,
        }
        for k, v in focus_rays.items():
            batch[k] = v
        return batch

    SMPLX_VERTEX_MAP = {
        'nose':		    9120,
        'reye':		    9929,
        'leye':		    9448,
        'rear':		    616,
        'lear':		    6,
        'rthumb':		8079,
        'rindex':		7669,
        'rmiddle':		7794,
        'rring':		7905,
        'rpinky':		8022,
        'lthumb':		5361,
        'lindex':		4933,
        'lmiddle':		5058,
        'lring':		5169,
        'lpinky':		5286,
        'LBigToe':		5770,
        'LSmallToe':    5780,
        'LHeel':		8846,
        'RBigToe':		8463,
        'RSmallToe': 	8474,
        'RHeel':  		8635
    }

    def init_pyrender(self, smpl_mesh):
        color = self.colors_dict['purple']
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )
        self.scene = pyrender.Scene()
        light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]) * 0.6, intensity=10)

        yrot = np.radians(120) # angle of lights

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)
        mesh = pyrender.Mesh.from_trimesh(smpl_mesh, material=material)
        mesh_node = self.scene.add(mesh, 'mesh')
        h, w = self.height, self.width
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=w,
            viewport_height=h,
            point_size=1.0
        )
        
    def generate_view_smpl_map(self, mvp_mtx, fovy):
        h, w = self.height, self.width
        focal_length = 0.5 * h / np.tan(0.5 * fovy)
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=w/2., cy=h/2.)
        mvp_mtx = mvp_mtx[0][[0,1,2,3]]
        cam_node = self.scene.add(camera, pose=mvp_mtx.numpy())
        render_flags = RenderFlags.SHADOWS_SPOT
        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        self.scene.remove_node(cam_node)
        cv2.imwrite("smpl222.png", rgb)
        return rgb

    def generate_view_smplx_openpose(self, mvp_mtx, fovy):
        # with torch.no_grad():
        # breakpoint()
        h, w = self.height, self.width
        focal_length = 0.5 * h / np.tan(0.5 * fovy)
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                        cx=w/2., cy=h/2.)
        mvp_mtx = mvp_mtx[0][[0,1,2,3]]
        camera_intrinsic = camera.get_projection_matrix(w, h)
        camera_intrinsic = torch.tensor(camera_intrinsic, dtype=torch.float32)
        # print(camera_intrinsic)
        #
        project_joints = self.openpose_joints_homogeneous
        # points_homogenous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        # breakpoint()
        # print(points_homogenous.shape)
        # print((pose @ points_homogenous.T)[:3,].shape)
        # trans = camera_intrinsic@pose

        projected_points=torch.inverse(mvp_mtx)@project_joints.T
        breakpoint()
        
        # aa/=aa[3,:]
        # print(aa)
        # aa =aa[:2,:].T
        # projected_points = aa
        projected_points = projected_points[:3,:].permute(1,0)
        projected_points = projected_points[:, :2] / projected_points[:, 2:3] * focal_length + h // 2
        # projected_points=camera_intrinsic@projected_points
        # projected_points = projected_points.T
        # projected_points/=projected_points[-1,:].clone()
        # projected_points = projected_points[:2,:].T
        # print((pose @ points_homogenous.T)[:3,])
        # print(camera_intrinsic.shape)
        # print(camera_intrinsic)
        # projected_points = camera_intrinsic @ (pose @ points_homogenous.T)
        # projected_points=  (projected_points+1)/2
        # projected_points = 1-projected_points


        detect_resolution = 512
        projected_points /= detect_resolution
        openpose_img = draw_poses(projected_points, detect_resolution,detect_resolution,draw_body=True, draw_hand=False, draw_face=False)
        openpose_img = cv2.resize(openpose_img, (w,h), interpolation=cv2.INTER_LINEAR)

        # cam_node = self.scene.add(camera, pose=mvp_mtx.numpy())
        # render_flags = RenderFlags.SHADOWS_SPOT
        # rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        # self.scene.remove_node(cam_node)
        cv2.imwrite("smplx_openpose.png", openpose_img)
        breakpoint()
        return openpose_img

    def focus_mode_camera_position(self, smpl_mesh):
        # todo: test nodes here.
        focus = {}
        if self.cfg.focus_mode >= 1:
            # head_mode
            nose_point, nose_normal = torch.tensor(smpl_mesh.vertices[8981], dtype=torch.float), torch.tensor(smpl_mesh.vertex_normals[8981], dtype=torch.float)
            nose_normal = F.normalize(nose_normal, dim=-1)
            center_perturb: Float[Tensor, "B 3"] = (
                torch.randn(self.cfg.batch_size, 3) * self.cfg.center_perturb
            ) * 0.1
            center = nose_point.unsqueeze(0) + center_perturb
            head_camera_positions = center + nose_normal.unsqueeze(0) * self.cfg.focus_camera_distance
            focus["head"] = {"pos": head_camera_positions, "lookat": -nose_normal}
        if self.cfg.focus_mode >= 2:
            # torso mode
            torso_point, torso_normal = torch.tensor(smpl_mesh.vertices[3855], dtype=torch.float), torch.tensor(smpl_mesh.vertex_normals[3855], dtype=torch.float)
            nose_normal = F.normalize(torso_normal, dim=-1)
            center_perturb: Float[Tensor, "B 3"] = (
                torch.randn(self.cfg.batch_size, 3) * self.cfg.center_perturb
            ) * 0.1
            center = nose_point.unsqueeze(0) + center_perturb
            torso_camera_positions = center + torso_normal.unsqueeze(0) * self.cfg.focus_camera_distance
            focus["torso"] = {"pos": torso_camera_positions, "lookat": -torso_normal}
        return focus

    def generate_directions_map(self, focus, focal_length):
        focus_rays = {}
        up = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32).repeat(self.cfg.batch_size,1)
        for part, part_info in focus.items():
            lookat = part_info["lookat"].unsqueeze(0).repeat(self.cfg.batch_size, 1)
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
                [torch.stack([right, up, -lookat], dim=-1), part_info["pos"][:, :, None]],
                dim=-1,
            )
            c2w: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0
            directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                None, :, :, :
            ].repeat(self.cfg.batch_size, 1, 1, 1)
            directions[:, :, :, :2] = (
                directions[:, :, :, :2] / focal_length[:, None, None, None]
            )
            rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
            focus_rays[f"rays_o_{part}"] = rays_o
            focus_rays[f"rays_d_{part}"] = rays_d
        return focus_rays

    def project2pixel(self, c2w, bbox):
        import numpy as np
        # bbox: [6] # 1 part for now
        # c2w: [B, 4, 4]
        # return: [B, N, 2]
        points = bbox.unsqueeze(0).repeat(c2w.shape[0], 1).view(-1, 2, 3)
        points = (points - c2w[:,:3,3]) @ c2w[:, :3, :3]
        points = points[:, :, :2] / points[:, :, 2:3] * 64
        # points = points[:, :, :2]
        points, _ = torch.sort(points, 1)
        rays_pb = []
        for b in range(points.shape[0]):
            p_x = torch.linspace(points[b,0,0].item(), points[b,1,0].item(), 64)
            p_y = torch.linspace(points[b,1,1].item(), points[b,0,1].item(), 64)
            p_d = torch.stack(torch.meshgrid(p_x, p_y, indexing="xy"), dim=-1)
            p_d = torch.cat([p_d, -torch.ones_like(p_d[..., :1])], dim=-1)
            rays_pb.append(p_d)
        return torch.stack(rays_pb, 0)

class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = torch.linspace(0, 360.0, self.n_views)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("zoom-random-camera-datamodule")
class ZoomRandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
