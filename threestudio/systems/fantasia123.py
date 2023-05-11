from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import h5py
import numpy as np
from PIL import Image

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("fantasia123-system")
class Fantasia3D(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = "implicit-sdf"
        geometry: dict = field(default_factory=lambda: {"n_feature_dims": 0})
        material_type: str = "no-material"  # unused
        material: dict = field(default_factory=lambda: {"n_output_dims": 0})
        background_type: str = "solid-color-background"  # unused
        background: dict = field(default_factory=dict)
        renderer_type: str = "nvdiff-rasterizer"
        renderer: dict = field(default_factory=dict)
        prompt_processor_type: str = "stable-diffusion-guidance"
        prompt_processor: dict = field(default_factory=dict)
        guidance_type: str = "stable-diffusion-guidance"
        guidance: dict = field(default_factory=dict)
        condition_data_path: str = "data/meta.h5"

        latent_steps: int = 2500

    cfg: Config

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,2]**2
        z = np.sqrt(xy + xyz[:,1]**2)
        theta = np.arctan2(xyz[:,1], np.sqrt(xy)) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,2], xyz[:,0])
        return np.array([-theta, -azimuth, z]).astype(np.float32)
    
    def configure(self):
        self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        )
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )
        cond_data = h5py.File(self.cfg.condition_data_path, "r")
        self.cond_rgb = cond_data["rgba"][:]
        self.cond_camera = cond_data["frames"][:]
        cond_state = np.random.randint(0, self.cond_rgb.shape[0])
        self.cond_rgb = self.cond_rgb[cond_state]
        self.cond_camera = self.cond_camera[cond_state]
        input_im = Image.fromarray(self.cond_rgb)
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3] * 2 - 1
        self.cond_rgb = torch.from_numpy(input_im.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        self.cond_camera = self.cartesian_to_spherical(self.cond_camera[None, ..., 3])
        self.automatic_optimization = False

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_normal=True, render_rgb=False)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        """
        Initialize guidance and prompt processor in this hook:
        (1) excluded from optimizer parameters (this hook executes after optimizer is initialized)
        (2) only used in training
        To avoid being saved to checkpoints, see on_save_checkpoint below.
        """
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        # self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
        #     self.cfg.prompt_processor
        # )
        # initialize SDF
        # FIXME: what if using other geometry types?
        self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        loss = 0.0

        out = self(batch)
        camera = np.array([batch["azimuth"], batch["elevation"], batch["camera_distances"]])
        # text_embeddings = self.prompt_processor(**batch)

        # if self.true_global_step < self.cfg.latent_steps:
        #     guidance_inp = torch.cat(
        #         [, out["opacity"]], dim=-1
        #     )
 
        # else:
        #     guidance_inp = out["comp_normal"] * 2.0 - 1.0
        #     guidance_out = self.guidance(
        #         guidance_inp, text_embeddings, rgb_as_latents=False
        #     )

        guidance_out = self.guidance(
            out["comp_normal"] * 2.0 - 1.0, camera, self.cond_rgb, self.cond_camera
        )
        loss += guidance_out["sds"] * self.C(self.cfg.loss.lambda_sds)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch_idx}.png",
            [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch_idx}.png",
            [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
        )
