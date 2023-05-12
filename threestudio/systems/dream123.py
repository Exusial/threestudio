from dataclasses import dataclass, field

import torch
import h5py
import numpy as np
from PIL import Image

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("dream123")
class Dream123(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = "implicit-volume"
        geometry: dict = field(default_factory=dict)
        material_type: str = "diffuse-with-point-light-material"
        material: dict = field(default_factory=dict)
        background_type: str = "neural-environment-map-background"
        background: dict = field(default_factory=dict)
        renderer_type: str = "nerf-volume-renderer"
        renderer: dict = field(default_factory=dict)
        guidance_type: str = "stable-diffusion-guidance"
        guidance: dict = field(default_factory=dict)
        prompt_processor_type: str = "stable-diffusion-prompt-processor"
        prompt_processor: dict = field(default_factory=dict)
        condition_data_path: str = "data/meta.h5"
        
    cfg: Config

    # def cartesian_to_spherical(self, xyz):
    #     ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    #     xy = xyz[:,0]**2 + xyz[:,1]**2
    #     z = np.sqrt(xy + xyz[:,2]**2)
    #     theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #     #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    #     azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    #     return np.array([theta, azimuth, z])

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
        input_im.save(f"ref_image.png")
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0

        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3] * 2 - 1
        self.cond_rgb = torch.from_numpy(input_im.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        self.cond_camera = torch.Tensor(self.cartesian_to_spherical(self.cond_camera[None, ..., 3])).to(self.device)
        print(self.cond_camera)
        # self.automatic_optimization = False

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor, self.trainer
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        camera = [batch["azimuth"], batch["elevation"], batch["camera_distances"]]
        # input_im = torch.cat([( * 2.0 - 1.0), out["opacity"]], dim=-1)
        input_im = torch.cat([out["comp_rgb"], out["opacity"]], dim=-1)
        # print(input_im.max(), input_im.min)
        # print(input_im.shape)
        # text_embeddings = self.prompt_processor(**batch)
        # guidance_out = self.guidance(
        #     out["comp_rgb"], text_embeddings, rgb_as_latents=False
        # )
        guidance_out = self.guidance(
            input_im, camera, self.cond_rgb, self.cond_camera
        )
        loss = 0.0

        loss += guidance_out["sds"] * self.C(self.cfg.loss.lambda_sds)

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
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
