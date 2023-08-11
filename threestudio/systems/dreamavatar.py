from dataclasses import dataclass, field
import numpy as np
import cv2
import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from threestudio.utils.smpl_utils import zoom_bbox_in_apos

@threestudio.register("dreamavatar-system")
class DreamAvatar(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # only used when refinement=True and from_coarse=True
        geometry_type: str = "coarse-implicit-volume"
        geometry: dict = field(default_factory=dict)
        use_vsd: int = 1
        zoomable: int = 0
        part_stage: float = 10000
    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        if self.cfg.use_vsd:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils = self.prompt_processor()
            # self.renderer.training = True
        self.head_bbox = zoom_bbox_in_apos()
        self.focus_mode = ["head"]

    def forward(self, batch: Dict[str, Any], training=False) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_normal=True)
        part_render_out = None
        if self.cfg.zoomable and training:
            batch["rays_o"] = batch["rays_o_head"]
            batch["rays_d"] = batch["rays_d_head"]
            part_render_out = self.renderer(**batch, render_normal=False)
        render_dict = {**render_out}
        if self.cfg.zoomable and training:
            # todo: add more part factorized process.
            render_dict["comp_rgb_head"] = part_render_out["comp_rgb"]
        return render_dict

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        # self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
        #     self.cfg.prompt_processor
        # )
        # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        out = self(batch, training=True)
        if batch_idx == self.cfg.part_stage:
            self.cfg.use_vsd = 0
            self.guidance = self.sds_guidance
        loss = 0.0
        origin_prompt = self.prompt_processor.prompt
        # FB
        self.prompt_processor.prompt = "Full body photo of " + origin_prompt
        guidance_out = self.guidance(
            out["comp_rgb"], self.prompt_utils, **batch, rgb_as_latents=False
        )
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        if self.cfg.zoomable:
            # todo: add more part factorized process.
            self.prompt_processor.prompt = "Front headshot of " + origin_prompt
            with torch.no_grad():
                part_guidance_out = self.guidance(
                    out["comp_rgb_head"], self.prompt_utils, **batch, rgb_as_latents=False
                )
            # debug head part?
            cv2.imwrite("body.png", np.rint(out["comp_rgb"][0].detach().cpu().numpy() * 255))
            cv2.imwrite("head.png", np.rint(out["comp_rgb_head"][0].detach().cpu().numpy() * 255))
            
            for name, value in part_guidance_out.items():
                self.log(f"train/part_{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")]) * 0.2

        if not self.cfg.use_vsd:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal correlation loss, no normal is found in the output."
                )
            loss_normal = torch.mean((out["normal"] - out["shading_normal"]) ** 2)
            lambda_normal = torch.mean(torch.sqrt((1 - torch.exp(-out["density"])) ** 2))
            self.log("train/loss_normal", loss_normal * lambda_normal)
            loss += loss_normal * lambda_normal * self.cfg.loss.lambda_sparsity
        
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
        self.log("train/loss_z_variance", loss_z_variance)
        loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

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
            name="validation_step",
            step=self.true_global_step,
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
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
