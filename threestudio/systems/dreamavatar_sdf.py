from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("dreamavatar-sdf-system")
class DreamAvatarSDF(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # only used when refinement=True and from_coarse=True
        # not necessary, just for backward compatibility
        material_type: str = "no-material"  # unused
        material: dict = field(default_factory=lambda: {"n_output_dims": 0})
        background_type: str = "solid-color-background"  # unused
        background: dict = field(default_factory=dict)
        renderer_type: str = "nvdiff-rasterizer"
        renderer: dict = field(default_factory=dict)
        geometry_type: str = "tetrahedra_sdf_grid"
        geometry: dict = field(default_factory=dict)

        latent_steps: int = 1000
        refinement: bool = False
    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.geometry_guidance = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry_guidance)
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch, render_normal=True, render_rgb=False)
        else:
            render_out = self.renderer(**batch)
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        # self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
        #     self.cfg.prompt_processor
        # )
        # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        prompt_utils = self.prompt_processor()
        if self.cfg.stage == "geometry":
            guidance_out = self.guidance(
                out["comp_normal"], prompt_utils, **batch, rgb_as_latents=False
            )
        else:
            guidance_out = self.guidance(
                out["comp_rgb"], prompt_utils, **batch, rgb_as_latents=False
            )
        loss = 0.0
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        if self.cfg.stge == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.cfg.loss.lambda_regular:
                sdf = self.geometry.sdf[self.geometry.particle_index] + self.geometry.sdf_bias
                sdf_edges = sdf[self.geometry.isosurface_helper.all_edges].reshape(-1, 2)
                edge_masks = torch.sign(sdf_edges[:,0]) != torch.sign(sdf_edges[:,1])
                sdf_edges = sdf_edges[edge_masks]
                sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_edges[...,0], (sdf_edges[...,1] > 0).float()) + \
                torch.nn.functional.binary_cross_entropy_with_logits(sdf_edges[...,1], (sdf_edges[...,0] > 0).float())
                loss += self.C(self.cfg.loss.lambda_regular) * sdf_diff

            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))
        else:
            pass

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