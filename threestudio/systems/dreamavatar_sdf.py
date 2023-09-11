from dataclasses import dataclass, field
import numpy as np
import cv2
import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from threestudio.utils.smpl_utils import zoom_bbox_in_apos

import torchvision.transforms as transforms
from threestudio.models.guidance.controlnet_guidance import ControlNetGuidance

@threestudio.register("dreamavatar-sdf-system")
class DreamAvatarSDF(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # only used when refinement=True and from_coarse=True
        geometry_type: str = "coarse-implicit-volume"
        geometry: dict = field(default_factory=dict)
        sds_guidance_type: str = "stable-diffusion-guidance"
        sds_guidance: dict = field(default_factory=dict)
        use_vsd: int = 1
        zoomable: int = 0
        part_stage: float = 10000
        fine_stage: float = 10000
        stage: str = "nerf"
    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        import ipdb
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        # if self.cfg.part_stage >= 0:
        #     self.sds_guidance = threestudio.find(self.cfg.sds_guidance_type)(self.cfg.sds_guidance)
        # self.renderer.training = True
        # self.head_bbox = zoom_bbox_in_apos()
        # self.focus_mode = ["head"]
        # self.stage = self.cfg.stage
        # self.convert_from = self.cfg.geometry_convert_from
        # if self.stage == "mesh" and self.cfg.use_vsd == 1:
        #     params = torch.load(self.convert_from, map_location="cpu")
        #     self.load_state_dict(params["state_dict"], strict=False)
        self.stage = self.cfg.stage
        print("pass configuration.")

    def forward(self, batch: Dict[str, Any], training=False) -> Dict[str, Any]:
        #import pdb; pdb.set_trace()
        if self.stage == "nerf":
            render_out = self.renderer(**batch, render_normal=True)
            part_render_out = None
            render_dict = {**render_out}
            if self.cfg.zoomable and training:
                if "rays_o_head" in batch:
                    batch["rays_o"] = batch["rays_o_head"]
                    batch["rays_d"] = batch["rays_d_head"]
                    part_render_out = self.renderer(**batch, render_normal=False)
                    render_dict["comp_rgb_head"] = part_render_out["comp_rgb"]
                if "rays_o_torso" in batch:
                    batch["rays_o"] = batch["rays_o_torso"]
                    batch["rays_d"] = batch["rays_d_torso"]
                    part_render_out = self.renderer(**batch, render_normal=False)
                    render_dict["comp_rgb_torso"] = part_render_out["comp_rgb"]
        else:
            render_out = self.renderer(**batch, render_normal=True)
            render_out["comp_rgb"] = render_out["comp_rgb"]
            part_render_out = None
            render_dict = {**render_out}
            if self.cfg.zoomable and training:
                if "rays_o_head" in batch:
                    batch["mvp_mtx"] = batch["c2w_head"]
                    batch["camera_positions"] = batch["camera_positions_head"]
                    part_render_out = self.renderer(**batch, render_normal=True)
                    render_dict["comp_rgb_head"] = part_render_out["comp_rgb"]
                if "rays_o_torso" in batch:
                    batch["mvp_mtx"] = batch["c2w_torso"]
                    batch["camera_positions"] = batch["camera_positions_torso"]
                    part_render_out = self.renderer(**batch, render_normal=True)
                    render_dict["comp_rgb_torso"] = part_render_out["comp_rgb"]
        return render_dict

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        # self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
        #     self.cfg.prompt_processor
        # )
        # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def change_rep(self):
        pass

    def training_step(self, batch, batch_idx):
        # if batch_idx == self.cfg.part_stage:
        #     self.cfg.use_vsd = 0
        #     del self.guidance 
        #     self.guidance = self.sds_guidance
        out = self(batch, training=True)
        loss = 0.0
        origin_prompt = self.prompt_processor.prompt
        # FB
        # self.prompt_processor.prompt = "Full body photo of " + origin_prompt
        #import pdb; pdb.set_trace()
        # rendered_openpose = torch.from_numpy(batch["rendered_openpose"].copy())
        rendered_openpose = batch["rendered_openpose"].unsqueeze(0)
        # import pdb; pdb.set_trace()
        if self.cfg.use_vsd:
            guidance_out = self.guidance(
                out["comp_rgb"], self.prompt_utils, **batch, rgb_as_latents=False, cond_rgb=rendered_openpose
            )
            pass
        elif self.cfg.guidance_type == "stable-diffusion-controlnet-guidance":
            guidance_out = self.guidance(
                out["comp_rgb"], rendered_openpose, self.prompt_utils, **batch
            )
        else:
            guidance_out = self.guidance(
                out["comp_rgb"], self.prompt_utils, **batch
            )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        if self.stage == "sdf":
            # todo: add normal calc.
            pass 
        else:
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )
            
            # if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
            #     loss_laplacian_smoothness = out["mesh"].laplacian()
            #     self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
            #     loss += loss_laplacian_smoothness * self.C(
            #         self.cfg.loss.lambda_laplacian_smoothness
            #     )            
            cv2.imwrite("body.png", np.rint(out["comp_rgb"][0].detach().cpu().numpy() * 255))
            pass
    
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
