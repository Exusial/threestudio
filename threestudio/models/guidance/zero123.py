from dataclasses import dataclass, field

from PIL import Image
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.ops import SpecifyGradient
from threestudio.utils.typing import *

from threestudio.models.guidance.ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from threestudio.models.guidance.ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model

@threestudio.register("zero123")
class Zero123Guidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "300000.ckpt"
        sd_conf: str = ""
        # FIXME: xformers error
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = True
        guidance_scale: float = 20.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        num_train_timesteps: int = 50
        weighting_strategy: str = "sds"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Zero-123 Model ...")

        # self.weights_dtype = (
        #     torch.float16 if self.cfg.half_precision_weights else torch.float32
        # )

        # Create models
        self.models = dict()
        sd_conf = OmegaConf.load(self.cfg.sd_conf)
        self.models['turncam'] = load_model_from_config(sd_conf, self.cfg.pretrained_model_name_or_path, device=self.device)
        # self.models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        # 'CompVis/stable-diffusion-safety-checker')

        for p in self.models['turncam'].parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        # self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
        #     self.device
        # )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Zero-123!")

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        camera: Float[Tensor, "B 3"],
        cond_rgb: Float[Tensor, "1 C H W"],
        cond_camera: Float[Tensor, "1 3"]
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        # cond_BCHW = cond_rgb.permute(0, 3, 1, 2)
        # assert rgb_as_latents == False, f"No latent space in {self.__class__.__name__}"
        rgb_BCHW = rgb_BCHW * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
        )

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        model = self.model["turncam"]
        c = model.get_learned_conditioning(cond_rgb)
        x, y, z = camera - cond_camera
        T = torch.tensor([x, math.sin(y), math.cos(y), z])
        c = torch.cat([c, T], dim=-1)
        c = model.cc_projection(c)
        cond = {}
        cond['c_crossattn'] = [c]
        c_concat = model.encode_first_stage((cond_rgb.to(self.device))).mode().detach()
        cond['c_concat'] = [model.encode_first_stage((cond_rgb.to(self.device))).mode().detach()]
        if scale != 1.0:
            uc = {}
            uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
            uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
        else:
            uc = None
        
        sampler = DDIMSampler(models['turncam'])
        sampler.make_schedule(self.max_step)
        # shape = [4, h // 8, w // 8]
        # samples_ddim, _ = sampler.sample(S=ddim_steps,
        #                                     conditioning=cond,
        #                                     batch_size=n_samples,
        #                                     shape=shape,
        #                                     verbose=False,
        #                                     unconditional_guidance_scale=scale,
        #                                     unconditional_conditioning=uc,
        #                                     eta=ddim_eta,
        #                                     x_T=None)
        # print(samples_ddim.shape)
        # # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
        # x_samples_ddim = model.decode_first_stage(samples_ddim)
        # return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = sampler.q_sample(x, t, noise)
            # pred noise
            noise_pred = sampler.get_sds_loss(latents_noisy, cond, t, self.cfg.guidance_scale)
            # latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            # noise_pred = self.forward_unet(
            #     latent_model_input,
            #     torch.cat([t] * 2),
            #     encoder_hidden_states=text_embeddings,
            # )  # (B, 6, 64, 64)

        # # perform guidance (high scale from paper!)
        # noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        # noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
        # noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        # noise_pred = noise_pred_text + self.cfg.guidance_scale * (
        #     noise_pred_text - noise_pred_uncond
        # )

        """
        # thresholding, experimental
        if self.cfg.thresholding:
            assert batch_size == 1
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
            noise_pred = custom_ddpm_step(self.scheduler,
                noise_pred, int(t.item()), latents_noisy, **self.pipe.prepare_extra_step_kwargs(None, 0.0)
            )
        """

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
        # latents.backward(grad, retain_graph=True)

        return {
            "sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)


"""
# used by thresholding, experimental
def custom_ddpm_step(ddpm, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, generator=None, return_dict: bool = True):
    self = ddpm
    t = timestep

    prev_t = self.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[t].item()
    alpha_prod_t_prev = self.alphas_cumprod[prev_t].item() if prev_t >= 0 else 1.0
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    noise_thresholded = (sample - (alpha_prod_t ** 0.5) * pred_original_sample) / (beta_prod_t ** 0.5)
    return noise_thresholded
"""
