import PIL
import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from pytorch_lightning import seed_everything, LightningModule
from torch import autocast
from contextlib import nullcontext
import accelerate
import mimetypes

mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

import k_diffusion as K
from ldm.util import instantiate_from_config


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False) -> LightningModule:
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def prepare_image(image: Image, width: int, height: int) -> torch.Tensor:
    image = image.convert("RGB")
    image = image.resize((width, height), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = 2. * image - 1.
    image = torch.from_numpy(image)
    return image


def prepare_mask(init_mask: Image, width: int, height: int) -> torch.Tensor:
    mask = init_mask.convert("L")
    mask = mask.resize((width // 8, height // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask = 1 - mask
    mask = torch.from_numpy(mask)
    return mask


# modification of k_diffusion.sampling.sample_lms
@torch.no_grad()
def _sample_lms_masked(model, x, x_orig, mask, noise, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = K.sampling.to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        cur_order = min(i + 1, order)
        coeffs = [K.sampling.linear_multistep_coeff(cur_order, sigmas.cpu(), i, j) for j in range(cur_order)]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
        x_orig_noised = x_orig + noise * sigmas[i]
        x = (x_orig_noised * mask) + (x * (1 - mask))
    return x


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.half().to(device)


def dream(prompt: str, ddim_steps: int, n_iter: int, n_samples: int, cfg_scale: float, seed: int, width: int,
          height: int):
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    rng_seed = seed_everything(seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())

    model_wrap = K.external.CompVisDenoiser(model)

    batch_size = n_samples
    assert prompt is not None
    data = [batch_size * [prompt]]
    seedit = 0

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    output_images = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling", disable=not accelerator.is_main_process):
                    for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                        uc = None
                        if cfg_scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, height // opt.f, width // opt.f]

                        sigmas = model_wrap.get_sigmas(ddim_steps)
                        torch.manual_seed(rng_seed + seedit)
                        x = torch.randn([n_samples, *shape], device=device) * sigmas[0]
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}
                        samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args,
                                                             disable=not accelerator.is_main_process)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = accelerator.gather(x_samples_ddim)

                        if accelerator.is_main_process:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                image = Image.fromarray(x_sample.astype(np.uint8))
                                output_images.append(image)
                                seedit += 1

    return output_images, rng_seed


def translation(prompt: str, init_img, ddim_steps: int, n_iter: int, n_samples: int, cfg_scale: float,
                denoising_strength: float, seed: int, width: int, height: int):
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    rng_seed = seed_everything(seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())
    model_wrap = K.external.CompVisDenoiser(model)

    batch_size = n_samples
    assert prompt is not None
    data = [batch_size * [prompt]]
    seedit = 0

    init_image = prepare_image(init_img, width, height)

    output_images = []
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            init_image = init_image.to(device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
            x0 = init_latent

            assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
            t_enc = int(denoising_strength * ddim_steps)
            print(f"target t_enc is {t_enc} steps")
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling", disable=not accelerator.is_main_process):
                    for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                        uc = None
                        if cfg_scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        sigmas = model_wrap.get_sigmas(ddim_steps)
                        torch.manual_seed(rng_seed + seedit)
                        noise = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1]
                        xi = x0 + noise
                        sigma_sched = sigmas[ddim_steps - t_enc - 1:]
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}
                        samples_ddim = K.sampling.sample_lms(model_wrap_cfg, xi, sigma_sched, extra_args=extra_args,
                                                             disable=not accelerator.is_main_process)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = accelerator.gather(x_samples_ddim)

                        if accelerator.is_main_process:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                image = Image.fromarray(x_sample.astype(np.uint8))
                                output_images.append(image)
                                seedit += 1

    return output_images, rng_seed


def inpainting(prompt: str, init_img, init_mask, ddim_steps: int, n_iter: int, n_samples: int, cfg_scale: float,
               seed: int, width: int, height: int):
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    rng_seed = seed_everything(seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())

    model_wrap = K.external.CompVisDenoiser(model)

    batch_size = n_samples
    assert prompt is not None
    data = [batch_size * [prompt]]
    seedit = 0

    init_image = prepare_image(init_img, width, height)
    mask = prepare_mask(init_mask, width, height)

    output_images = []
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            init_image = init_image.to(device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
            x0 = init_latent

            mask = mask.to(device)
            mask = repeat(mask, '1 ... -> b ...', b=batch_size)

            print(mask.shape)
            print(x0.shape)
            assert mask.shape == x0.shape, 'Mask shape should be same as input'

            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling", disable=not accelerator.is_main_process):
                    for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                        uc = None
                        if cfg_scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        sigmas = model_wrap.get_sigmas(ddim_steps)
                        torch.manual_seed(rng_seed + seedit)
                        noise = torch.randn_like(x0)
                        noise_i = noise * sigmas[0]
                        xi = x0 + noise_i
                        xi = (xi * mask) + (noise_i * (1 - mask))
                        sigma_sched = sigmas[0:]
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}
                        samples_ddim = _sample_lms_masked(model_wrap_cfg, xi, x0, mask, noise, sigma_sched,
                                                          extra_args=extra_args,
                                                          disable=not accelerator.is_main_process)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = accelerator.gather(x_samples_ddim)

                        if accelerator.is_main_process:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                image = Image.fromarray(x_sample.astype(np.uint8))
                                output_images.append(image)
                                seedit += 1
    return output_images, rng_seed


import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO


class Text2ImageRequest(BaseModel):
    prompt: str
    sampling_steps: int
    cfg: float
    width: int
    height: int
    seed: int


class Image2ImageRequest(BaseModel):
    prompt: str
    sampling_steps: int
    cfg: float
    denoising_strength: float
    width: int
    height: int
    seed: int
    base64_image: str


class InpaintingRequest(BaseModel):
    prompt: str
    sampling_steps: int
    cfg: float
    width: int
    height: int
    seed: int
    base64_image: str


class ImageResponse(BaseModel):
    seed: int
    base64_image: str


app = FastAPI()


def get_closest_divisible(x, divisor):
    a = x - x % divisor
    b = (x + divisor) - (x + divisor) % divisor
    return a if abs(x - a) < abs(x - b) and a > 0 else b


def get_img_mask_from_transparent_image(img):
    mask_img = ImageOps.invert(img.split()[-1])

    arr = np.array(img)
    arr[arr[:, :, 3] == 0] = 0
    arr = arr[:, :, 0:3]
    init_img = Image.fromarray(arr, 'RGB')
    return init_img, mask_img


@app.post("/txt2img", response_model=ImageResponse)
async def txt2img_api(request: Text2ImageRequest):
    output_width = get_closest_divisible(request.width, 64)
    output_height = get_closest_divisible(request.height, 64)
    output_images, rng_seed = dream(request.prompt, request.sampling_steps, 1, 1, request.cfg, request.seed,
                                    output_width, output_height)
    image = output_images[0]
    image = image.resize((request.width, request.height))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue())
    encoded_string = encoded_string.decode("utf-8")
    return ImageResponse(seed=rng_seed, base64_image=encoded_string)


@app.post("/img2img", response_model=ImageResponse)
async def img2img_api(request: Image2ImageRequest):
    output_width = get_closest_divisible(request.width, 64)
    output_height = get_closest_divisible(request.height, 64)
    init_img = Image.open(BytesIO(base64.b64decode(request.base64_image)), formats=['jpeg'])
    output_images, rng_seed = translation(request.prompt, init_img, request.sampling_steps, 1, 1, request.cfg,
                                          request.denoising_strength, request.seed, output_width, output_height)
    image = output_images[0]
    image = image.resize((request.width, request.height))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue())
    encoded_string = encoded_string.decode("utf-8")
    return ImageResponse(seed=rng_seed, base64_image=encoded_string)


@app.post("/inpainting", response_model=ImageResponse)
async def inpainting_api(request: InpaintingRequest):
    output_width = get_closest_divisible(request.width, 64)
    output_height = get_closest_divisible(request.height, 64)
    request_image = Image.open(BytesIO(base64.b64decode(request.base64_image)), formats=['png'])
    init_img, mask_img = get_img_mask_from_transparent_image(request_image)

    output_images, rng_seed = inpainting(request.prompt, init_img, mask_img, request.sampling_steps, 1, 1, request.cfg,
                                         request.seed, output_width, output_height)
    image = output_images[0]
    image = image.resize((request.width, request.height))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue())
    encoded_string = encoded_string.decode("utf-8")
    return ImageResponse(seed=rng_seed, base64_image=encoded_string)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
