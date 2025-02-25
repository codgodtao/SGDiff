import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import logging
import inspect
from itertools import cycle
import clip
import math

from typing import Dict, Optional, Tuple
import wandb
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import scipy.io as scio

from tuneavideo.models.resnet import MOENetWork
from utils.metrics import SSIM_numpy, SAM_numpy
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DDPMScheduler,DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from tuneavideo.data.LRHR_dataset import MMDataset
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
import shutil

logger = get_logger(__name__, log_level="INFO")


def get_data_generator(loader):
    while True:
        for data in loader:
            yield data


def sample_data(dataloaders):
    dataloader = next(dataloaders)
    return next(dataloader)


def add_prefix(dct, prefix):
    return {f'{prefix}/{key}': val for key, val in dct.items()}


def main(
        pretrained_model_path: str,
        output_dir: str,
        train_data: Dict,
        validation_data: Dict,
        validation_steps: int = 10000,
        train_batch_size: int = 8,
        max_train_steps: int = 100000,
        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = True,
        checkpointing_steps: int = 500,
        stop_steps: int = 100000,
        resume_from_checkpoint: Optional[str] = None,
        enable_xformers_memory_efficient_attention: bool = True,
        seed: Optional[int] = None,
        dim: int=32,
        heads: list = [1, 2, 4, 8]
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)


    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    print(noise_scheduler)
    model_clip, _ = clip.load("ViT-L/14")
    model_clip.requires_grad_(False)
    unet = MOENetWork(inp_channels=3, out_channels=1, dim=32, model_clip=model_clip).cuda()
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    unet, optimizer,  = accelerator.prepare(
        unet, optimizer,
    )
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


    weight_dtype = torch.float32

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        path = os.path.basename(resume_from_checkpoint)
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        print(path)

    # Only show the progress bar once on each machine.

    def val(validation_pipeline, valid_dataloader, name, global_step=0):
        sam_val = ssim_val = 0.
        for idx, batch in enumerate(valid_dataloader):
            condition = batch["condition"].to(weight_dtype)
            MS = batch['MS'].to(weight_dtype)
            noise = torch.randn_like(batch['Res']).to(weight_dtype)
            Res = validation_pipeline(scene_ids = batch["scene_id"], grounding_ids = batch["grounding_id"], condition=condition, latents=noise,
                                      num_inference_steps=50).videos.squeeze(1)
            sample = np.transpose((Res + MS).clamp(0, 1).squeeze().float().numpy(), (1, 2, 0))
            HRMS = np.transpose(batch['HR'].squeeze().cpu().float().numpy(), (1, 2, 0))
            img_scale = 1023.0 if "GF2" in name else 2047.0
            os.makedirs(os.path.join(output_dir, name,str(global_step)), exist_ok=True)
            scio.savemat(os.path.join(output_dir, name, str(global_step), f'output_mulExm_{str(idx)}.mat'),
                         {"sr": sample * img_scale})  # H*W*C
            ssim_val += SSIM_numpy(sample, HRMS, 1)
            sam_val += SAM_numpy(sample, HRMS)
        ssim_val = float(ssim_val / valid_dataloader.__len__())
        sam_val = float(sam_val / valid_dataloader.__len__())
        wandb.log(add_prefix({f"ssim_{name}": ssim_val, f"sam_{name}": sam_val}, "train"), step=global_step)
        logger.info(f"ssim is {ssim_val},sam is {sam_val}")


    validation_pipeline = TuneAVideoPipeline(
        unet=unet,
        scheduler=noise_scheduler)
    valid_dataset = MMDataset(**validation_data['val_QB'])
    valid_dataset.encoding()
    valid_dataset2 = MMDataset(**validation_data['val_GF2'])
    valid_dataset2.encoding()
    valid_dataset3 = MMDataset(**validation_data['val_WV3'])
    valid_dataset3.encoding()
    valid_dataset4 = MMDataset(**validation_data['val_QB_full'])
    valid_dataset4.encoding()
    valid_dataset5 = MMDataset(**validation_data['val_GF2_full'])
    valid_dataset5.encoding()
    valid_dataset6 = MMDataset(**validation_data['val_WV3_full'])
    valid_dataset6.encoding()
    valid_dataset = MMDataset(**validation_data['val_WV2'])
    valid_dataset.encoding()
    val(validation_pipeline, torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False), "WV2")
    val(validation_pipeline, torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False), "QB")
    val(validation_pipeline, torch.utils.data.DataLoader(
        valid_dataset2, batch_size=1, shuffle=False), "GF2")
    val(validation_pipeline, torch.utils.data.DataLoader(
        valid_dataset3, batch_size=1, shuffle=False), "WV3")

    val(validation_pipeline, torch.utils.data.DataLoader(
        valid_dataset4, batch_size=1, shuffle=False), "QB_full")
    val(validation_pipeline, torch.utils.data.DataLoader(
        valid_dataset5, batch_size=1, shuffle=False), "GF2_full")
    val(validation_pipeline, torch.utils.data.DataLoader(
        valid_dataset6, batch_size=1, shuffle=False), "WV3_full")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/general_finetune.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
