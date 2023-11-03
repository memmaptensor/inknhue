import argparse
import copy
import gc
import logging
import os

import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.optim as optim
import torchvision.transforms.functional as VF
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from rich.traceback import install
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm.auto import tqdm

from src.conditional.conditional_dataset import ConditionalDataset
from src.conditional.conditional_decoder import ConditionalDecoder
from src.conditional.conditional_encoder import ConditionalEncoder
from src.encoder import Encoder
from src.gaussian_distribution import GaussianDistribution
from src.perceptual_loss import LPIPSWithDiscriminator
from src.utils import resize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_path",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Load configuration
    logging.info("Loading configuration")
    conf = OmegaConf.load(args.conf_path)

    # Create checkpoint directory
    logging.info("Creating checkpoint directory")
    os.makedirs(conf.paths.checkpoint_path, exist_ok=True)

    # Allow TF32
    logging.info("Allowing TF32")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load Accelerate
    logging.info("Setting up Accelerate")
    accelerator = Accelerator()

    # Load pretrained parameters
    logging.info("Loading pretrained checkpoints")
    pretrained_ckpt = torch.load(conf.paths.pretrained_ckpt)
    pretrained_yaml = OmegaConf.load(conf.paths.pretrained_yaml)

    # Load wandb
    logging.info("Setting up wandb")
    wandb.init(
        project=conf.wandb.project,
        config={
            "optimizer": conf.wandb.config.optimizer,
            "architecture": conf.wandb.config.architecture,
            "base_lr": conf.params.base_lr,
            "epoch": conf.params.epoch,
        },
    )

    # Load models
    logging.info("Setting up models")
    # Convolution to map from embedding space to quantized embedding space moments
    quant_conv = nn.Conv2d(
        2 * pretrained_yaml.params.ddconfig.z_channels,
        2 * pretrained_yaml.params.embed_dim,
        1,
    )
    # Convolution to map from quantized embedding space back to embedding space
    post_quant_conv = nn.Conv2d(
        pretrained_yaml.params.embed_dim,
        pretrained_yaml.params.ddconfig.z_channels,
        1,
    )
    encoder = Encoder(
        channels=pretrained_yaml.params.ddconfig.ch,
        channel_multipliers=pretrained_yaml.params.ddconfig.ch_mult,
        n_resnet_blocks=pretrained_yaml.params.ddconfig.num_res_blocks,
        in_channels=pretrained_yaml.params.ddconfig.in_channels,
        z_channels=pretrained_yaml.params.ddconfig.z_channels,
    )
    cond_encoder = ConditionalEncoder(
        channels=pretrained_yaml.params.ddconfig.ch,
        channel_multipliers=pretrained_yaml.params.ddconfig.ch_mult,
        n_resnet_blocks=pretrained_yaml.params.ddconfig.num_res_blocks,
        in_channels=pretrained_yaml.params.ddconfig.in_channels,
    )
    cond_decoder = ConditionalDecoder(
        channels=pretrained_yaml.params.ddconfig.ch,
        channel_multipliers=pretrained_yaml.params.ddconfig.ch_mult,
        n_resnet_blocks=pretrained_yaml.params.ddconfig.num_res_blocks,
        out_channels=pretrained_yaml.params.ddconfig.out_ch,
        z_channels=pretrained_yaml.params.ddconfig.z_channels,
    )
    discriminator = LPIPSWithDiscriminator(
        disc_start=pretrained_yaml.params.lossconfig.disc_start,
        disc_weight=pretrained_yaml.params.lossconfig.disc_weight,
        kl_weight=pretrained_yaml.params.lossconfig.kl_weight,
    )

    # Setup flags
    logging.info("Setting up flags")
    quant_conv.requires_grad_(False)
    post_quant_conv.requires_grad_(False)
    encoder.requires_grad_(False)
    cond_encoder.train()
    cond_decoder.train()
    discriminator.train()

    # Load state_dicts
    logging.info("Loading state_dicts")
    quant_conv_state_dict = {}
    post_quant_conv_state_dict = {}
    encoder_state_dict = {}
    cond_encoder_state_dict = {}
    cond_decoder_state_dict = {}
    loss_state_dict = {}

    for k, v in pretrained_ckpt["state_dict"].items():
        if k.startswith("quant_conv"):
            quant_conv_state_dict[k.replace("quant_conv.", "", 1)] = v
        elif k.startswith("post_quant_conv"):
            post_quant_conv_state_dict[k.replace("post_quant_conv.", "", 1)] = v
        elif k.startswith("encoder"):
            encoder_state_dict[k.replace("encoder.", "", 1)] = v
            if not (
                k.startswith("encoder.mid")
                or k.startswith("encoder.norm_out")
                or k.startswith("encoder.conv_out")
            ):
                cond_encoder_state_dict[k.replace("encoder.", "", 1)] = copy.deepcopy(v)
        elif k.startswith("decoder"):
            cond_decoder_state_dict[k.replace("decoder.", "", 1)] = v
        elif k.startswith("loss"):
            loss_state_dict[k.replace("loss.", "", 1)] = v
        else:
            raise KeyError(f"Unexpected state_dict key: {k}")

    quant_conv.load_state_dict(quant_conv_state_dict, strict=True)
    post_quant_conv.load_state_dict(post_quant_conv_state_dict, strict=True)
    encoder.load_state_dict(encoder_state_dict, strict=True)
    cond_encoder.load_state_dict(cond_encoder_state_dict, strict=False)
    cond_decoder.load_state_dict(cond_decoder_state_dict, strict=True)
    discriminator.load_state_dict(loss_state_dict, strict=True)

    # Load dataset & dataloader
    logging.info("Setting up Dataset and DataLoader")

    def transform(g, s, c):
        g, s, c = (
            resize(g, conf.params.size),
            resize(s, conf.params.size),
            resize(c, conf.params.size),
        )

        i, j, h, w = transforms.RandomCrop.get_params(
            img=g,
            output_size=(
                conf.params.crop_size,
                conf.params.crop_size,
            ),
        )
        g, s, c = VF.crop(g, i, j, h, w), VF.crop(s, i, j, h, w), VF.crop(c, i, j, h, w)

        pil_to_tensor = transforms.PILToTensor()
        g, s, c = pil_to_tensor(g), pil_to_tensor(s), pil_to_tensor(c)
        g, s, c = (
            ((g / 255.0) * 2.0 - 1.0).clamp(-1, 1),
            ((s / 255.0) * 2.0 - 1.0).clamp(-1, 1),
            ((c / 255.0) * 2.0 - 1.0).clamp(-1, 1),
        )

        return g, s, c

    cond_dataset = cond_dataset_full = ConditionalDataset(
        dataset_path=conf.paths.dataset_path, transform=transform
    )
    if not conf.params.use_entire_dataset:
        if conf.params.use_sequential_dataset:
            cond_dataset_indices = np.arange(conf.params.dataset_size)
        else:
            cond_dataset_indices = np.random.choice(
                len(cond_dataset), conf.params.dataset_size, replace=False
            )
        cond_dataset = Subset(cond_dataset_full, cond_dataset_indices)

    cond_dataloader = DataLoader(
        dataset=cond_dataset,
        batch_size=conf.params.batch_size,
        num_workers=min(12, conf.params.batch_size * 2),
        shuffle=True,
    )

    # Setup optimizers
    logging.info("Setting up optimizers")
    if conf.params.unlock_decoder:
        optimizer_g = optim.AdamW(
            list(cond_encoder.parameters()) + list(cond_decoder.parameters()),
            lr=conf.params.base_lr,
            betas=tuple(conf.params.betas),
        )
    else:
        optimizer_g = optim.AdamW(
            cond_encoder.parameters(),
            lr=conf.params.base_lr,
            betas=tuple(conf.params.betas),
        )
    optimizer_d = optim.AdamW(
        discriminator.parameters(),
        lr=conf.params.base_lr,
        betas=tuple(conf.params.betas),
    )

    # Training
    logging.info("Start training")

    def do_log():
        logging.info(f"Logging to wandb for global step {global_step}")
        colored = wandb.Image(((c[0] + 1.0) * 0.5).clamp(0, 1))
        grayscale = wandb.Image(((g[0] + 1.0) * 0.5).clamp(0, 1))
        style2paints = wandb.Image(((s[0] + 1.0) * 0.5).clamp(0, 1))
        reconstruction = wandb.Image(((y[0] + 1.0) * 0.5).clamp(0, 1))
        step_log.update(
            {
                "colored": colored,
                "grayscale": grayscale,
                "style2paints": style2paints,
                "reconstruction": reconstruction,
            }
        )
        wandb.log(step_log)

    def do_checkpoint():
        logging.info(f"Saving checkpoint for epoch {epoch}")
        accelerator.wait_for_everyone()
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "rec_loss": step_log["rec_loss"],
                "cond_encoder_state_dict": accelerator.get_state_dict(cond_encoder),
                "cond_decoder_state_dict": accelerator.get_state_dict(cond_decoder),
                "discriminator_state_dict": accelerator.get_state_dict(discriminator),
                "optimizer_g_state_dict": accelerator.get_state_dict(optimizer_g),
                "optimizer_d_state_dict": accelerator.get_state_dict(optimizer_d),
            },
            f"{conf.paths.checkpoint_path}/model_epoch{epoch:06}.ckpt",
        )

    (
        quant_conv,
        post_quant_conv,
        encoder,
        cond_encoder,
        cond_decoder,
        discriminator,
        cond_dataloader,
        optimizer_g,
        optimizer_d,
    ) = accelerator.prepare(
        quant_conv,
        post_quant_conv,
        encoder,
        cond_encoder,
        cond_decoder,
        discriminator,
        cond_dataloader,
        optimizer_g,
        optimizer_d,
    )
    global_step = 0
    for epoch in range(conf.params.epoch):
        with tqdm(
            cond_dataloader, unit="batch", disable=not accelerator.is_local_main_process
        ) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # Should checkpoint?
            should_checkpoint = (
                epoch % conf.params.checkpoint_epochs == 0
                or epoch == conf.params.epoch - 1
            ) and accelerator.is_local_main_process

            step_log = {}
            for batch, (g, s, c) in enumerate(tepoch):
                # Should log?
                should_log = (
                    global_step % conf.params.log_steps == 0 or batch == len(tepoch) - 1
                ) and accelerator.is_local_main_process

                # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
                pre_z = encoder(s)

                # Get the moments in the quantized embedding space
                moments = quant_conv(pre_z)

                # Get the distribution
                z = GaussianDistribution(moments)
                z_sample = z.sample()

                # Encode grayscale input to residuals
                conds_z = cond_encoder(g)

                # Map to embedding space from the quantized representation
                post_z = post_quant_conv(z_sample)

                # Decode the image of shape `[batch_size, channels, height, width]`
                y = cond_decoder(post_z, conds_z)

                # Compute loss and optimize [`ConditionalEncoder`, `ConditionalDecoder`, `LPIPSWithDiscriminator`]
                for idx, optimizer in enumerate([optimizer_g, optimizer_d]):
                    loss, log = discriminator(
                        inputs=s,
                        reconstructions=y,
                        posteriors=z,
                        optimizer_idx=idx,
                        global_step=global_step,
                        last_layer=cond_decoder.conv_out.weight,
                        cond=c,
                    )

                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    optimizer.step()

                    if should_log:
                        step_log.update(log)

                # Log
                if should_log:
                    tepoch.set_postfix(rec_loss=step_log["rec_loss"])
                    do_log()

                # Update global_step
                if accelerator.is_local_main_process:
                    global_step += 1

            # Checkpoint
            if should_checkpoint:
                do_checkpoint()

    # Clean up
    wandb.finish()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install(show_locals=False)
    args = parse_args()
    main(args)
