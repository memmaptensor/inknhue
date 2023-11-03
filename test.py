import argparse
import gc
import logging
import os

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from rich.traceback import install
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm.auto import tqdm

from src.conditional.conditional_autoencoder import ConditionalAutoencoder
from src.conditional.conditional_test_dataset import ConditionalTestDataset
from src.utils import resize_max


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


@torch.no_grad()
def main(args):
    # Load configuration
    logging.info("Loading configuration")
    conf = OmegaConf.load(args.conf_path)

    # Create output directory
    logging.info("Creating output directory")
    os.makedirs(conf.paths.results_path, exist_ok=True)

    # Setup models
    logging.info("Setting up models")
    cond_autoencoder = ConditionalAutoencoder.load_from_saved(
        conf.paths.pretrained_ckpt,
        conf.paths.pretrained_yaml,
        conf.paths.conditional_ckpt,
    ).to(device="cuda", dtype=torch.bfloat16)
    cond_autoencoder.eval()

    # Load dataset & dataloader
    logging.info("Setting up Dataset and DataLoader")

    def transform(g, s):
        g, s = resize_max(g, conf.params.max_size), resize_max(s, conf.params.max_size)
        g = g.resize(
            (((g.size[0] + 7) // 8) * 8, ((g.size[1] + 7) // 8) * 8),
            Image.Resampling.LANCZOS,
        )
        s = s.resize(g.size, Image.Resampling.LANCZOS)

        pil_to_tensor = transforms.PILToTensor()
        g, s = pil_to_tensor(g), pil_to_tensor(s)
        g, s = (
            ((g / 255.0) * 2.0 - 1.0).clamp(-1, 1),
            ((s / 255.0) * 2.0 - 1.0).clamp(-1, 1),
        )

        g, s = g.to(device="cuda", dtype=torch.bfloat16), s.to(
            device="cuda", dtype=torch.bfloat16
        )

        return g, s

    cond_dataset = cond_dataset_full = ConditionalTestDataset(
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
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    # Testing
    logging.info("Start testing")
    i = 0
    with tqdm(cond_dataloader, unit="batch") as tepoch:
        for batch, (g, s) in enumerate(tepoch):
            z = cond_autoencoder.encode(s)
            z_sample = z.sample()
            conds_z = cond_autoencoder.encode_cond(g)
            y = cond_autoencoder.decode(z_sample, conds_z)

            for result in y.detach():
                img = ((result + 1.0) * 0.5 * 255.0).clamp(0, 255)
                img = rearrange(img, "c h w -> h w c")
                img = img.to(device="cpu", dtype=torch.uint8).numpy()
                img = Image.fromarray(img)

                path = f"{conf.paths.results_path}/{i:06}.png"
                img.save(path)

                i += 1

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install(show_locals=False)
    args = parse_args()
    main(args)
