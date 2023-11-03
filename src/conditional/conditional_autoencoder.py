from typing import List

import torch
from omegaconf import OmegaConf
from torch import nn

from ..encoder import Encoder
from ..gaussian_distribution import GaussianDistribution
from .conditional_decoder import ConditionalDecoder
from .conditional_encoder import ConditionalEncoder


class ConditionalAutoencoder(nn.Module):
    def __init__(
        self,
        emb_channels: int,
        z_channels: int,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            channels=channels,
            channel_multipliers=channel_multipliers,
            n_resnet_blocks=n_resnet_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
        )
        self.cond_decoder = ConditionalDecoder(
            channels=channels,
            channel_multipliers=channel_multipliers,
            n_resnet_blocks=n_resnet_blocks,
            out_channels=out_channels,
            z_channels=z_channels,
        )
        self.cond_encoder = ConditionalEncoder(
            channels=channels,
            channel_multipliers=channel_multipliers,
            n_resnet_blocks=n_resnet_blocks,
            in_channels=in_channels,
        )
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, img: torch.Tensor) -> GaussianDistribution:
        z = self.encoder(img)
        moments = self.quant_conv(z)
        return GaussianDistribution(moments)

    def encode_cond(self, cond: torch.Tensor) -> List[torch.Tensor]:
        conds_z = self.cond_encoder(cond)
        return conds_z

    def decode(self, z: torch.Tensor, conds_z: List[torch.Tensor]) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.cond_decoder(z, conds_z)

    @classmethod
    def load_from_saved(cls, pretrained_ckpt, pretrained_yaml, checkpoint_ckpt):
        pretrained_ckpt = torch.load(pretrained_ckpt)
        pretrained_yaml = OmegaConf.load(pretrained_yaml)
        checkpoint_ckpt = torch.load(checkpoint_ckpt)

        cond_autoencoder = cls(
            emb_channels=pretrained_yaml.params.embed_dim,
            z_channels=pretrained_yaml.params.ddconfig.z_channels,
            channels=pretrained_yaml.params.ddconfig.ch,
            channel_multipliers=pretrained_yaml.params.ddconfig.ch_mult,
            n_resnet_blocks=pretrained_yaml.params.ddconfig.num_res_blocks,
            in_channels=pretrained_yaml.params.ddconfig.in_channels,
            out_channels=pretrained_yaml.params.ddconfig.out_ch,
        )

        quant_conv_state_dict = {}
        post_quant_conv_state_dict = {}
        encoder_state_dict = {}
        cond_encoder_state_dict = {}
        cond_decoder_state_dict = {}

        for k, v in pretrained_ckpt["state_dict"].items():
            if k.startswith("quant_conv"):
                quant_conv_state_dict[k.replace("quant_conv.", "", 1)] = v
            elif k.startswith("post_quant_conv"):
                post_quant_conv_state_dict[k.replace("post_quant_conv.", "", 1)] = v
            elif k.startswith("encoder"):
                encoder_state_dict[k.replace("encoder.", "", 1)] = v
            elif k.startswith("decoder") or k.startswith("loss"):
                continue
            else:
                raise KeyError(f"Unexpected state_dict key: {k}")

        cond_encoder_state_dict = checkpoint_ckpt["cond_encoder_state_dict"]
        cond_decoder_state_dict = checkpoint_ckpt["cond_decoder_state_dict"]

        cond_autoencoder.quant_conv.load_state_dict(quant_conv_state_dict, strict=True)
        cond_autoencoder.post_quant_conv.load_state_dict(
            post_quant_conv_state_dict, strict=True
        )
        cond_autoencoder.encoder.load_state_dict(encoder_state_dict, strict=True)
        cond_autoencoder.cond_encoder.load_state_dict(
            cond_encoder_state_dict, strict=True
        )
        cond_autoencoder.cond_decoder.load_state_dict(
            cond_decoder_state_dict, strict=True
        )

        return cond_autoencoder
