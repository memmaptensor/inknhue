from typing import List

import torch
from torch import nn

from ..attn_block import AttnBlock
from ..resnet_block import ResnetBlock
from ..upsample import UpSample
from ..utils import normalization, swish


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        out_channels: int,
        z_channels: int
    ) -> None:
        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        num_resolutions = len(channel_multipliers)

        # Number of channels in each top level block, in the reverse order
        channels_list = [m * channels for m in channel_multipliers]

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # Initial $3 \times 3$ convolution layer that maps the embedding space to `channels`
        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        # ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # List of top-level blocks
        self.up = nn.ModuleList()

        # Create top-level blocks
        for i in reversed(range(num_resolutions)):
            # Each top level block consists of multiple ResNet Blocks and up-sampling
            resnet_blocks = nn.ModuleList()

            # Add ResNet Blocks
            for _ in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]

            # Top-level block
            up = nn.Module()
            up.block = resnet_blocks

            # Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upsample = UpSample(channels)
            else:
                up.upsample = nn.Identity()

            # Prepend to be consistent with the checkpoint
            self.up.insert(0, up)

        # Map to image space with a $3 \times 3$ convolution
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor, conds_z: List[torch.Tensor]) -> torch.Tensor:
        # Map to `channels` with the initial convolution
        h = self.conv_in(z)

        # ResNet blocks with attention
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Top-level blocks
        for up, cond_z in reversed(list(zip(self.up, conds_z))):
            # ResNet Blocks
            for block in up.block:
                h = block(h)

            h += cond_z

            # Up-sampling
            h = up.upsample(h)

        # Normalize and map to image space
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(h)

        return img
