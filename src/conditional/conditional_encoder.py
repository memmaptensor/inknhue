from typing import List

import torch
from torch import nn

from ..downsample import DownSample
from ..resnet_block import ResnetBlock
from ..utils import zero_module


class ConditionalEncoder(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        in_channels: int,
    ) -> None:
        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        n_resolutions = len(channel_multipliers)

        # Initial $3 \times 3$ convolution layer that maps the image to `channels`
        self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)

        # Number of channels in each top level block
        channels_list = [m * channels for m in [1] + channel_multipliers]

        # List of top-level blocks
        self.down = nn.ModuleList()
        self.proj = nn.ModuleList()

        # Create top-level blocks
        for i in range(n_resolutions):
            # Each top level block consists of multiple ResNet Blocks and down-sampling
            resnet_blocks = nn.ModuleList()

            # Add ResNet Blocks
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]

            # Top-level block
            down = nn.Module()
            down.block = resnet_blocks

            # Down-sampling at the end of each top level block except the last
            if i != n_resolutions - 1:
                down.downsample = DownSample(channels)
            else:
                down.downsample = nn.Identity()

            self.down.append(down)

            # Projection
            proj = nn.Conv2d(channels, channels, 1, 1, 0)
            proj = zero_module(proj)
            self.proj.append(proj)

    def forward(self, cond: torch.Tensor) -> List[torch.Tensor]:
        # Map to `channels` with the initial convolution
        x = self.conv_in(cond)

        conds_z = []

        # Top-level blocks
        for down, proj in zip(self.down, self.proj):
            # ResNet Blocks
            for block in down.block:
                x = block(x)

            conds_z.append(proj(x))

            # Down-sampling
            x = down.downsample(x)

        return conds_z
