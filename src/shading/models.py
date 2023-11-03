import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor import BottleneckX_Origin, SEResNeXt_Origin

"""https://github.com/orashi/AlacGAN/blob/master/models/standard.py"""


class Selayer(nn.Module):
    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class ResNeXtBottleneck(nn.Module):
    def __init__(
        self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1
    ):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.conv_conv = nn.Conv2d(
            D,
            D,
            kernel_size=2 + stride,
            stride=stride,
            padding=dilate,
            dilation=dilate,
            groups=cardinality,
            bias=False,
        )
        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module("shortcut", nn.AvgPool2d(2, stride=2))

        self.selayer = Selayer(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.selayer(bottleneck)

        x = self.shortcut.forward(x)
        return x + bottleneck


class Generator(nn.Module):
    def __init__(self, ngf=64):
        super(Generator, self).__init__()

        self.encoder = SEResNeXt_Origin(
            BottleneckX_Origin, [3, 4, 6, 3], num_classes=370, input_channels=1
        )

        self.to0 = self._make_encoder_block_first(5, 32)
        self.to1 = self._make_encoder_block(32, 64)
        self.to2 = self._make_encoder_block(64, 92)
        self.to3 = self._make_encoder_block(92, 128)
        self.to4 = self._make_encoder_block(128, 256)

        self.deconv_for_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, 3, stride=2, padding=1, output_padding=1
            ),  # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),  # output is 128 * 128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                64, 32, 3, stride=1, padding=1, output_padding=0
            ),  # output is 256 * 256
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                32, 3, 3, stride=1, padding=1, output_padding=0
            ),  # output is 256 * 256
            nn.Tanh(),
        )

        tunnel4 = nn.Sequential(
            *[ResNeXtBottleneck(512, 512, cardinality=32, dilate=1) for _ in range(20)]
        )

        self.tunnel4 = nn.Sequential(
            nn.Conv2d(1024 + 128, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            tunnel4,
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
        )  # 64

        depth = 2
        tunnel = [
            ResNeXtBottleneck(256, 256, cardinality=32, dilate=1) for _ in range(depth)
        ]
        tunnel += [
            ResNeXtBottleneck(256, 256, cardinality=32, dilate=2) for _ in range(depth)
        ]
        tunnel += [
            ResNeXtBottleneck(256, 256, cardinality=32, dilate=4) for _ in range(depth)
        ]
        tunnel += [
            ResNeXtBottleneck(256, 256, cardinality=32, dilate=2),
            ResNeXtBottleneck(256, 256, cardinality=32, dilate=1),
        ]
        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            tunnel3,
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
        )  # 128

        tunnel = [
            ResNeXtBottleneck(128, 128, cardinality=32, dilate=1) for _ in range(depth)
        ]
        tunnel += [
            ResNeXtBottleneck(128, 128, cardinality=32, dilate=2) for _ in range(depth)
        ]
        tunnel += [
            ResNeXtBottleneck(128, 128, cardinality=32, dilate=4) for _ in range(depth)
        ]
        tunnel += [
            ResNeXtBottleneck(128, 128, cardinality=32, dilate=2),
            ResNeXtBottleneck(128, 128, cardinality=32, dilate=1),
        ]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(
            nn.Conv2d(128 + 256 + 64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            tunnel2,
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
        )

        tunnel = [ResNeXtBottleneck(64, 64, cardinality=16, dilate=1)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=2)]
        tunnel += [ResNeXtBottleneck(64, 64, cardinality=16, dilate=4)]
        tunnel += [
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=2),
            ResNeXtBottleneck(64, 64, cardinality=16, dilate=1),
        ]
        tunnel1 = nn.Sequential(*tunnel)

        self.tunnel1 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            tunnel1,
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
        )

        self.exit = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        )

    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, sketch):
        x0 = self.to0(sketch)
        aux_out = self.to1(x0)
        aux_out = self.to2(aux_out)
        aux_out = self.to3(aux_out)

        x1, x2, x3, x4 = self.encoder(sketch[:, 0:1])

        out = self.tunnel4(torch.cat([x4, aux_out], 1))

        x = self.tunnel3(torch.cat([out, x3], 1))

        x = self.tunnel2(torch.cat([x, x2, x1], 1))

        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))

        decoder_output = self.deconv_for_decoder(out)

        return x, decoder_output


class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()

        self.generator = Generator()

    def forward(self, x):
        fake, guide = self.generator(x)
        return fake, guide
