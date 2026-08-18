
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """
    NAFNet SimpleGate:
    splits channels into two halves and multiplies them.
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Compact NAFNet-style restoration block.

    The block keeps the characteristic NAFNet ideas:
      - LayerNorm
      - channel expansion
      - depthwise convolution
      - SimpleGate
      - simplified channel attention
      - residual scaling
      - second feed-forward branch
    """

    def __init__(
        self,
        channels,
        dw_expand=2,
        ffn_expand=2,
        dropout=0.0,
    ):
        super().__init__()

        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = nn.LayerNorm(channels)

        self.conv1 = nn.Conv2d(
            channels,
            dw_channels,
            kernel_size=1,
            bias=True,
        )

        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True,
        )

        self.sg = SimpleGate()

        # Simplified channel attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channels // 2,
                dw_channels // 2,
                kernel_size=1,
                bias=True,
            ),
        )

        self.conv2 = nn.Conv2d(
            dw_channels // 2,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.beta = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

        self.norm2 = nn.LayerNorm(channels)

        self.conv3 = nn.Conv2d(
            channels,
            ffn_channels,
            kernel_size=1,
            bias=True,
        )

        self.sg2 = SimpleGate()

        self.conv4 = nn.Conv2d(
            ffn_channels // 2,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.gamma = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    @staticmethod
    def layer_norm_2d(x, norm):
        """
        Apply LayerNorm over channels while preserving NCHW.
        """
        x = x.permute(0, 2, 3, 1)
        x = norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):

        # -------------------------------------------------------
        # Branch 1
        # -------------------------------------------------------
        y = self.layer_norm_2d(x, self.norm1)

        y = self.conv1(y)
        y = self.dwconv(y)

        y = self.sg(y)

        y = y * self.sca(y)

        y = self.conv2(y)

        y = self.dropout1(y)

        x = x + y * self.beta

        # -------------------------------------------------------
        # Branch 2
        # -------------------------------------------------------
        y = self.layer_norm_2d(x, self.norm2)

        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)

        y = self.dropout2(y)

        x = x + y * self.gamma

        return x


class NAFNetX2(nn.Module):
    """
    REVIXEL Experiment B

    NAFNet-style 2× image restoration network.

    Supported:
        128×128  -> 256×256
        256×256  -> 512×512

    Input:
        (N, 1, H, W)

    Output:
        (N, 1, 2H, 2W)

    The network operates directly on the normalized LR image.
    Final output is residual-enhanced reconstruction.
    """

    def __init__(
        self,
        img_channel=1,
        width=32,
        enc_blk_nums=(2, 2),
        middle_blocks=4,
        dec_blk_nums=(2, 2),
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            img_channel,
            width,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # -------------------------------------------------------
        # Encoder
        # -------------------------------------------------------

        self.enc1 = nn.Sequential(
            *[
                NAFBlock(width)
                for _ in range(enc_blk_nums[0])
            ]
        )

        self.down1 = nn.Conv2d(
            width,
            width * 2,
            kernel_size=2,
            stride=2,
            bias=True,
        )

        self.enc2 = nn.Sequential(
            *[
                NAFBlock(width * 2)
                for _ in range(enc_blk_nums[1])
            ]
        )

        self.down2 = nn.Conv2d(
            width * 2,
            width * 4,
            kernel_size=2,
            stride=2,
            bias=True,
        )

        # -------------------------------------------------------
        # Middle
        # -------------------------------------------------------

        self.middle = nn.Sequential(
            *[
                NAFBlock(width * 4)
                for _ in range(middle_blocks)
            ]
        )

        # -------------------------------------------------------
        # Decoder
        #
        # IMPORTANT:
        # PixelShuffle converts:
        #
        # 128 channels -> 64 channels
        # 64 channels  -> 32 channels
        #
        # Therefore the skip tensors are already channel-compatible.
        # No reduce layers are required.
        # -------------------------------------------------------

        self.up2 = nn.Sequential(
            nn.Conv2d(
                width * 4,
                width * 8,
                kernel_size=1,
                bias=True,
            ),
            nn.PixelShuffle(2),
        )

        # 64 channels after PixelShuffle
        self.dec2 = nn.Sequential(
            *[
                NAFBlock(width * 2)
                for _ in range(dec_blk_nums[0])
            ]
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(
                width * 2,
                width * 4,
                kernel_size=1,
                bias=True,
            ),
            nn.PixelShuffle(2),
        )

        # 32 channels after PixelShuffle
        self.dec1 = nn.Sequential(
            *[
                NAFBlock(width)
                for _ in range(dec_blk_nums[1])
            ]
        )

        # -------------------------------------------------------
        # Final ×2 reconstruction
        # -------------------------------------------------------

        self.upscale = nn.Sequential(
            nn.Conv2d(
                width,
                width * 4,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.PixelShuffle(2),
        )

        self.ending = nn.Conv2d(
            width,
            img_channel,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x):

        # Preserve input for global residual.
        inp = x

        # -------------------------------------------------------
        # Encoder
        # -------------------------------------------------------

        x = self.intro(x)

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.down1(e1)
        )

        m = self.middle(
            self.down2(e2)
        )

        # -------------------------------------------------------
        # Decoder
        # -------------------------------------------------------

        u2 = self.up2(m)

        # u2: 64 channels
        # e2: 64 channels
        u2 = u2 + e2

        u2 = self.dec2(u2)

        u1 = self.up1(u2)

        # u1: 32 channels
        # e1: 32 channels
        u1 = u1 + e1

        u1 = self.dec1(u1)

        # -------------------------------------------------------
        # Final ×2
        # -------------------------------------------------------

        out = self.upscale(u1)

        out = self.ending(out)

        # Bicubic-like/global residual formulation:
        # interpolate original LR input to target resolution
        # and predict a restoration residual around it.
        base = F.interpolate(
            inp,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        out = base + out

        return out
