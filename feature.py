#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Optional, Union, Iterable, Any, Dict

import torch as th
nn = th.nn
F = nn.functional
import einops

# NOTE(ycho): adapted from the NLE baselines repo:
# torchbeast/models/baseline.py
from nle import nethack

NUM_GLYPHS = nethack.MAX_GLYPH
NUM_FEATURES = 25
PAD_CHAR = 0
NUM_CHARS = 256


class CBA(nn.Module):
    """Convolution + Batch-Norm + Activation.

    NOTE(ycho): by default, uses LeakyReLU activation.
    """

    def __init__(self,
                 c_in: int, c_out: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: Union[int, Tuple[int, int]] = 1,
                 stride: Union[int, Tuple[int, int]] = 1,
                 use_bn: bool = True,
                 lrelu_eps: Optional[float] = 1e-2,
                 use_bias: Optional[bool] = None
                 ):
        """

        Args:
            c_in: Number of input channels.
            c_out: Number of output channels.
            kernel_size: Convolution kernel size.
            padding: Convolution padding size.
            stride: Convolution stride.
            use_bn: Whether to use batch normalization. (nn.BatchNorm3d)
            lrelu_eps: nn.LeakyRelu slope parameter.
                If set to `None`, then skips lrelu activation.
            use_bias: Whether to use bias in convolution.
        """
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Conv2d(
            c_in, c_out,
            kernel_size,
            stride,
            padding,
            bias=(use_bias and (not use_bn))))
        if use_bn:
            layers.append(nn.BatchNorm2d(c_out))
        if lrelu_eps is not None:
            layers.append(nn.LeakyReLU(lrelu_eps, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)


class Residual(nn.Module):
    """CBA + Residual Block."""

    def __init__(self, base: nn.Module):
        self.base = base

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.base(x) + self.residual(x)


class GlyphEncoder(nn.Module):
    """Glyph (basically, the map/env) encoder."""

    def __init__(self):
        super().__init__()

        # Most naive implementation in the world
        self.f = nn.Sequential(*[
            CBA(1, 16, (3, 3), stride=(3, 3), padding=(0, 1)),
            CBA(16, 32, (3, 3), padding=0),
            CBA(32, 64, (3, 3), padding=0),
            nn.Flatten()
        ])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: int16 tensor of shape (..., 21, 79),
               in range (0, 5976).
        Returns:
            y: (..., D) feature embedding of `x`.
        """
        # NOTE(ycho): just directly converting to
        # a float is probably the worst possible idea.
        # But it works, for now.
        x_float = th.as_tensor(x,
                               dtype=th.float32) / 5976.0
        y = self.f(x_float)
        return y


class BlStatsEncoder(nn.Module):
    """Bottom-line statistics encoder."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(26, 128)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: int64 tensor of shape (..., 26)
        Returns:
            y: (..., D) feature embedding of `x`.
        """
        x = th.as_tensor(x, dtype=th.float32)
        return self.emb(x)


class MsgEncoder(nn.Module):
    """Message encoder."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Embedding(NUM_CHARS, 16),  # ..., 256x16
            nn.Flatten()  # ..., 4096
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: uint8 tensor of shape (..., 256),
               in range (0, 255).
        Returns:
            y: (..., D) feature embedding of `x`.
        """
        return self.emb(x.to(th.int32))


class NethackEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.glyph_encoder = GlyphEncoder()
        self.blstats_encoder = BlStatsEncoder()
        self.msg_encoder = MsgEncoder()

    def forward(self, x: Dict[str, th.Tensor]) -> th.Tensor:
        z_g = self.glyph_encoder(x['glyphs'])
        z_b = self.blstats_encoder(x['blstats'])
        z_m = self.msg_encoder(x['message'])
        z = th.cat((z_g, z_b, z_m), dim=1)
        # If you'd like, you can do more processing here.
        return z
