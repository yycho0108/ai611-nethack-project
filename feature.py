#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Optional, Union, Iterable, Any, Dict, List

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

    def __init__(
        self, 
        c_in: int, 
        c_out: int, 
        padding: Union[int, Tuple[int, int]] = 1, 
        stride: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()
        self.conv1 = CBA(c_in, c_out, padding=padding, stride=stride)
        self.conv2 = CBA(c_out, c_out, lrelu_eps=None)
        self.residual = nn.Sequential()
        if c_in != c_out or stride != 1:
            self.residual = CBA(c_in, c_out, padding=padding, stride=stride, lrelu_eps=None)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.residual(x)
        out = F.leaky_relu(out)
        return out


class GlyphEncoder(nn.Module):
    """Glyph (basically, the map/env) encoder."""

    def __init__(self, in_channels: List[int], out_channels: List[int]):
        super().__init__()

        self.f = nn.Sequential(*[
            # -- [K x 79 x 21]
            Residual(in_channels[0], out_channels[0]),
            # -- [A x 27 x 7]
            Residual(in_channels[1], out_channels[1], padding=(1, 0), stride=3),
            # -- [A x 9 x 3]
            CBA(in_channels[2], out_channels[2], padding=(0, 1), stride=3),
            # -- [A x 27]
            nn.Flatten(start_dim=2)
        ])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: tensor of shape (..., embedding_dim, 21, 79).
        Returns:
            y: (..., A, 27) feature embedding of `x`.
        """
        x_float = th.as_tensor(x, dtype=th.float32)
        y = self.f(x_float)
        return y


class BlStatsEncoder(nn.Module):
    """Bottom-line statistics encoder."""

    def __init__(self, blstats_size: int, embedding_dim: int):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(blstats_size, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: int64 tensor of shape (..., 26)
        Returns:
            y: (..., E) feature embedding of `x`.
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


class NetHackEncoder(nn.Module):
    def __init__(
        self,
        observation_shape,
        device:th.device,
        use_lstm: bool = True,
        embedding_dim: int = 32,
    ):
        super(NetHackEncoder, self).__init__()
        self.device = device

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]
        self.use_lstm = use_lstm
        self.h_dim = 512
        self.attention_dim = 64

        in_channels = [embedding_dim, embedding_dim, self.attention_dim]
        out_channels = [embedding_dim, self.attention_dim, self.attention_dim]

        self.embed = nn.Embedding(NUM_GLYPHS, embedding_dim)
        self.blstats_encoder = BlStatsEncoder(self.blstats_size, embedding_dim)
        self.glyph_encoder = GlyphEncoder(in_channels, out_channels)
        self.attention_layer = nn.Linear(embedding_dim, self.attention_dim)

        self.fc = nn.Sequential(
            nn.Linear((embedding_dim + self.attention_dim), self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)

    def initial_state(self, batch_size=1, device:th.device=None):
        if not self.use_lstm:
            return tuple()
        return tuple(
            th.zeros(
                self.core.num_layers, batch_size, self.core.hidden_size,
                device=device)
            for _ in range(2))

    def _select(self, embed: nn.Embedding, x: th.Tensor):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs: Dict[str, th.Tensor], core_state, done):
        # -- [T x B x F]
        blstats = env_outputs["blstats"]
        blstats = th.as_tensor(blstats, device = self.device)

        T, B, _ = blstats.shape
        # -- [B' x F] (B' = T x B)
        blstats = blstats.view(T * B, -1).float()
        # -- [B' x E]
        blstats_rep = self.blstats_encoder(blstats)
        reps = [blstats_rep]

        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]
        glyphs = th.as_tensor(glyphs, device = self.device)
        # -- [B' x H x W] 
        glyphs = th.flatten(glyphs, 0, 1)
        glyphs = glyphs.long()
        # -- [B' x H x W x E]
        glyphs_emb = self._select(self.embed, glyphs)
        # -- [B' x E x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)
        # -- [B' x A x 27]
        glyphs_rep = self.glyph_encoder(glyphs_emb)

        # Attention
        attention_weight = F.softmax(
            th.einsum(
                'bi,bij->bj', 
                self.attention_layer(blstats_rep), 
                glyphs_rep
            ), dim=-1
        )
        reps.append(th.einsum('bij,bj->bi', glyphs_rep, attention_weight))

        # -- [B x (E + A)]
        st = st = th.cat(reps, dim=1)
        st = self.fc(st)

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~done).float() # -- [T x B]
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            # core_output = th.flatten(th.cat(core_output_list), 0, 1)
            core_output = th.cat(core_output_list, dim=0)
            # print('core_output', core_output.shape) # 1x1x16x512
        else:
            core_output = st

        return core_output, core_state
