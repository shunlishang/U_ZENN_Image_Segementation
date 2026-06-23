import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder


class UZennDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision: bool,
                 T: float = 1.0,
                 gamma: float = 1.0,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 ):
        """
        U-ZENN decoder: two independent U-Net decoders sharing the same encoder skips.
        One produces per-class energy E, the other per-class entropy S. Both are passed
        through softplus to enforce non-negativity, then combined into per-class logits

            l_k = -softplus(E_k) / T + softplus(S_k) - (softplus(S_k) / gamma)^2

        which corresponds to p_k proportional to exp(-F_k / (k_B T) - (S_k / (k_B gamma))^2)
        with F_k = E_k - T S_k and k_B = 1.

        The shape contract matches UNetDecoder: a single (B, K, ...) tensor when
        deep_supervision is False, or a list of tensors (largest resolution first)
        when deep_supervision is True.
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.T = T
        self.gamma = gamma

        self.E_decoder = UNetDecoder(
            encoder, num_classes, n_conv_per_stage, deep_supervision,
            nonlin_first=nonlin_first,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias,
        )
        self.S_decoder = UNetDecoder(
            encoder, num_classes, n_conv_per_stage, deep_supervision,
            nonlin_first=nonlin_first,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias,
        )

    def _combine(self, E_raw: torch.Tensor, S_raw: torch.Tensor) -> torch.Tensor:
        E = F.softplus(E_raw)
        S = F.softplus(S_raw)
        return -E / self.T + S - (S / self.gamma) ** 2

    def forward(self, skips):
        E_out = self.E_decoder(skips)
        S_out = self.S_decoder(skips)
        if self.deep_supervision:
            return [self._combine(e, s) for e, s in zip(E_out, S_out)]
        return self._combine(E_out, S_out)

    def compute_conv_feature_map_size(self, input_size):
        return (
            self.E_decoder.compute_conv_feature_map_size(input_size)
            + self.S_decoder.compute_conv_feature_map_size(input_size)
        )
