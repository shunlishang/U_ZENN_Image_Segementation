from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.uzenn_decoder import UZennDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class PlainConvUZenn(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        T: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin

        T, gamma: fixed scalar hyperparameters of the U-ZENN Boltzmann combiner.
        See UZennDecoder for the formula.
        """
        super().__init__()

        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",
        )

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, (
            "n_conv_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )
        self.decoder = UZennDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            T=T,
            gamma=gamma,
            nonlin_first=nonlin_first,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


if __name__ == "__main__":
    data = torch.rand((1, 4, 128, 128, 128))

    model = PlainConvUZenn(
        input_channels=4,
        n_stages=6,
        features_per_stage=(32, 64, 125, 256, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2, 2),
        n_conv_per_stage=(2, 2, 2, 2, 2, 2),
        num_classes=4,
        n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
        conv_bias=False,
        norm_op=nn.BatchNorm3d,
        nonlin=nn.ReLU,
        deep_supervision=True,
        T=1.0,
        gamma=1.0,
    )
    model.initialize(model)
    test_submodules_loadable(model)

    out = model(data)
    assert isinstance(out, list)
    assert out[0].shape == (1, 4, 128, 128, 128)
    print("3D deep_supervision shapes:", [tuple(o.shape) for o in out])
    print("3D feature map size:", model.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 4, 256, 256))
    model = PlainConvUZenn(
        input_channels=4,
        n_stages=7,
        features_per_stage=(32, 64, 128, 256, 512, 512, 512),
        conv_op=nn.Conv2d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2, 2, 2),
        n_conv_per_stage=2,
        num_classes=2,
        n_conv_per_stage_decoder=2,
        conv_bias=False,
        norm_op=nn.BatchNorm2d,
        nonlin=nn.ReLU,
        deep_supervision=False,
        T=1.5,
        gamma=2.0,
    )
    model.initialize(model)
    test_submodules_loadable(model)

    out = model(data)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 256, 256)
    print("2D no-DS output shape:", tuple(out.shape))

    p = torch.softmax(out, dim=1)
    assert torch.allclose(p.sum(dim=1), torch.ones_like(p.sum(dim=1)), atol=1e-5)

    target = torch.randint(0, 2, (1, 256, 256))
    loss = nn.functional.cross_entropy(out, target)
    loss.backward()

    e_grad = next(model.decoder.E_decoder.parameters()).grad
    s_grad = next(model.decoder.S_decoder.parameters()).grad
    assert e_grad is not None and e_grad.abs().sum() > 0, "E branch received no gradient"
    assert s_grad is not None and s_grad.abs().sum() > 0, "S branch received no gradient"
    print("Gradients flow into both E and S branches. CE loss:", loss.item())
