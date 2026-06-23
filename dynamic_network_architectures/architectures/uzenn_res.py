"""
uzenn_res.py  —  ResEncUZenn: Residual Encoder + Zentropy Dual Decoder

Applies the U-ZENN Boltzmann combiner (from uzenn.py / uzenn_decoder.py)
to a residual encoder instead of a plain-conv encoder.

Relation to existing files
---------------------------
uzenn.py          PlainConvUZenn  — plain conv encoder  + UZennDecoder
uzenn_res.py      ResEncUZenn     — residual encoder    + UZennDecoder  ← NEW

The UZennDecoder (building_blocks/uzenn_decoder.py) already accepts
Union[PlainConvEncoder, ResidualEncoder] — no changes needed there.

The only structural difference from PlainConvUZenn is the encoder:
  PlainConvEncoder  →  ResidualEncoder
  n_conv_per_stage  →  n_blocks_per_stage  (number of residual blocks)
  (no new parameters in the decoder)

Usage in fourcases_v40.py
--------------------------
  from dynamic_network_architectures.architectures.uzenn_res import ResEncUZenn

  model = ResEncUZenn(
      input_channels=1,
      n_stages=6,
      features_per_stage=(32, 64, 128, 256, 320, 320),
      conv_op=nn.Conv2d,
      kernel_sizes=(3, 3, 3, 3, 3, 3),
      strides=(1, 2, 2, 2, 2, 2),
      n_blocks_per_stage=(1, 1, 1, 1, 1, 1),   # 1 residual block per stage
      num_classes=2,
      n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
      block=BasicBlockD,
      norm_op=nn.BatchNorm2d,
      nonlin=nn.ReLU,
      deep_supervision=False,
      T=5.0,
      gamma=5.0,
  )
"""

from typing import Union, Type, List, Tuple

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
)
from dynamic_network_architectures.building_blocks.residual_encoders import (
    ResidualEncoder,
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.uzenn_decoder import (
    UZennDecoder,
)
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He,
)


class ResEncUZenn(AbstractDynamicNetworkArchitectures):
    """
    Residual Encoder U-ZENN.

    Encoder  : ResidualEncoder  (same as ResidualEncoderUNet)
    Decoder  : UZennDecoder     (dual E/S branches, Boltzmann combiner)

    The Boltzmann combiner produces per-class logits:
        l_k = -softplus(E_k)/T + softplus(S_k) - (softplus(S_k)/γ)²

    Parameters
    ----------
    input_channels          : number of input image channels
    n_stages                : number of encoder/decoder resolution stages
    features_per_stage      : feature channels at each stage
    conv_op                 : nn.Conv2d or nn.Conv3d
    kernel_sizes            : conv kernel size(s) per stage
    strides                 : downsampling stride(s) per stage
    n_blocks_per_stage      : number of residual blocks per encoder stage
    num_classes             : number of segmentation output classes
    n_conv_per_stage_decoder: number of conv layers per decoder stage
    block                   : residual block type (default BasicBlockD)
    conv_bias               : use bias in convolutions
    norm_op                 : normalisation layer class (e.g. nn.BatchNorm2d)
    norm_op_kwargs          : kwargs for norm_op
    dropout_op              : dropout layer class (or None)
    dropout_op_kwargs       : kwargs for dropout_op
    nonlin                  : non-linearity class (e.g. nn.ReLU)
    nonlin_kwargs           : kwargs for nonlin
    deep_supervision        : if True, return list of predictions (multi-scale)
    nonlin_first            : if True: conv→nonlin→norm; else conv→norm→nonlin
    T                       : Boltzmann temperature (controls sharpness)
    gamma                   : Boltzmann entropy scale (controls uncertainty)
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        block: Type[nn.Module] = BasicBlockD,
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
        super().__init__()

        # Key paths used by TS2D pretrained weight loader and abstract_arch
        self.key_to_encoder = "encoder.stages"
        self.key_to_stem    = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",
        )

        # Normalise scalar → list arguments
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        assert len(n_blocks_per_stage) == n_stages, (
            f"n_blocks_per_stage must have {n_stages} entries, "
            f"got {len(n_blocks_per_stage)}: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            f"n_conv_per_stage_decoder must have {n_stages - 1} entries, "
            f"got {len(n_conv_per_stage_decoder)}: {n_conv_per_stage_decoder}"
        )

        # ── Residual encoder (same as ResidualEncoderUNet) ────────────
        # Note: ResidualEncoder does NOT accept nonlin_first — that argument
        # is specific to PlainConvEncoder. Block ordering is fixed in BasicBlockD.
        self.encoder = ResidualEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=block,
            return_skips=True,
        )

        # ── Zentropy dual decoder (E-branch + S-branch) ───────────────
        # UZennDecoder already accepts ResidualEncoder — no changes needed
        self.decoder = UZennDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            T=T,
            gamma=gamma,
            nonlin_first=nonlin_first,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "Give input_size without batch/channel dims, e.g. (128, 128) for 2D."
        )
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


# ======================================================================
# SELF-TEST  —  run with:  python uzenn_res.py
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ResEncUZenn self-test")
    print("=" * 60)

    # ── 2D, no deep supervision ───────────────────────────────────────
    data_2d = torch.rand((1, 1, 128, 128))
    model_2d = ResEncUZenn(
        input_channels=1,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3, 3, 3, 3),
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        num_classes=2,
        n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
        block=BasicBlockD,
        norm_op=nn.BatchNorm2d,
        nonlin=nn.ReLU,
        deep_supervision=False,
        T=5.0,
        gamma=5.0,
    )
    model_2d.initialize(model_2d)
    test_submodules_loadable(model_2d)

    out_2d = model_2d(data_2d)
    assert isinstance(out_2d, torch.Tensor), "Expected tensor output"
    assert out_2d.shape == (1, 2, 128, 128), f"Wrong shape: {out_2d.shape}"
    print(f"2D no-DS output shape : {tuple(out_2d.shape)}  ✓")

    # Softmax sums to 1
    p = torch.softmax(out_2d, dim=1)
    assert torch.allclose(p.sum(dim=1), torch.ones_like(p.sum(dim=1)), atol=1e-5)
    print("Softmax sums to 1     ✓")

    # Both E and S branches receive gradients
    target = torch.randint(0, 2, (1, 128, 128))
    loss = nn.functional.cross_entropy(out_2d, target)
    loss.backward()
    e_grad = next(model_2d.decoder.E_decoder.parameters()).grad
    s_grad = next(model_2d.decoder.S_decoder.parameters()).grad
    assert e_grad is not None and e_grad.abs().sum() > 0, "E branch: no gradient"
    assert s_grad is not None and s_grad.abs().sum() > 0, "S branch: no gradient"
    print(f"Gradient flows to E and S branches  ✓  CE loss: {loss.item():.4f}")

    # Parameter count
    total = sum(p.numel() for p in model_2d.parameters())
    print(f"Total parameters      : {total:,}")
    print(f"Feature map size      : "
          f"{model_2d.compute_conv_feature_map_size(data_2d.shape[2:])}")

    # ── 2D, deep supervision ──────────────────────────────────────────
    model_2d_ds = ResEncUZenn(
        input_channels=1,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3, 3, 3, 3),
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        num_classes=2,
        n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
        block=BasicBlockD,
        norm_op=nn.BatchNorm2d,
        nonlin=nn.ReLU,
        deep_supervision=True,
        T=5.0,
        gamma=5.0,
    )
    model_2d_ds.initialize(model_2d_ds)
    out_ds = model_2d_ds(data_2d)
    assert isinstance(out_ds, list), "Expected list for deep supervision"
    assert out_ds[0].shape == (1, 2, 128, 128), f"Wrong DS shape: {out_ds[0].shape}"
    print(f"\n2D deep-supervision shapes: {[tuple(o.shape) for o in out_ds]}  ✓")

    # ── 3D, no deep supervision ───────────────────────────────────────
    data_3d = torch.rand((1, 1, 64, 64, 64))
    model_3d = ResEncUZenn(
        input_channels=1,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=(3, 3, 3, 3, 3, 3),
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        num_classes=2,
        n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
        block=BasicBlockD,
        norm_op=nn.BatchNorm3d,
        nonlin=nn.ReLU,
        deep_supervision=False,
        T=5.0,
        gamma=5.0,
    )
    model_3d.initialize(model_3d)
    out_3d = model_3d(data_3d)
    assert out_3d.shape == (1, 2, 64, 64, 64), f"Wrong 3D shape: {out_3d.shape}"
    print(f"3D no-DS output shape : {tuple(out_3d.shape)}  ✓")

    print("\nAll tests passed.")
