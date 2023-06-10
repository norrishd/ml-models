"""
residual building blocks
residual unit
identity mappings - skip connections

"""

from abc import ABC
from typing import Sequence, Union

import torch
from torch import nn


class ResNetUnit1D(nn.Module, ABC):
    """Abstract base class for 1D ResNet residual unit."""


class BasicUnit1D(ResNetUnit1D):
    """1D version of the basic residual unit for use in smaller ResNet models."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        # Both self.conv1 and self.projection_shortcut layers downsample the time dimension and
        # upsample the channel dimension when stride > 1
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1, stride=stride, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        if stride > 1:
            self.projection_shortcut = nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, 1, stride=stride, bias=bias),
                nn.BatchNorm1d(hidden_channels),
            )
        else:
            self.projection_shortcut = None

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Perform a forward pass through the basic residual learning block."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If the dimensionality is changed by conv1, transform the input to the same shape
        if self.projection_shortcut:
            identity = self.projection_shortcut(x)
        out += identity

        out = self.relu(out)

        return out


class Bottleneck1D(ResNetUnit1D):
    """1D version of the bottleneck residual unit designed to speed up training on larger ResNet
    models.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        out_channels = hidden_channels * 4

        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 1, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        # Both self.conv2 and self.projection_shortcut layers downsample the time dimension and
        # upsample the channel dimension when stride > 1
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels, 3, padding=1, stride=stride, bias=bias
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = nn.Conv1d(hidden_channels, out_channels, 1, bias=bias)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.projection_shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=bias),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.projection_shortcut = None

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Perform a forward pass through the BottleNeck block."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # If the dimensionality is changed by conv2, transform the input to the same shape
        if self.projection_shortcut:
            identity = self.projection_shortcut(x)
        out += identity

        out = self.relu(out)

        return out


class PreActivation1D(ResNetUnit1D):
    # TODO implement
    pass


class ResNet1D(nn.Module):
    """1D version of the ResNet architecture introduced in:

    "Deep Residual Learning for Image Recognition"
    https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        residual_unit: ResNetUnit1D,
        units_per_block: Sequence[int],
        channels_per_block: Sequence[int] = (64, 128, 256, 512),
        padding: Union[str, int] = "same",
        bias: bool = False,
        dilate_instead_of_stride: bool = True,
    ):
        """1D version of the ResNet architecture.

        Args:
            in_channels: Number of channels in the input data.
            num_classes: Number of output classes.
            residual_unit: Type of residual unit to use
            units_per_block: A sequence of ints indicating how many residual units should be
                included in each block.
            channels_per_block: A sequence of ints indicating how many channels / activation maps
                should be output from each block.
            dilate_instead_of_stride: TODO fill
        """
        super().__init__()

        # TODO add dilation option https://github.com/pytorch/vision/pull/866

        assert (
            len(units_per_block) == 4
        ), f"`blocks_per_section` should be a len-4 sequence of int; got {units_per_block}."
        self.residual_unit = residual_unit
        self.padding = padding
        self.bias = bias

        # Define the backbone model/encoder
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Each block (called "layers" in the paper) contains 2-6 residual units, with each residual
        # unit either containing 2 layers (basic blocks) or 3 layers (bottlenecks).
        # Each section starts by halving the temporal dimensionality with a max pool or stride-2
        # convolution, and thdoubling the number of filters with 1x1 convolutions, to maintain
        # roughly equal computational complexity per section
        self.in_channels = units_per_block[0]
        blocks = [self._make_block(units_per_block[0], channels_per_block[0], stride=1)]
        for units, features in zip(units_per_block[1:], channels_per_block[1:]):
            blocks.append(self._make_block(units, features, stride=2))
        self.blocks = nn.Sequential(*blocks)

    # TODO initialise weights

    def _make_block(self, num_blocks, features: int, stride: int = 2):
        """Create a block of residual units, which each have the same number of features.

        Typically the stride increases by a factor of 2 in each block

        Block 1 is preceded by a max pool that downsamples the temporal dimension. Subsequent
        blocks all downsample the time dimension using strided convolution in the first residual
        block, while doubling the number of filters to maintain computational complexity.

        Basic blocks have two kernel-3 layers. To reduce complexity in deeper networks, bottleneck
        blocks insert a layer to first reduce the number of filters, then perform the expensive
        kernel-3 convolution, and finally restore the number of filters in a third layer.
        """
        # First residual block may performing temporal downsampling and filter upsampling
        blocks = [self.residual_unit(self.in_channels, features, stride=stride)]

        # Update number of input channels applied to subsequent blocks and the next section
        if isinstance(self.residual_unit, Bottleneck1D):
            self.in_channels = features * 4
        else:
            self.in_channels = features

        # Subsequent residual blocks within a section don't downsample or stride
        for _ in range(num_blocks - 1):
            blocks.append(
                self.residual_unit(
                    self.in_channels,
                    features,
                    stride=1,
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.blocks(out)

        return out
