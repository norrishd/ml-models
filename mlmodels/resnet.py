from abc import ABC

import torch
from torch import nn


class ResNetBlock(nn.Module, ABC):
    """Abstract base class for ResNet activation blocks."""


class BasicBlock(ResNetBlock):
    pass


class BottleneckBlock(ResNetBlock):
    pass


class PreActivationBlock(ResNetBlock):
    pass


class ResNet(nn.Module):
    pass
