"""Utilities for initializing parameters.

Note: Code adapted from
https://github.com/google/jax/blob/main/jax/_src/nn/initializers.py.
"""

import math

from torch import nn


def pytorch_default_init(module: nn.Module) -> None:
  # Perform lecun_normal initialization.
  fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
  std = math.sqrt(1.0 / fan_in) / 0.87962566103423978
  nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
  if module.bias is not None:
    nn.init.constant_(module.bias, 0.0)
