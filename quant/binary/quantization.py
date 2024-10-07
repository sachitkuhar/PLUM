"""Quantization functions and classes."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from quant.binary.ste import binarize, binary_sign


def clamp_identity(x: torch.Tensor) -> torch.Tensor:
    """Identity clamp."""
    return x


def clamp_symmetric(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Clamp x to [-alpha, +alpha]."""
    return x.clamp(-alpha, alpha)


class QuantizerFP(nn.Module):
    """Weight / activation quantizer using full precision."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of full-precision quantizer."""
        return x


def quantizer_sb(
    x: torch.Tensor, v1: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: A 4D tensor
        v1: A vector of scaling factors
    """
    v1 = torch.tensor([1]).to(x.device)
    return v1, v1.view(-1, 1, 1, 1) * binarize(x)
