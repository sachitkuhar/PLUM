"""Weight quantization."""

import torch
import torch.nn as nn

import quant.binary.quantization as quantization


class WeightQuantizerSB(nn.Module):
    """
    Weight quantizer using Signed Binary.
    """

    def __init__(self, size: int) -> None:
        """Construct a weight quantizer using least squares with 1 bit."""
        super(WeightQuantizerSB, self).__init__()
        self.register_buffer('v1', torch.tensor([0.0] * size))

    def forward(self, w: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of quantizing weight using least squares 1 bit."""
        if self.training:
            v1, w_q = quantization.quantizer_sb(w)
            self.v1.copy_(v1)  # type: ignore
        else:
            _, w_q = quantization.quantizer_sb(w, self.v1)  # type: ignore
        return w_q
