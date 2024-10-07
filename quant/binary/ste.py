"""Straight-through estimator."""

from typing import Any, NewType

import torch
from torch.autograd import Function

BinaryTensor = NewType('BinaryTensor', torch.Tensor)  # A type where each element is in {-1, 1}


def binary_sign(x: torch.Tensor) -> BinaryTensor:
    """Return -1 if x < 0, 1 if x >= 0."""
    return x.sign() + (x == 0).type(torch.float)  # type: ignore


class STESign(Function):
    """
    Binarize tensor using sign function.

    Straight-Through Estimator (STE) is used to approximate the gradient of sign function.

    See:
    Bengio, Yoshua, Nicholas Léonard, and Aaron Courville.
    "Estimating or propagating gradients through stochastic neurons for
     conditional computation." arXiv preprint arXiv:1308.3432 (2013).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> BinaryTensor:  # type: ignore
        """
        Return a Sign tensor.

        Args:
            ctx: context
            x: input tensor

        Returns:
            Output type is float tensor where each element is either ±1 or 0.
        """
        p = x.clone()
        delta = (p.data.abs().max(1)[0].max(1)[0].max(1)[0] * 0.05).to(p.device)
        alpha_pos = p[:p.shape[0] // 2] >= delta[:p.shape[0] // 2, None, None, None]
        alpha_neg = p[p.shape[0] // 2:] <= -delta[p.shape[0] // 2:, None, None, None]
        alpha_ = torch.cat([alpha_pos, alpha_neg]).to(p.device)
        unitary_vals = torch.cat([torch.ones(p.data.shape[0] // 2), torch.ones(p.data.shape[0] - p.data.shape[0] // 2).mul(-1)]).to(p.device)
        ctx.save_for_backward(x, alpha_, unitary_vals)

        p[:p.shape[0] // 2][alpha_pos] = 1
        p[:p.shape[0] // 2][~alpha_pos] = 0
        p[p.shape[0] // 2:][alpha_neg] = -1
        p[p.shape[0] // 2:][~alpha_neg] = 0
        return p



    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore  # pragma: no cover (since this is called by C++ code) # noqa: E501
        """
        Compute gradient using STE.

        Args:
            ctx: context
            grad_output: gradient w.r.t. output of Sign

        Returns:
            Gradient w.r.t. input of the Sign function
        """
        ###FOR UTQ
        x, alpha_, unitary_vals = ctx.saved_tensors
        grad_input = grad_output.clone()

        temp = grad_input.data.mul((~alpha_).float())
        grad_input.data.mul_(alpha_.float()).mul_(unitary_vals[:, None, None, None].abs()).add_(temp)
        grad_input.data.clamp_(-1, 1)

        grad_input[x.gt(1)] = 0
        grad_input[x.lt(-1)] = 0
        return grad_input

# Convenience function to binarize tensors
binarize = STESign.apply    # type: ignore
