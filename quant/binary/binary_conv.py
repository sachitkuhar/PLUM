from collections import defaultdict
from functools import partial
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import quant.binary.quantization as quantization
import quant.binary.weight_quantization as weight_quantization


class QuantConv2d(nn.Conv2d):
    """
    2D convolution based on scaled binary quantization.

    This performs `Conv2d(w_quant(w), x_quant(clamp(x))` where Conv2d is the regular 2D convolution.
    """

    def __init__(
        self,
        x_quant: str,
        w_quant: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        clamp: Optional[Dict] = None,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
        **kwargs: Any,
    ):
        """
        Construct a QuantConv2d instance.

        Args:
            x_quant: quantization scheme for activations
            w_quant: quantization scheme for weights
            clamp: clamping scheme for activations
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: size of convolving kernel
        """
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

        if clamp is None:
            clamp = {'kind': 'identity'}

        self.x_approximate = self._get_x_quantizer(
            x_quant, moving_average_mode, moving_average_momentum)
        self.w_approximate = self._get_w_quantizer(w_quant, out_channels)

        self.clamping_fn = self._get_clamper(**clamp)

        self.quantized_parameters: Dict[str, List[torch.Tensor]] = defaultdict(list)
        if self.bias is not None:
            self.quantized_parameters['fp'].append(self.bias)
        self.quantized_parameters[w_quant].append(self.weight)

    @staticmethod
    def _validate_scheme(scheme: str) -> None:
        if scheme not in {'fp', 'sb'} and not re.fullmatch(r'gf-\d+', scheme):
            raise ValueError(f'Scheme {scheme} is invalid. Please see docs for valid schemes.')

    @staticmethod
    def _get_x_quantizer(
        scheme: str,
        moving_average_mode: str = 'off',
        moving_average_momentum: float = 0.99,
    ) -> nn.Module:
        """Get activation quantizer from quantizer scheme."""
        QuantConv2d._validate_scheme(scheme)

        if scheme == 'fp':
            return quantization.QuantizerFP()
        else:
            raise Exception("not supported")

    @staticmethod
    def _get_w_quantizer(scheme: str, size: int) -> nn.Module:
        """Get weight quantizer function from quantizer scheme."""
        QuantConv2d._validate_scheme(scheme)

        if scheme == 'fp':
            return quantization.QuantizerFP()
        elif scheme.startswith('sb'):
            quantizer_map = {
                'sb': weight_quantization.WeightQuantizerSB,
            }
            return quantizer_map[scheme](size)

    @staticmethod
    def _get_clamper(
        kind: str, alpha: float = 2
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get clamping function from kind of clamping function."""
        try:
            clamper_map: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
                'identity': quantization.clamp_identity,
                'symmetric': partial(quantization.clamp_symmetric, alpha=alpha),
            }
            return clamper_map[kind]
        except KeyError:
            raise ValueError(f"{kind} is not a valid clamping function.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of this layer."""
        x_q = self.x_approximate(self.clamping_fn(x))
        w_q = self.w_approximate(self.weight)
        return F.conv2d(
            input=x_q,
            weight=w_q,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
