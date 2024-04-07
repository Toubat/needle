"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = (kernel_size - 1) // 2
        self.weight = Parameter(init.kaiming_uniform(
            in_channels * kernel_size * kernel_size, 
            out_channels * kernel_size * kernel_size,
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device, 
            dtype=dtype,
            requires_grad=True,
        ))

        interval = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
        self.bias = Parameter(init.rand(
            out_channels, 
            low=-interval, 
            high=interval, 
            device=device, 
            dtype=dtype,
            requires_grad=True,
        )) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor: # (N, C, H, W)
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) == 4 and x.shape[1] == self.in_channels

        x = ops.transpose(x, axes=(1, 2)) # (N, H, C, W)
        x = ops.transpose(x, axes=(2, 3)) # (N, H, W, C)
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.padding) # (N, H_out, W_out, C_out)

        if self.bias:
            N, H, W, C = out.shape
            out = out.reshape((N * H * W, C)) + self.bias.broadcast_to((N * H * W, C))
            out = out.reshape((N, H, W, C))

        out = ops.transpose(out, axes=(2, 3)) # (N, H_out, C_out, W_out)
        out = ops.transpose(out, axes=(1, 2)) # (N, C_out, H_out, W_out)
        return out
        ### END YOUR SOLUTION
