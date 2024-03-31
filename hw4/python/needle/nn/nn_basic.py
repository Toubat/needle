"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
from functools import reduce
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
          init.kaiming_uniform(
            fan_in=in_features, 
            fan_out=out_features, 
            device=device, 
            dtype=dtype,
          )
        )
        self.bias = Parameter(
          init.kaiming_uniform(
            fan_in=out_features,
            fan_out=1,
            device=device, 
            dtype=dtype,
          ).reshape((1, out_features))
        ) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight # (n, out_features)

        if self.bias:
          y += ops.broadcast_to(self.bias, (*X.shape[:-1], self.out_features))
        
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        flattened_dim = reduce(lambda a, b: a * b, X.shape[1:])
        return X.reshape((X.shape[0], flattened_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules: List["Module"]):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        assert len(logits.shape) == 2 and len(y.shape) == 1
        assert logits.shape[0] == y.shape[0]
        
        n, k = logits.shape[0], logits.shape[1]
        log_sum_exp = ops.logsumexp(logits, axes=(1,)) # (n,)
        y_one_hot = init.one_hot(k, y, device=logits.device) # (n, k)
        softmax = log_sum_exp - (logits * y_one_hot).sum(axes=(1,))

        return softmax.sum(axes=0) / n
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) == 2 and x.shape[1] == self.dim

        n = x.shape[0]
        w = ops.broadcast_to(self.weight, (n, self.dim)) # (n, d)
        b = ops.broadcast_to(self.bias, (n, self.dim)) # (n, d)

        if self.training:
          mu = x.sum(axes=(0,)) / n
          self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * mu

          mu = ops.broadcast_to(mu, (n, self.dim))
          var = ((x - mu) ** 2).sum(axes=(0,)) / n
          self.running_var.data = (1 - self.momentum) * self.running_var + self.momentum * var

          std = ops.broadcast_to((var + self.eps) ** 0.5, (n, self.dim)) # (n, d)
          out = (x - mu) / std
          return w * out + b
        else:
          mu = self.running_mean # (d,)
          mu = ops.broadcast_to(mu, (n, self.dim)) # (n, d)

          var = self.running_var # (d,)
          std = ops.broadcast_to((var + self.eps) ** 0.5, (n, self.dim)) # (n, d)
          out = (x - mu) / std
          return w * out + b


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) == 2 and x.shape[1] == self.dim

        n = x.shape[0]
        w = ops.broadcast_to(self.weight, (n, self.dim)) # (n, d)
        b = ops.broadcast_to(self.bias, (n, self.dim)) # (n, d)

        mu = (x.sum(axes=(1,)) / self.dim).reshape((n, 1)) # (n, 1)
        mu = ops.broadcast_to(mu, (n, self.dim)) # (n, d)

        var = ((x - mu) ** 2).sum(axes=(1,)) / self.dim # (n,)
        std = ((var + self.eps) ** 0.5).reshape((n, 1)) # (n, 1)
        std = ops.broadcast_to(std, (n, self.dim)) # (n, d)

        return w * ((x - mu) / std) + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
          return x

        mask = init.randb(*x.shape, p=1-self.p, device=x.device)
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
