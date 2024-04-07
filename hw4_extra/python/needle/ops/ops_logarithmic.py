from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION

        # return array_api.log(array_api.sum(array_api.exp(Z), axis=self.axes))


        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_exp = array_api.exp(Z - Z_max.broadcast_to(Z.shape))
        Z_sum = array_api.sum(Z_exp, axis=self.axes, keepdims=True)
        Z_log = array_api.log(Z_sum)
        
        assert Z_max.shape == Z_log.shape
        return (Z_log + Z_max).squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        
        Z_max = array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=True)
        Z_exp = exp(Z - Z_max.broadcast_to(Z.shape))

        Z_sum_exp = Z_exp.sum(axes=self.axes)
        Z_sum_exp = Z_sum_exp.reshape(Z_max.shape)
        Z_sum_exp = broadcast_to(Z_sum_exp, Z_exp.shape)

        out_grad = out_grad.reshape(Z_max.shape)
        out_grad = broadcast_to(out_grad, Z_exp.shape)

        return out_grad * Z_exp / Z_sum_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

