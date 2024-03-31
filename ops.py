"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from ..init import ones, zeros
import numpy as np

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        return out_grad / b, out_grad * (-a) * (b ** (-2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def _full_axes(self, a):
        n_dims = len(a.shape)
        x, y = self.axes if self.axes else (n_dims - 2, n_dims - 1)
        full_axes = [i for i in range(n_dims)]
        full_axes[x], full_axes[y] = full_axes[y], full_axes[x]
        return tuple(full_axes)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.permute(a, self._full_axes(a))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def _get_summed_dims(self, old_shape, new_shape):
        if len(old_shape) == 0:
            return tuple(range(len(new_shape)))

        broadcasted = [True for _ in new_shape]
        for old_idx in reversed(range(len(old_shape))):
            new_idx = old_idx + len(new_shape) - len(old_shape)

            if old_shape[old_idx] == new_shape[new_idx]:
                broadcasted[new_idx] = False
                continue

            assert old_shape[old_idx] == 1, "Should never happen"

        return tuple(i for i, is_broadcast in enumerate(broadcasted) if is_broadcast)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        summed_dims = self._get_summed_dims(a.shape, out_grad.shape)
        return out_grad.sum(summed_dims).reshape(a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = (axes,) if isinstance(axes, int) else axes

    def _full_axes(self, shape):
      n_dims = len(shape)

      if not self.axes:
        return tuple(range(n_dims))

      return tuple(axis if axis >= 0 else axis + n_dims for axis in self.axes)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        restored_shape = tuple(1 if i in self._full_axes(a.shape) else dim for i, dim in enumerate(a.shape))
        return broadcast_to(out_grad.reshape(restored_shape), a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):

    def _align_shape(self, x, y):
      if x.shape == y.shape:
        return x

      summed_axes = tuple(range(len(x.shape) - len(y.shape)))
      return x.sum(summed_axes)

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        a_grad = out_grad @ b.transpose()
        b_grad = a.transpose() @ out_grad

        a_grad = self._align_shape(a_grad, a)
        b_grad = self._align_shape(b_grad, b)

        assert a.shape == a_grad.shape
        assert b.shape == b_grad.shape

        return a_grad, b_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raw = node.realize_cached_data().copy()
        raw[raw > 0] = 1
        return out_grad * Tensor(raw)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        one = ones(*out_grad.shape, device=out_grad.device, dtype=out_grad.dtype)
        return out_grad * (one - tanh(a) ** 2.0)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class TensorGetItem(TensorOp):
  def __init__(self, idxs: Tuple[slice]):
      self.idxs = idxs

  def process_slice(self, sl, dim, shape):
      """Convert a slice to an explicit start/stop/step"""
      start, stop, step = sl.start, sl.stop, sl.step
      if start == None:
          start = 0
      if start < 0:
          start = shape[dim]
      if stop == None:
          stop = shape[dim]
      if stop < 0:
          stop = shape[dim] + stop
      if step == None:
          step = 1

      # we're not gonna handle negative strides and that kind of thing
      assert stop > start, "Start must be less than stop"
      assert step > 0, "No support for  negative increments"
      return slice(start, stop, step)

  def get_validated_idxs(self, a):
      idxs = self.idxs

      if not isinstance(idxs, tuple):
          idxs = (idxs,)

      idxs = tuple(
          [
              self.process_slice(s, i, a.shape) if isinstance(s, slice) else slice(s, s + 1, 1)
              for i, s in enumerate(idxs)
          ]
      )

      assert len(idxs) == len(a.shape), "Need indexes equal to number of dimensions"
      return idxs

  def compute(self, a):
      idxs = self.get_validated_idxs(a)
      return a[idxs].compact()

  def gradient(self, out_grad, node):
      a = node.inputs[0]
      idxs = self.get_validated_idxs(a)

      # NOTE: only support first-order differentiation for now
      grad_np = np.zeros(a.shape)
      grad_np[idxs] = out_grad.numpy()
      grad = Tensor(grad_np, device=out_grad.device, dtype=out_grad.dtype, requires_grad=True)

      return grad


def get_item(a, idxs):
    return TensorGetItem(idxs)(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0, "Stack needs at least one array!"
        shape = args[0].shape

        for a in args:
            assert shape == a.shape, "All arrays need to be of the same size!"

        n = len(args)
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        out = array_api.empty(new_shape, device=args[0].device)
        slices = [slice(0, s) for s in new_shape]

        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arr

        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(0, s) for s in A.shape]
        splits = []

        for i in range(n):
            slices[self.axis] = slice(i, i+1)
            splits.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(splits)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        idxs = [slice(None, None, 1) for _ in a.shape]

        for axis in self.axes:
            new_shape[axis] *= 1 + self.dilation
            idxs[axis] = slice(None, None, 1 + self.dilation)

        arr = NDArray.make(tuple(new_shape), device=a.device)
        arr.fill(0.0)

        arr[tuple(idxs)] = a
        return arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        idxs = [slice(None, None, 1) for _ in a.shape]

        for axis in self.axes:
            idxs[axis] = slice(None, None, 1 + self.dilation)

        return a[tuple(idxs)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION

def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        if self.padding > 0:
            A = A.pad(axes=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        N, H, W, C_in = A.shape
        K_H, K_W, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        inner_dim = K_H * K_W * C_in
        H_out = (H - K_H) // self.stride + 1
        W_out = (W - K_W) // self.stride + 1

        A = A.as_strided(
            shape=(N, H_out, W_out, K_H, K_W, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
        ).compact().reshape((N * H_out * W_out, inner_dim))
        B = B.compact().reshape((inner_dim, C_out))
        out = A @ B

        return out.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # H_out = (H + 2 * padding - K) // stride + 1
        # W_out = (W + 2 * padding - K) // stride + 1
        # out_grad: (N, H_out, W_out, C_out)
        # A: (N, H, W, C_in)
        # B: (K, K, C_in, C_out)

        # H_out = H - K + 1
        # W_out = W - K + 1
        # out_grad: (N, H_out, W_out, C_out)
        # A: (N, H, W, C_in)
        # B: (K, K, C_in, C_out)

        A, B = node.inputs
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        B = flip(B, axes=(0, 1)) # (K, K, C_in, C_out)
        B = transpose(B) # (K, K, C_out, C_in)
        A_grad = conv(out_grad, B, stride=1, padding=K - self.padding - 1) # (N, H, W, C_in)

        A = transpose(A, axes=(0, 3)) # (C_in, H, W, N)
        out_grad = transpose(out_grad, axes=(0, 1)) # (H_out, N, W_out, C_out)
        out_grad = transpose(out_grad, axes=(1, 2)) # (H_out, W_out, N, C_out)

        B_grad = conv(A, out_grad, stride=1, padding=self.padding) # (C_in, K, K, C_out)
        B_grad = transpose(B_grad, axes=(0, 1)) # (K, C_in, K, C_out)
        B_grad = transpose(B_grad, axes=(1, 2)) # (K, K, C_in, C_out)

        assert A_grad.shape == node.inputs[0].shape, f"{A_grad.shape} {node.inputs[0].shape}"
        assert B_grad.shape == node.inputs[1].shape, f"{B_grad.shape} {node.inputs[1].shape}"

        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



