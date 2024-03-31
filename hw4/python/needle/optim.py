"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            if w.grad is None:
                continue

            grad = w.grad.data + max(self.weight_decay, 0.) * w.data
            self.u[w] = self.momentum * self.u.get(w, 0.) + (1 - self.momentum) * grad

            w.data = ndl.Tensor(
              w.data - self.lr * self.u[w],
              dtype=w.data.dtype,
              decive=w.data.device,
            )
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            raw = param.grad.realize_cached_data()
            raw[raw > max_norm] = max_norm
            raw[raw < -max_norm] = -max_norm
            param.grad.data = ndl.Tensor(raw, dtype=param.grad.dtype, device=param.grad.device)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
            if w.grad is None:
                continue

            grad = w.grad.data + max(self.weight_decay, 0.) * w.data
            self.u[w] = self.beta1 * self.u.get(w, 0.) + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v.get(w, 0.) + (1 - self.beta2) * (grad ** 2)

            u_hat = self.u[w] / (1 - self.beta1 ** self.t)
            v_hat = self.v[w] / (1 - self.beta2 ** self.t)

            w.data = ndl.Tensor(
              w.data - self.lr * u_hat / (v_hat ** 0.5 + self.eps),
              dtype=w.data.dtype,
              device=w.data.device,
            )
        ### END YOUR SOLUTION
