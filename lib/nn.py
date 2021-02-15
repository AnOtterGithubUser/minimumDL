import numpy as np
from im2col import im2col_indices, col2im_indices


class Module:

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def step(self, optimizer):
        self._bias = optimizer(self._bias, self._grad_bias)
        self._weight = optimizer(self._weight, self._grad_weight)


class Sequential(Module):

    def __init__(self, *modules):
        self._modules = list(modules)

    def forward(self, x):
        for layer in self._modules:
            x = layer(x)
        return x

    def backward(self, out_grad):
        for layer in self._modules[::-1]:
            out_grad = layer.backward(out_grad)


class FullyConnected(Module):

    def __init__(self, in_features, out_features):
        self._weight = np.random.uniform(size=(in_features, out_features))
        self._bias = np.random.uniform(size=(out_features,))

    def forward(self, x):
        self.x = x
        return np.dot(x.T, self._weight) + self._bias

    def backward(self, out_grad):
        self._grad_weight = np.outer(out_grad, self.x)
        self._grad_bias = out_grad
        grad_x = np.dot(out_grad, self._weight)
        return grad_x


class ReLU(Module):

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, out_grad):
        return out_grad * np.maximum(self.x, 0)


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._weight = np.random.uniform(size=(out_channels, in_channels, kernel_size, kernel_size))
        self._bias = np.random.uniform(size=(out_channels,))

    def forward(self, x):
        self.x = x
        n, c_in, h_in, w_in = x.shape
        n_filters, c_kernel, h_kernel, w_kernel = self._weight.shape
        h_out = int(((h_in + 2 * self._padding - self._stride * (h_kernel - 1) -1) / self._stride) + 1)
        w_out = int(((w_in + 2 * self._padding - self._stride * (w_kernel -1) - 1) / self._stride) + 1)

        x_col = im2col_indices(x, h_kernel, w_kernel, padding=self._padding, stride=self._stride)
        w_col = self._weight.reshape(n_filters, -1)

        out = w_col @ x_col + self._bias

        out = out.reshape(n_filters, h_out, w_out, n)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, out_grad):
        n_filters, c_kernel, h_kernel, w_kernel = self._weight.shape
        self._grad_bias = np.sum(out_grad, axis=(0, 2, 3))
        self._grad_bias = self._grad_bias.reshape(n_filters, -1)

        x_col = im2col_indices(self.x, h_kernel, w_kernel, padding=self._padding, stride=self._stride)

        out_grad_reshaped = out_grad.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        self._grad_weight = out_grad_reshaped @ x_col.T
        self._grad_weight = self._grad_weight.reshape(self._weight.shape)

        weight_reshaped = self._weight.reshape(n_filters, -1)
        grad_x_col = weight_reshaped @ out_grad_reshaped
        grad_x = col2im_indices(grad_x_col, self.x.shape, h_kernel, w_kernel, padding=padding, stride=stride)

        return grad_x
