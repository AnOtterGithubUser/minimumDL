import numpy as np


class Optimizer:

    def __init__(self, seq, *args, **kwargs):
        self._seq = seq

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def step(self):
        for layer_id, layer in enumerate(self._seq.modules):
            layer._weight = self.update(layer._weight, layer._grad_weight, layer_id, 'weight')
            layer._bias = self.update(layer._bias, layer._grad_bias, layer_id, 'bias')


class SGD(Optimizer):

    def __init__(self, seq, lr=0.01, momentum=0):
        super().__init__(seq)
        self._lr = lr
        self._momentum = momentum
        self._state = {}

    def get_state(self, key):
        return self._state[key] if key in self._state else 0

    def set_state(self, key, value):
        self._state[key] = value if self._momentum != 0 else 0

    def update(self, parameter, grad_parameter, layer_id, weight_type):
        key = '{}_{}'.format(layer_id, weight_type)
        v = self.get_state(key)
        new_v = self._lr * grad_parameter + self._momentum * v
        new_parameter = parameter + new_v
        self.set_state(key, new_v)
        return new_parameter


class Adam(Optimizer):

    def __init__(self, seq, lr=0.01, betas=(0.9, 0.999), epsilon=10e-8):
        super().__init__(seq)
        self._lr = lr
        self._betas = betas
        self._epsilon = epsilon
        self._state = {}

    def get_state(self, key, grad_shape):
        return self._state[key] if key in self._state else np.zeros(grad_shape), np.zeros(grad_shape), 0

    def set_state(self, key, values):
        self._state[key] = values

    def update(self, parameter, grad_parameter, layer_id, weight_type):
        key = '{}_{}'.format(layer_id, weight_type)
        beta1, beta2 = self._betas
        v, s, n_it = self.get_state(key, grad_parameter.shape)
        v = beta1 * v + (1 - beta1) * grad_parameter
        s = beta2 * s + (1 - beta2) * grad_parameter ** 2
        n_it += 1
        self.set_state(key, (v, s, n_it))
        v_corrected = v / (1 - beta1 ** n_it)
        s_corrected = s / (1 - beta2 ** n_it)
        new_parameter = parameter + self._lr * (v_corrected / np.sqrt(s_corrected) + self._epsilon)
        return new_parameter


class RMSProp(Optimizer):

    def __init__(self, seq, lr, gamma, epsilon):
        super().__init__(seq)
        self._lr = lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._state = {}

    def get_state(self, key, grad_shape):
        return self._state[key] if key in self._state else np.zeros(grad_shape)

    def set_state(self, key, values):
        self._state[key] = values

    def update(self, parameter, grad_parameter, layer_id, weight_type):
        key = '{}_{}'.format(layer_id, weight_type)
        s = self.get_state(key, grad_parameter.shape)
        s = self._gamma * s + (1 - self._gamma) * grad_parameter ** 2
        self.set_state(key, s)
        new_parameter = parameter - grad_parameter * (self._lr / np.sqrt(s + self._epsilon))
        return new_parameter
