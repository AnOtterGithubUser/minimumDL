import numpy as np

from nn import Module
import functional as F


class CrossEntropyLoss(Module):

    def __init__(self, seq):
        self._seq = seq

    def forward(self, outputs, labels):
        predictions = F.softmax(outputs)
        self._labels = labels
        self._predictions = predictions
        loss = np.dot(labels, predictions)
        return loss

    def backward(self):
        grad_loss = self._labels - self._predictions
        self._seq.backward(grad_loss)


