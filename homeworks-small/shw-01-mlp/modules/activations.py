import numpy as np
from .base import Module
from scipy import special
from scipy import stats


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(0, input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        mask = input > 0
        return grad_output * mask.astype(float)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return special.expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * special.expit(input) * (1 - special.expit(input))


class GELU(Module):
    """
    Applies element-wise GELU function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return input * special.ndtr(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * (input * stats.norm.pdf(input) + special.ndtr(input))


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        maximum = input.max(axis=1, keepdims=True)
        exp_max = np.exp(input - maximum)

        return exp_max / exp_max.sum(axis=1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        maximum = input.max(axis=1, keepdims=True)
        exp_max = np.exp(input - maximum)

        prob = exp_max / exp_max.sum(axis=1, keepdims=True)

        return prob * (grad_output - (grad_output * prob).sum(axis=1, keepdims=True))


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """

        return input - special.logsumexp(input, axis=1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        maximum = input.max(axis=1, keepdims=True)
        exp_max = np.exp(input - maximum)

        prob = exp_max / exp_max.sum(axis=1, keepdims=True)
        return grad_output - prob * grad_output.sum(axis=1, keepdims=True)
