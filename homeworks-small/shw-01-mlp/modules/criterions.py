import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'

        return np.power(input-target, 2).mean()

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'

        return 2 * (input - target) / input.size


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        log_probs = self.log_softmax(input)
        true_log_probs = log_probs[np.arange(input.shape[0]), target]
        loss = - (1 - self.label_smoothing) * true_log_probs - \
            self.label_smoothing * log_probs.mean(axis=1)

        return loss.mean()

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        log_probs = self.log_softmax(input)
        probs = np.exp(log_probs)

        probs -= (self.label_smoothing / input.shape[1])
        probs[np.arange(input.shape[0]), target] -= 1 - self.label_smoothing

        return probs / input.shape[0]
