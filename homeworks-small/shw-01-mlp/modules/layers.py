import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features,
                                        in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / \
            np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        linear = input @ self.weight.T
        output = linear + self.bias if self.bias is not None else linear
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """

        return grad_output @ self.weight

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += grad_output.T @ input
        if self.grad_bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
            f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            B = input.shape[0]

            # считаем mean и var
            self.mean = input.mean(axis=0)
            self.input_mean = input - self.mean
            self.var = np.power(self.input_mean, 2).mean(axis=0)

            # считаем output
            self.sqrt_var = np.sqrt(self.var + self.eps)
            self.inv_sqrt_var = 1 / self.sqrt_var
            self.norm_input = self.input_mean * self.inv_sqrt_var

            # накапливаем статистики
            self.running_mean = (1 - self.momentum) * \
                self.running_mean + self.momentum * self.mean

            self.running_var = (1 - self.momentum) * \
                self.running_var + self.momentum * B / (B-1) * \
                self.var
        else:
            self.input_mean = input - self.running_mean
            self.sqrt_var = np.sqrt(self.running_var + self.eps)
            self.inv_sqrt_var = 1 / self.sqrt_var
            self.norm_input = self.input_mean * self.inv_sqrt_var

        if self.affine:
            output = self.norm_input * self.weight + self.bias
        else:
            output = self.norm_input

        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """

        if self.affine:
            grad_norm = grad_output * self.weight
        else:
            grad_norm = grad_output

        if self.training:

            output = grad_norm * self.inv_sqrt_var - \
                grad_norm.mean(axis=0) * self.inv_sqrt_var - self.input_mean * \
                np.power(self.inv_sqrt_var, 3) * \
                (grad_norm * self.input_mean).mean(axis=0)

        else:
            output = grad_norm * self.inv_sqrt_var

        return output

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.affine:
            self.grad_weight += (grad_output * self.norm_input).sum(axis=0)
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, 'f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """

    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            self.mask = np.random.binomial(n=1, p=1-self.p, size=input.shape)
            output = input * self.mask / (1 - self.p)
        else:
            output = input
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            output = grad_output * self.mask / (1 - self.p)
        else:
            output = grad_output
        return output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """

    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        for module in self.modules:
            input = module.forward(input)
        return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        modules = self.modules[::-1]
        len_modules = len(modules)
        for ind, module in enumerate(modules, start=0):
            if ind != len_modules - 1:
                grad_output = module.backward(
                    modules[ind+1].output, grad_output)
            else:
                grad_output = module.backward(input, grad_output)
        return grad_output

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str


class LowRankLinear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """

    def __init__(self, in_features: int, out_features: int, max_rank: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v = np.random.uniform(-1, 1, (max_rank,
                                           in_features)) / np.sqrt(in_features)
        self.u = np.random.uniform(-1, 1, (out_features,
                                           max_rank)) / np.sqrt(max_rank)

        self.bias = np.random.uniform(-1, 1, out_features) / \
            np.sqrt(in_features) if bias else None

        self.grad_u = np.zeros_like(self.u)
        self.grad_v = np.zeros_like(self.v)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        linear = input @ self.v.T @ self.u.T
        output = linear + self.bias if self.bias is not None else linear
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """

        return grad_output @ self.u @ self.v

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        hidden = input @ self.v.T

        self.grad_u += grad_output.T @ hidden
        self.grad_v += (grad_output @ self.u).T @ input

        if self.grad_bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        self.grad_u.fill(0)
        self.grad_v.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.u, self.v, self.bias]

        return [self.u, self.v]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_u, self.grad_v, self.grad_bias]

        return [self.grad_u, self.grad_v]

    def __repr__(self) -> str:
        out_features, in_features = self.u.shape[0], self.v.shape[1]
        return f'LowRankLinearLinear(in_features={in_features}, out_features={out_features}, ' \
            f'bias={not self.bias is None})'
