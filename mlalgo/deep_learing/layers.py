import numpy as np


class Activation:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        """Return the output of the activation function given input x."""
        raise NotImplementedError

    def backprop(self):
        """Return the derivative of the activation function given input x."""
        raise NotImplementedError


class Sigmoid(Activation):
    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backprop(self):
        return self.y * (1 - self.y)


class Softmax(Activation):
    def forward(self, x):
        self.x = x
        exp = np.exp(x - np.max(x))
        self.y = exp / np.sum(exp)
        return self.y

    def backprop(self):
        return self.y * (1 - self.y)


class ReLU(Activation):
    def forward(self, x):
        self.x = x
        self.y = np.where(x >= 0, x, 0)
        return self.y

    def backprop(self):
        return np.where(self.x >= 0, 1, 0)


class LeakyRelu(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        self.y = np.where(x >= 0, x, self.alpha * x)
        return self.y

    def backprop(self):
        return np.where(self.x >= 0, 1, self.alpha)


class Linear(Activation):
    def forward(self, x):
        self.x = x
        self.y = x
        return self.y

    def backprop(self):
        return np.ones_like(self.x)


class LossFunction:
    def __init__(self):
        self.y_pred = None
        self.y = None

    def loss(self, y_pred, y):
        """Return the loss given prediction y_pred and target y."""
        raise NotImplementedError

    def grad(self):
        """Return the gradient of the loss function."""
        raise NotImplementedError


class CrossEntropy(LossFunction):
    def loss(self, y_pred, y):
        self.y_pred = y_pred
        self.y = y
        return -np.sum(y * np.log(y_pred), keepdims=True)

    def grad(self):
        return -self.y / self.y_pred


class MSE(LossFunction):
    def loss(self, y_pred, y):
        self.y_pred = y_pred
        self.y = y
        return 0.5 * np.sum((y_pred - y) ** 2)

    def grad(self):
        return self.y_pred - self.y


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation()
        self.weights = np.random.uniform(
            -1 / np.sqrt(input_size),
            1 / np.sqrt(input_size),
            size=(output_size, input_size),
        )
        self.bias = np.random.uniform(
            -1 / np.sqrt(input_size), 1 / np.sqrt(input_size), size=(output_size, 1)
        )

    def forward(self, input):
        self.input = input
        self.output = self.activation.forward(self.weights @ input + self.bias)
        return self.output

    def backprop(self, input_gradient):
        if isinstance(self.activation, Softmax):
            input_gradient = input_gradient
        else:
            input_gradient = input_gradient * self.activation.backprop()
        self.weights_gradient = input_gradient @ self.input.T
        self.bias_gradient = input_gradient
        self.output_gradient = self.weights.T @ input_gradient
        return self.output_gradient

    def update(self, optimizer):
        self.weights = optimizer.update(self.weights, self.weights_gradient)
        self.bias = optimizer.update(self.bias, self.bias_gradient)


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, W, w_grad):
        """Return the updated weights given the gradient w_grad."""
        raise NotImplementedError


class GradientDesent(Optimizer):
    def update(self, W, w_grad):
        return W - self.learning_rate * w_grad


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.previous = None

    def update(self, W, w_grad):
        if self.previous is None:
            self.velocity = np.zeros_like(w_grad)
        momentum = self.momentum * self.velocity + (1 - self.momentum) * w_grad
        return W - self.learning_rate * momentum


class RmsProp(Optimizer):
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.previous = None

    def update(self, W, w_grad):
        if self.previous is None:
            self.cache = np.zeros_like(w_grad)
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * w_grad**2
        return W - self.learning_rate * w_grad / (np.sqrt(self.cache) + self.epsilon)
