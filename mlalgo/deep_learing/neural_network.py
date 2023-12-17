from .layers import GradientDesent, Softmax


class NeuralNetwork:
    def __init__(self, loss_function, layers=None, optimizer=GradientDesent()):
        if layers is None:
            layers = []
        self.loss_function = loss_function
        self.layers = layers
        self.optimizer = optimizer

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backprop(self, input_gradient):
        for layer in reversed(self.layers):
            input_gradient = layer.backprop(input_gradient)
        return input_gradient

    def update(self):
        for layer in self.layers:
            layer.update(self.optimizer)

    def train(self, x, y):
        pred = self.forward(x)
        loss = self.loss_function.loss(pred, y)
        if self.layers[-1].activation.__class__ == Softmax:
            loss_gradient = pred - y
        else:
            loss_gradient = self.loss_function.grad()
        self.backprop(loss_gradient)
        self.update()
        return loss

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            loss = 0.0
            for i in range(len(X)):
                loss += self.train(X[i, :], y[i])
            print(f"Epoch {epoch}: {loss / len(X)}")

    def predict(self, x):
        return self.forward(x)
