import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


np.random.seed(13)

def target_func(x):
    return x ** 2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def loss_func(y_out, y):
    return (y_out - y) ** 2

def d_loss_func(y_out, y):
    return 2 * (y_out - y)


class MLP:
    def __init__(self, hidden_l_size, learning_rate):
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.randn(hidden_l_size, 1)
        self.weights_hidden_output = np.zeros([1, hidden_l_size])

        self.bias_input_hidden = np.zeros([hidden_l_size, 1])
        self.bias_hidden_output = np.zeros([1, 1])

    def forward(self, x, y):
        y_out, values = self.predict(x)
        loss = loss_func(y_out, y)
        return y_out, loss, values

    def predict(self, x_set):
        hidden_layer_input = np.dot(self.weights_input_hidden, x_set) + self.bias_input_hidden
        hidden_layer_output = tanh(hidden_layer_input)

        output_layer_input = np.dot(self.weights_hidden_output, hidden_layer_output) + self.bias_hidden_output
        y_out = sum(output_layer_input)

        return y_out, {"HI": hidden_layer_input, "HO": hidden_layer_output, "OI": output_layer_input}

    def backward(self, x, y, y_out, values):
        # Output layer error
        output_error = d_loss_func(y_out, y)
        output_delta = output_error * np.sum(self.weights_hidden_output)

        self.weights_hidden_output -= self.learning_rate * (1 / x.shape[1]) * np.dot(output_error, values["HO"].T)
        self.bias_hidden_output -= self.learning_rate * (1 / x.shape[1]) * np.dot(output_error, np.ones([output_error.shape[1], 1]))

        # Hidden layer error
        hidden_l_error = self.weights_hidden_output.T * output_delta
        hidden_layer_delta = hidden_l_error * d_tanh(values["HI"])

        self.weights_input_hidden -= self.learning_rate * (1 / x.shape[1]) * np.dot(hidden_layer_delta, x.T)
        self.bias_input_hidden -= self.learning_rate * (1 / x.shape[1]) * np.dot(hidden_layer_delta, np.ones([hidden_layer_delta.shape[1], 1]))

    def train(self, x_set, y_set, epochs):
        for epoch in range(epochs):
            y_out, loss, values = self.forward(x_set, y_set)
            self.backward(x_set, y_set, y_out, values)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {np.mean(loss)}")


if __name__ == '__main__':
    x_train = np.linspace(-10, 10, 1000).reshape(1, 1000)
    np.random.shuffle(x_train)
    y_train = np.array(target_func(x_train)).reshape(1, 1000)
    x_test = np.sort(np.random.uniform(low=-10, high=10, size=1000).reshape(1, 1000))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train.reshape(-1, 1)).T
    x_test = scaler.transform(x_test.reshape(-1, 1)).T

    mlp = MLP(hidden_l_size=30, learning_rate=0.01)
    loss = mlp.train(x_train, y_train, 20000)
    predicted_output, _ = mlp.predict(x_test)

    plt.plot(x_train.flatten(), y_train.flatten(), label='Actual function')
    plt.scatter(x_test, predicted_output, color='red', marker='o', label='Predicted points', s=10)
    plt.legend()
    plt.show()