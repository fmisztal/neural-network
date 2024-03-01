from sklearn.metrics import mean_squared_error
from mlp import *

np.random.seed(13)

def test_all():
    x_train = np.linspace(-10, 10, 1000).reshape(1, 1000)
    y_train = np.array(target_func(x_train)).reshape(1, 1000)
    x_test = np.sort(np.random.uniform(low=-10, high=10, size=1000).reshape(1, 1000))
    y_test = np.array(target_func(x_test)).reshape(1, 1000)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train.reshape(-1, 1)).T
    x_test = scaler.transform(x_test.reshape(-1, 1)).T

    learning_rates = [0.01, 0.005, 0.001, 0.0001]
    layer_sizes = [5, 10, 12, 15, 20, 30, 50]
    epochs = [1000, 10000, 15000, 20000, 50000]

    for ls in layer_sizes:
        for ep in epochs:
            mse_ep = []
            for lr in learning_rates:
                mlp = MLP(hidden_l_size=ls, learning_rate=lr)
                mlp.train(x_train, y_train, ep)
                predicted_output, _ = mlp.predict(x_test)
                mse_ep.append(mean_squared_error(y_test.flatten(), predicted_output.flatten()))
            plt.plot(learning_rates, mse_ep, marker='o', label=f'Epochs: {ep}')
        plt.xlabel('Learning Rate')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title(f'MSE vs learning rate for {ls} hidden neurons')
        plt.xscale('log')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    test_all()