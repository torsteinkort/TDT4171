import numpy as np


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int):
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test

### MY CODE ###

class NeuralNetwork:
    def __init__(self):
        self.n_inputs = 2 # number of inputs
        self.n_hidden_units = 2 # number of hidden units
        self.n_output_units = 1 # number of output nodes 

        # initial weights
        self.weights_1 = np.random.randn(self.n_inputs, self.n_hidden_units) # (2 x 2)
        self.weights_2 = np.random.randn(self.n_hidden_units, self.n_output_units) # (2 X 1)
    
    # Activation function
    def sigmoid(self, x):
        for i in range(len(x)):
            x[i][0] = 1/(1+np.exp(-x[i][0]))
            x[i][1] = 1/(1+np.exp(-x[i][1]))
        return x
    
    # Derivative of activation function
    def sigmoid_derivative(self, x):
        for i in range(len(x)):
            x[i][0] = x[i][0]*(1-x[i][0])
            x[i][1] = x[i][0]*(1-x[i][0])
        return x
    
    # Propagate
    def propagate(self, X): # X is a tuple of two values
        input_after_weights = np.dot(X, self.weights_1)
        self.hidden_layer = self.sigmoid(input_after_weights)

        self.output_value = np.dot(self.hidden_layer, self.weights_2)
        return self.output_value
    
    def output_delta(self, output, y_train):
        for i in range(len(output)):
            output[i][0] = output[i][0] - y_train[i]
        return output
    
    # Train neural net
    def train(self, X_train, y_train, iterations = 20000, learning_rate = 0.1):
        for _ in range(iterations):
            # forward propagate
            output = self.propagate(X_train)

            # backward propagate
            hidden_error = self.output_delta(output, y_train).dot(self.weights_2.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer)

            # update weights
            self.weights_2 = self.weights_2 - learning_rate * self.hidden_layer.T.dot(self.output_delta(output, y_train)) 
            self.weights_1 = self.weights_1 - learning_rate * X_train.T.dot(hidden_delta)

        return output

    def test(self, X_test, y_test):
        # must be run after training the network
        output = self.propagate(X_test)

        # calculating mean squared error
        mse = np.mean(np.square(y_test - output))
        return mse


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # TODO: Your code goes here.

    # Create neural network object
    neural_network = NeuralNetwork()

    # Train network on training data
    neural_network.train(X_train, y_train)

    # Test network on test data
    mse = neural_network.test(X_test, y_test)
    print(f"Mean squared error on test data: {mse}")
