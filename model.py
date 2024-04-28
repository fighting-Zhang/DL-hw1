import numpy as np
from utils import load_model_params

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def softplus(x):
    return np.log1p(np.exp(x))

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

def softplus_derivative(x):
    return sigmoid(x)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu'):
        self.params = self.initialize_parameters(layer_sizes)
        self.activation = activation
        self.layer_sizes = layer_sizes

    def load_params(self, path):
        params = load_model_params(path)
        for i in range(1, len(self.layer_sizes)):
            self.params['W' + str(i)] = params['W' + str(i)]
            self.params['b' + str(i)] = params['b' + str(i)]

    def initialize_parameters(self, layer_sizes):
        params = {}
        for i in range(1, len(layer_sizes)):
            params['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.01
            params['b' + str(i)] = np.zeros((layer_sizes[i], 1))
        return params

    def activate(self, z):
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'tanh':
            return tanh(z)
        elif self.activation == 'leaky_relu':
            return leaky_relu(z)
        elif self.activation == 'softplus':
            return softplus(z)
        else:
            raise ValueError("Unsupported activation function.")

    def derivative_activation(self, z):
        if self.activation == 'relu':
            return relu_derivative(z)
        elif self.activation == 'sigmoid':
            return sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return tanh_derivative(z)
        elif self.activation == 'leaky_relu':
            return leaky_relu_derivative(z)
        elif self.activation == 'softplus':
            return softplus_derivative(z)
        else:
            raise ValueError("Unsupported activation function.")

    def forward(self, x):
        caches = []
        a = x
        L = len(self.params) // 2  # number of layers in the neural network

        for l in range(1, L):
            a_prev = a
            z = np.dot(self.params['W' + str(l)], a_prev) + self.params['b' + str(l)]
            a = self.activate(z)
            caches.append((a_prev, z, a))

        z = np.dot(self.params['W' + str(L)], a) + self.params['b' + str(L)]
        a = softmax(z)
        caches.append((a, z, None))
        return a, caches
    

    def backward(self, x, y, caches):
        grads = {}
        m = y.shape[1]
        L = len(caches)  # the number of layers
        aL, zL, _ = caches[-1]
        dz = aL - y  # derivative of loss with respect to last layer output
        grads["dW" + str(L)] = np.dot(dz, caches[L-2][2].T) / m
        grads["db" + str(L)] = np.sum(dz, axis=1, keepdims=True) / m

        for l in reversed(range(1, L)):
            a_prev, z, a = caches[l-1]
            da = np.dot(self.params["W" + str(l+1)].T, dz)
            dz = da * self.derivative_activation(z)
            grads["dW" + str(l)] = np.dot(dz, a_prev.T) / m
            grads["db" + str(l)] = np.sum(dz, axis=1, keepdims=True) / m

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.params) // 2
        for l in range(1, L + 1):
            self.params["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            self.params["b" + str(l)] -= learning_rate * grads["db" + str(l)]
