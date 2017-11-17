import numpy as np

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork:
    def __init__(self, layers, activation="tanh"):
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_deriv = sigmoid_derivative
        elif activation == "tanh":
            self.activation = np.tanh
            self.activation_deriv = tanh_derivative

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i-1] + 1, layers[i] + 1))-1)*0.25) #0.25 is the range of initial W
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i+1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X #adding the bias
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for m in range(len(self.weights)):
                a.append(self.activation(np.dot(a[m], self.weights[m])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # starting back-propagation
            for m in range(len(a)-2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[m].T) * self.activation_deriv(a[m])) # calculate Oj(1-Oj)Sum(Errk)Wjk
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predic(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for m in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[m]))
        return a

