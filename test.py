import numpy as np

class NeuralNet:
    all_funcs = {
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'relu': lambda x: np.maximum(x, 0)
    }
    deriv_funcs = {
        'sigmoid': lambda x: NeuralNet.all_funcs['sigmoid'](x) * (1 - NeuralNet.all_funcs['sigmoid'](x)),
        'relu': lambda x: (x > 0) * 1
    }

    crossentropy = lambda y_prediction, y: -np.dot(np.log(y_prediction), y)
    mse = lambda y_prediction, y: np.sum(np.square(y_prediction - y)) / y.shape[0]

    def __init__(self, shape, activations=['relu', 'relu', 'sigmoid'], lr=0.1):
        self.shape = shape
        self.activations = activations
        self.lr = lr
        self.f = [NeuralNet.all_funcs[i] for i in activations]
        self.f_deriv = [NeuralNet.deriv_funcs[i] for i in activations]

        self.weights = [np.random.random((shape[s:s + 2])) * 2 - 1 for s in range(len(shape) - 1)]
        
        self.inp = 0
        self.values = [0 for s in range(len(shape[1:]))]
        self.f_values = [0 for s in range(len(shape[1:]))]

    def feedforward(self, x):
        self.inp = x
        self.values[0] = np.dot(x, self.weights[0])
        self.f_values[0] = self.f[0](self.values[0])
        for i in range(len(self.f[1:])):
            try:
                self.values[i + 1] = np.dot(self.f_values[i], self.weights[i + 1])
            except ValueError:
                self.values[i + 1] = np.dot(self.f_values[i], self.weights[i + 1].T)
            self.f_values[i + 1] = self.f[i + 1](self.values[i + 1])
        
        return self.f_values[-1]
    
    def fit(self, inputs, targets, epochs: int):
        for e in range(epochs):
            for i in range(inputs.shape[0]):
                predicted = self.feedforward(inputs[i])

                error_o = predicted - targets[i]
                if e % 1000 == 0:
                    print(error_o)
                grad_o = error_o * self.f_deriv[-1](self.values[-1])
                grads = [np.empty(s) for s in self.shape[1:]]
                grads.reverse()
                self.values.reverse()
                grads[0][0] = grad_o
                for grad in range(len(grads) - 1):
                    for k in range(grads[grad + 1].shape[0]):
                        grads[grad + 1][k] = sum(grads[grad] * self.weights[grad][k]) * self.f_deriv[grad](self.values[grad + 1][k])
                
                grads.reverse()
                for matrix in range(len(self.weights)):
                    for vector in range(self.weights[matrix].shape[0]):
                        for k in range(self.weights[matrix][vector].shape[0]):
                            if matrix == 0:
                                delta = self.lr * grads[matrix][k] * self.inp[vector]
                            else:
                                delta = self.lr * grads[matrix][k] * self.f_values[matrix][vector - 1]
                            self.weights[matrix][vector][k] -= delta

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0, 1, 1, 0]]).T
net = NeuralNet((2, 3, 2, 1), ['sigmoid', 'sigmoid', 'sigmoid'], 0.1)
net.feedforward(np.array([0, 1]))
net.fit(inputs, targets, 10000)
for i in inputs:
    print(net.feedforward(i))