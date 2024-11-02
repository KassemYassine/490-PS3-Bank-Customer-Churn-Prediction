#neural network section
import numpy as np

class Neural_Network:
    #initialize weights and biases randomly
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
        self.bias_hidden = np.random.rand(hidden_size) - 0.5
        self.bias_output = np.random.rand(output_size) - 0.5
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input) 
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, x, y, output):
        print(y.shape)
        print(output.shape)
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        #update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += x.T.dot(hidden_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, x, y, epochs=100):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output)

    def predict(self, x):
        return self.forward(x)