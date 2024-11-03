import numpy as np

class Neural_Network:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        # Weights for input to hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        # Weights for hidden to output layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        
        # Biases for hidden and output layer
        self.b1 = np.random.randn(self.hidden_size)
        self.b2 = np.random.randn(self.output_size)

    
    def forward(self, X):
        self.z2 = np.dot(X, self.W1) + self.b1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2) + self.b2
        y_hat = self.sigmoid(self.z3)  # Output prediction
        return y_hat
    
    def backward(self, X, y, y_hat):
        output_error = y - y_hat
        output_delta = output_error * self.sigmoid_derivative(self.z3)
        
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.z2)
        
        self.W2 += np.dot(self.a2.T, output_delta) * self.learning_rate
        
        self.W1 += np.dot(X.T, hidden_delta) * self.learning_rate
        
        self.b2 += np.sum(output_delta, axis=0) * self.learning_rate
        self.b1 += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            y_hat = self.forward(X)
            y_hat = y_hat.squeeze()
            self.backward(X, y, y_hat)

    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)