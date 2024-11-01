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
        # Calculate error in output
        delta3 = np.multiply(-(y - y_hat), self.sigmoid_derivative(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)  # Derivative of loss w.r.t. W2
        
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(self.z2)
        dJdW1 = np.dot(X.T, delta2)  # Derivative of loss w.r.t. W1
        
        # Update the weights with gradient descent step
        self.W1 -= self.learning_rate * dJdW1
        self.W2 -= self.learning_rate * dJdW2
        
        # Update biases
        self.b1 -= self.learning_rate * np.sum(delta2, axis=0)
        self.b2 -= self.learning_rate * np.sum(delta3, axis=0)
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            y_hat = self.forward(X)
            self.backward(X, y, y_hat)
            if epoch % 100 == 0:
                loss = np.mean((y - y_hat) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))