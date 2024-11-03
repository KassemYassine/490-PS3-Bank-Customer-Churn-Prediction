# 490-PS3-Bank-Customer-Churn-Prediction

## Project Overview
This project aims to develop a neural network model to predict customer churn for a bank. It utilizes various customer attributes such as demographics, account details, and transaction history to determine the likelihood of customers discontinuing their service.

## Repository Contents

- `Churn_Modelling_dataset.csv`: Dataset used for modeling and analysis containing detailed customer information and churn status.
- `ClassNeuralNetwork.py`: Contains the neural network class with methods for training the model using forward and backward propagation.
- `ProcessingAndSplitting.py`: Script for data preprocessing, including normalization and encoding, and splitting data into training and test sets.
- `README.md`: Provides an overview of the project, setup instructions, and usage details.
- `Test.ipynb`: Jupyter notebook demonstrating the model training, evaluation, and prediction processes.

## Usage

To use this repository:
1. Ensure Python is installed on your system.
2. Install required libraries using the command:
3. Run the `Test.ipynb` notebook to see the neural network in action, which predicts customer churn using the preprocessed data.

## Neural Network Structure

The neural network implemented in this project includes:
- An input layer sized according to the number of features in the preprocessed dataset.
- A single hidden layer, customizable in size based on desired complexity.
- An output layer designed to predict customer churn as a binary outcome.
- Sigmoid activation functions are used both for output normalization of the neurons and in the gradient descent process during backpropagation.

Execute the `Test.ipynb` notebook to train the neural network.
