import numpy as np


def train_perceptron(X_train, y_train, X_test, y_test, eta=0.1, n_epochs=10):
    """
    Trains a perceptron model using the input training data and returns the model weights and accuracy on the test data.
    
    Parameters:
    X_train (ndarray): The input training data as a 2D array.
    y_train (ndarray): The target training labels.
    X_test (ndarray): The input test data as a 2D array.
    y_test (ndarray): The target test labels.
    eta (float): The learning rate for updating the model weights (default is 0.1).
    n_epochs (int): The number of training epochs (default is 10).
    
    Returns:
    tuple: A tuple containing the model weights (ndarray) and the accuracy of the model on the test data (float).
    """
    w = np.zeros(X_train.shape[1])
    for epoch in range(n_epochs):
        for i in range(X_train.shape[0]):
            xi = np.insert(X_train[i], 0, 1)
            yi = y_train[i]
            update = eta * (yi - predict(xi, w))
            w += update * xi
    correct_predictions = 0
    for i in range(X_test.shape[0]):
        xi_test = np.insert(X_test[i], 0, 1)
        yi_test = y_test[i]
        prediction = predict(xi_test, w)
        if prediction == yi_test:
            correct_predictions += 1
    accuracy = correct_predictions / X_test.shape[0]
    return w, accuracy


def predict(xi, w):
    """
    Predicts the class label for a given input data point using the perceptron model.
    
    Parameters:
    xi (array): The input data point.
    w (array): The weights of the perceptron model.
    
    Returns:
    int: The predicted class label (-1 or 1).
    """
    activation = np.dot(xi, w)
    return 1 if activation >= 0 else -1

