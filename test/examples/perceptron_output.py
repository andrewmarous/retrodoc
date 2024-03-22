import numpy as np


def train_perceptron(X_train, y_train, X_test, y_test, eta=0.1, n_epochs=10):
    """
    Trains a perceptron model using the input training data and evaluates its performance on the test data.
    
    Parameters:
    X_train (numpy.ndarray): The input features for training.
    y_train (numpy.ndarray): The target labels for training.
    X_test (numpy.ndarray): The input features for testing.
    y_test (numpy.ndarray): The target labels for testing.
    eta (float): The learning rate (default is 0.1).
    n_epochs (int): The number of epochs for training (default is 10).
    
    Returns:
    tuple: A tuple containing the trained weight vector (w) and the accuracy of the perceptron model on the test data.
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
    Calculates the activation function and predicts the class label based on it.
    
    Parameters:
    xi (array): The input features for prediction.
    w (array): The weights associated with each feature.
    
    Returns:
    int: The predicted class label (1 or -1) based on the calculated activation function.
    """
    activation = np.dot(xi, w)
    return 1 if activation >= 0 else -1

