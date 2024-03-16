import numpy as np


def train_perceptron(X_train, y_train, X_test, y_test, eta=0.1, n_epochs=10):
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
    activation = np.dot(xi, w)
    return 1 if activation >= 0 else -1

