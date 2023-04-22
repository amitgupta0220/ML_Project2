import random
import math
import numpy as np

# Split data into training and test sets
from data import data_set
train_set = []
test_set = []
split = 0.7
for row in data_set:
    if random.random() < split:
        train_set.append(row)
    else:
        test_set.append(row)

# One-hot encoding


def one_hot_encode(data):
    # Get unique labels
    labels = set(data[:, -1])

    # Create dictionary to map labels to integers
    label_map = {}
    for i, label in enumerate(labels):
        label_map[label] = i

    # Create one-hot encoded matrix
    n_samples = len(data)
    n_features = len(data[0])-1
    n_classes = len(labels)
    X = np.zeros((n_samples, n_features*n_classes+1))
    for i in range(n_samples):
        for j in range(n_features):
            X[i][j*n_classes+label_map[data[i][-1]]] = data[i][j]
        X[i][-1] = 1

    # Create label matrix
    y = np.array([label_map[row[-1]] for row in data])

    return X, y, n_classes

# Define softmax function


def softmax(x):
    exp_x = [math.exp(i) for i in x]
    sum_exp_x = sum(exp_x)
    return [i/sum_exp_x for i in exp_x]

# Define function to train a softmax regression classifier


def train_classifier(X, y, n_classes):
    # Create weight matrix
    n_features = len(X[0])
    W = np.zeros((n_features, n_classes))

    # Set learning rate and number of epochs
    eta = 0.1
    n_epochs = 10

    # Train weight matrix using gradient descent
    for epoch in range(n_epochs):
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]
            z = [sum([W[k][j]*x_i[k] for k in range(n_features)])
                 for j in range(n_classes)]
            yhat = softmax(z)
            for j in range(n_classes):
                for k in range(n_features):
                    W[k][j] -= eta*(yhat[j] - (y_i == j))*x_i[k]
    # Return weight matrix
    return W

# Define function to make predictions using a trained classifier


def predict(W, x):
    z = [sum([W[k][j]*x[k] for k in range(len(x))]) for j in range(len(W[0]))]
    yhat = softmax(z)
    return yhat.index(max(yhat))

# Define function to evaluate a classifier on a dataset


def evaluate_classifier(W, X, y):
    n_correct = 0
    for i in range(len(X)):
        yhat = predict(W, X[i])
        if yhat == y[i]:
            n_correct += 1
    return n_correct/len(X)


def bagging(X, y, n_models):
    models = []
    for i in range(n_models):
        # Resample data with replacement
        X_resampled = []
        y_resampled = []
        for j in range(len(X)):
            index = random.randint(0, len(X)-1)
            X_resampled.append(X[index])
            y_resampled.append(y[index])

        # Train a classifier on the resampled data
        X_resampled = np.array(X_resampled)
        y_resampled = np.array(y_resampled)
        W = train_classifier(X_resampled, y_resampled, len(set(y)))

        # Add the trained classifier to the list of models
        models.append(W)

    return models


# One-hot encode training and test sets
X_train, y_train, n_classes = one_hot_encode(np.array(train_set))
X_test, y_test, _ = one_hot_encode(np.array(test_set))

# Evaluate error rate for single classifier on test set
W_single = train_classifier(X_train, y_train, n_classes)
error_single = 1 - evaluate_classifier(W_single, X_test, y_test)

# Evaluate error rate for ensemble classifiers with 10, 50, and 100 models on test set
n_models_list = [10, 50, 100]
error_ensemble = []
for n_models in n_models_list:
    models = bagging(X_train, y_train, n_models)
    W_ensemble = np.mean(models, axis=0)
    error_ensemble.append(1 - evaluate_classifier(W_ensemble, X_test, y_test))

# Print error rates
print(f"Error rate for single classifier: {error_single}")
for i, n_models in enumerate(n_models_list):
    print(
        f"Error rate for ensemble classifier with {n_models} models: {error_ensemble[i]}")
