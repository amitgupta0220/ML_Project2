import matplotlib.pyplot as plt
import numpy as np
from data import data_set

'''
Input:
z : an array of shape (m, n) where m is the number of samples and n is the number of classes.

Output:

exp / np.sum(exp, axis=1, keepdims=True) : a matrix of shape (m, n) representing the softmax activations for each sample and class.
'''


def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


'''
Input:
X: a matrix of shape (m, n) where m is the number of samples and n is the number of features.
y: a matrix of shape (m, n_classes) representing the one-hot-encoded target labels for each sample.
n_classes: an integer representing the number of classes in the target variable.
learning_rate: a float representing the learning rate for gradient descent optimization (default = 0.1).
epochs: an integer representing the number of iterations for which to run the gradient descent algorithm (default = 1000).

Output:
weights: a matrix of shape (n, n_classes) representing the learned weight parameters of the Softmax Regression model.
bias: a matrix of shape (1, n_classes) representing the learned bias parameters of the Softmax Regression model
'''


def SoftmaxRegression(X, y, n_classes, learning_rate=0.1, epochs=1000):
    _, n_features = X.shape
    weights = np.zeros((n_features, n_classes))
    bias = np.zeros((1, n_classes))

    for epoch in range(epochs):
        y_pred = softmax(np.dot(X, weights) + bias)
        error = y_pred - y
        gradient = np.dot(X.T, error)
        weights -= learning_rate * gradient
        bias -= learning_rate * np.sum(error, axis=0, keepdims=True)

    return weights, bias


'''
Input:
X: a matrix of shape (m, n) where m is the number of samples and n is the number of features.
weights: a matrix of shape (n, n_classes) representing the learned weight parameters of the Softmax Regression model.
bias: a matrix of shape (1, n_classes) representing the learned bias parameters of the Softmax Regression model.

Output :
np.argmax(softmax(z), axis=1): an array of shape (m, ) representing the predicted class labels for each input sample.
'''


def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return np.argmax(softmax(z), axis=1)


'''
Input:
labels: an array of shape (m, ) representing the target labels for each sample.

Output:

labels_encoded: a matrix of shape (m, n) representing the one-hot-encoded labels for each sample, where n is the number of unique labels in labels.
'''


def one_hot_encoding(labels):
    num_samples = len(labels)
    labels_encoded = np.zeros((num_samples, len(np.unique(labels))))
    for i in range(num_samples):
        if labels[i] == 'Plastic':
            labels_encoded[i, 0] = 1
        elif labels[i] == 'Metal':
            labels_encoded[i, 1] = 1
        elif labels[i] == 'Ceramic':
            labels_encoded[i, 2] = 1

    return labels_encoded


# Load the data
data = np.array(data_set)

# Split the data into features and labels
X = data[:, :-1].astype(float)
y = data[:, -1]

# Convert the categorical label to one-hot encoding
y_onehot = one_hot_encoding(y)

# Split the dataset into training and test sets
'''
Tuple of four 2D numpy arrays: train_features, train_labels, test_features, and test_labels. The input data is split into 80% training and 20% testing sets.
'''
n_samples = X.shape[0]
n_train = int(0.8 * n_samples)
train_indices = np.random.choice(range(n_samples), n_train, replace=True)
test_indices = np.array(list(set(range(n_samples)) - set(train_indices)))
X_train, y_train = X[train_indices], y_onehot[train_indices]
X_test, y_test = X[test_indices], y_onehot[test_indices]


# Train a single softmax regression classifier
'''
Softmax regression is performed on trainig data which returns the weights and bias as an output.
then using the weights and bias the model is used to predict the output using the test data and then error rate is calculated.
'''
weights, bias = SoftmaxRegression(X_train, y_train, n_classes=3)
y_pred = predict(X_test, weights, bias)
error_rate_single = np.mean(y_pred != np.argmax(y_test, axis=1))

# Train an ensemble of softmax regression classifiers using boosting
'''
Here ada boosting is performed for 10,25 and 50 boosts.
Initial weights are set according to the dimensions of the data. 
Then AdaBoost algorithm is used and classifiers are generated. Which is later used to predict the output for test data and error rates are calculated.
'''
n_boosts = [10, 25, 50]
error_rates = []
for n_boost in n_boosts:
    classifiers = []
    for i in range(n_boost):
        sample_weights = np.ones(n_train) / n_train
        sample_indices = np.random.choice(
            range(n_train), n_train, replace=True, p=sample_weights)
        X_sample, y_sample = X_train[sample_indices], y_train[sample_indices]
        weights, bias = SoftmaxRegression(X_sample, y_sample, n_classes=3)
        classifiers.append((weights, bias))

        y_pred = np.zeros((X_test.shape[0], 3))
        for weights, bias in classifiers:
            y_pred += softmax(np.dot(X_test, weights) + bias)
        y_pred = np.argmax(y_pred, axis=1)
        error_rate = np.mean(y_pred != np.argmax(y_test, axis=1))
    error_rates.append(error_rate)

# Print the results
print("Error rate of a Softmax Regression classifier model :", error_rate_single)
for i in range(len(n_boosts)):
    print(
        f"Error rate of an ensemble using Adaboost of {n_boosts[i]} classifiers:", error_rates[i])


# Plot the results
plt.plot([10, 50, 100], error_rates, marker='o')
plt.axhline(y=error_rate_single, linestyle='--', color='r')
plt.xlabel("Number of iterations")
plt.ylabel("Error rate")
plt.legend(["AdaBoosting", "Single classifier"])
plt.show()
