import matplotlib.pyplot as plt
import numpy as np
import random
from data import data_set
# Shuffling the data
random.shuffle(data_set)

# Split the dataset into train and test sets
train_size = int(0.75 * len(data_set))
train_data = data_set[:train_size]
test_data = data_set[train_size:]

# Separate input features from output variable
train_X = np.array([d[:-1] for d in train_data])
train_y = np.array([d[-1] for d in train_data])
test_X = np.array([d[:-1] for d in test_data])
test_y = np.array([d[-1] for d in test_data])

# One-hot encode output variable
classes = np.unique(train_y)
num_classes = len(classes)
train_y_onehot = np.zeros((len(train_y), num_classes))
for i, c in enumerate(classes):
    train_y_onehot[train_y == c, i] = 1
test_y_onehot = np.zeros((len(test_y), num_classes))
for i, c in enumerate(classes):
    test_y_onehot[test_y == c, i] = 1

# Normalize input features
mean_X = np.mean(train_X, axis=0)
std_X = np.std(train_X, axis=0)
train_X_norm = (train_X - mean_X) / std_X
test_X_norm = (test_X - mean_X) / std_X

# Define the softmax function

'''
Input:
X: a numpy array of shape (num_examples, num_features), representing the input data to the neural network.
W: a numpy array of shape (num_features, num_classes), representing the weights of the neural network.
b: a numpy array of shape (1, num_classes), representing the bias terms of the neural network.

Output:
probs : The output of the function is a numpy array of shape (m, k), where m is the number of examples in the input X and k is the number of classes. Each element in the array represents the probability of the corresponding example belonging to the corresponding class.
'''


def softmax(X, W, b):
    # Compute scores
    scores = np.dot(X, W) + b
    # Compute softmax probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

# Define the cross-entropy loss function


def cross_entropy_loss(y_onehot, probs):
    # Compute cross-entropy loss
    N = len(y_onehot)
    loss = -np.sum(y_onehot * np.log(probs)) / N

    return loss

# Define the gradient function


'''
Input:
X: the input features (a matrix of shape [N, D] where N is the number of samples and D is the number of features)
y_onehot: the one-hot encoded labels (a matrix of shape [N, K] where K is the number of classes)
probs: the predicted probabilities (a matrix of shape [N, K] where K is the number of classes)

Output:
grad_W: the gradient of the cross-entropy loss with respect to the weights W (a matrix of shape [D, K])
grad_b: the gradient of the cross-entropy loss with respect to the bias term b (a vector of shape [K])
'''


def gradient(X, y_onehot, probs):
    # Compute gradient
    N = len(y_onehot)
    error = probs - y_onehot
    grad_W = np.dot(X.T, error) / N
    grad_b = np.mean(error, axis=0)

    return grad_W, grad_b

# Define the function to train a softmax


'''
Input:
Trains a softmax regression model given the input features and one-hot encoded output variable. The model is trained using gradient descent and cross-entropy loss.
X: numpy array containing the input features of the training set.
y_onehot: numpy array containing the one-hot encoded output variable of the training set.
num_classes: integer representing the number of classes in the output variable.
learning_rate: float representing the learning rate used in gradient descent.
num_epochs: integer representing the number of epochs (i.e. passes through the training set) used in training the model.

Output:
W: numpy array containing the learned weights of the model.
b: numpy array containing the learned biases of the model.
'''


def train_softmax(X, y_onehot, num_classes=3, learning_rate=0.1, num_epochs=100):
    # Initialize weights and bias
    num_features = X.shape[1]
    W = np.zeros((num_features, num_classes))
    b = np.zeros(num_classes)
    N = len(X)

    # Train the model
    for epoch in range(num_epochs):
        # Compute probabilities and gradients for entire dataset
        probs = softmax(X, W, b)
        grad_W, grad_b = gradient(X, y_onehot, probs)

        # Update weights and bias
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

        # Compute train loss and accuracy for epoch
        train_loss = cross_entropy_loss(y_onehot, probs)
        train_preds = np.argmax(probs, axis=1)
        train_acc = np.mean(train_preds == np.argmax(y_onehot, axis=1))

    return W, b


'''
Input:
train_X: a numpy array containing the feature vectors of the training data
train_y_onehot: a numpy array containing the one-hot encoded labels of the training data
test_X: a numpy array containing the feature vectors of the test data
test_y_onehot: a numpy array containing the one-hot encoded labels of the test data
iterations: an integer indicating the number of iterations of bagging to perform

Output:
error_rate : returns the error rate of the combined predictions, which is calculated by computing the proportion of incorrect predictions in the combined predictions.
'''


def bagging(train_X, train_y_onehot, test_X, test_y_onehot, iterations):
    models = []
    for i in range(iterations):
        # Create a random subset of the training data
        indices = np.random.choice(
            len(train_X), size=len(train_X), replace=True)
        train_X_subset = train_X[indices]
        train_y_subset = train_y_onehot[indices]

        # Train a softmax regression model on the subset
        W, b = train_softmax(train_X_subset, train_y_subset,
                             num_epochs=1000, num_classes=3, learning_rate=0.1)
        models.append((W, b))

    # Make predictions using each model and combine using majority voting
    y_pred = np.zeros((len(test_y_onehot), num_classes))
    for W, b in models:
        probs = softmax(test_X, W, b)
        y_pred += (probs >= 0.5)
    y_pred = np.argmax(y_pred, axis=1)

    # Compute error rate
    error_rate = np.mean(y_pred != np.argmax(test_y_onehot, axis=1))

    return error_rate


error_rates = []
for iterations in [10, 50, 100]:
    error_rate = bagging(train_X_norm, train_y_onehot,
                         test_X_norm, test_y_onehot, iterations)
    error_rates.append(error_rate)
    print(f"Bagging {iterations} iterations: Error rate = {error_rate:.3f}")

# Compute error rate for a single classifier
W, b = train_softmax(
    train_X_norm, train_y_onehot, num_classes=3, num_epochs=100, learning_rate=0.01)
probs = softmax(test_X_norm, W, b)
y_pred = np.argmax(probs, axis=1)
single_error_rate = np.mean(y_pred != np.argmax(test_y_onehot, axis=1))
print(f"Single classifier: Error rate = {single_error_rate:.3f}")

# Plot the results
plt.plot([10, 50, 100], error_rates, marker='o')
plt.axhline(y=single_error_rate, linestyle='--', color='r')
plt.xlabel("Number of iterations")
plt.ylabel("Error rate")
plt.legend(["Bagging", "Single classifier"])
plt.show()
