from sklearn.model_selection import train_test_split
import numpy as np
from data import data_set


class SoftmaxRegression:
    def __init__(self, learning_rate=0.0001, num_iterations=1000, num_classes=3):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_classes = num_classes
        self.classes = None
        self.weights = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        y = self._one_hot_encoding(y)
        X = self._add_bias(X)
        self.weights = self._gradient_descent(X, y)

    def predict(self, X):
        X = self._add_bias(X)
        prob = self._softmax(X.dot(self.weights))
        return self._inverse_one_hot_encoding(prob)

    def _add_bias(self, X):
        return np.insert(X, 0, 1, axis=1)

    def _softmax(self, X):
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        res = exp / np.sum(exp, axis=1, keepdims=True)
        return res

    def _gradient_descent(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(self.classes)
        weights = np.zeros((num_features, num_classes))

        for i in range(self.num_iterations):
            scores = X.dot(weights)
            prob = self._softmax(scores)
            error = prob - y
            gradient = X.T.dot(error) / num_samples
            weights -= self.learning_rate * gradient

        return weights

    def _one_hot_encoding(self, y):
        num_samples = len(y)
        y_encoded = np.zeros((num_samples, self.num_classes))
        for i in range(num_samples):
            if y[i] == 'Plastic':
                y_encoded[i, 0] = 1
            elif y[i] == 'Metal':
                y_encoded[i, 1] = 1
            elif y[i] == 'Ceramic':
                y_encoded[i, 2] = 1
        return y_encoded

    def _inverse_one_hot_encoding(self, y_encoded):
        y = np.argmax(y_encoded, axis=1)
        return self._inverse_label_encoding(y)

    def _inverse_label_encoding(self, y_encoded):
        y = np.zeros(len(y_encoded), dtype=object)
        for i in range(len(self.classes)):
            y[y_encoded == i] = self.classes[i]
        return y


def evaluate_classifier(classifier, X, y):
    y_pred = classifier.predict(X)
    accuracy = np.mean(y_pred == y)
    return accuracy


# Convert the data to a numpy array
data = np.array(data_set)

# Separate the features and label value
X = data[:, :-1]
X = X.astype(float)
y = data[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# Convert string type data to numerical type data
y_train = np.array([0 if label == 'Plastic' else 1 if label ==
                   'Metal' else 2 for label in y_train])
y_test = np.array([0 if label == 'Plastic' else 1 if label ==
                  'Metal' else 2 for label in y_test])

# Create an instance of the Softmax Regression classifier
softmax_reg = SoftmaxRegression()

# Train the classifier on the training data
softmax_reg.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy_single = evaluate_classifier(softmax_reg, X_test, y_test)
print("Accuracy of single classifier: ", accuracy_single)

# Bagging ensemble of 10 classifiers
accuracy_ensemble_10 = []
for i in range(10):
    # Create a new instance of the Softmax Regression classifier for each bagging iteration
    softmax_reg_bagging = SoftmaxRegression()
    # Sample with replacement from the training data to create a new training set
    sample_indices = np.random.choice(
        X_train.shape[0], size=X_train.shape[0], replace=True)
    X_train_bagging = X_train[sample_indices]
    y_train_bagging = y_train[sample_indices]
    # Train the classifier on the new training set
    softmax_reg_bagging.fit(X_train_bagging, y_train_bagging)
    # Evaluate the classifier on the test data
    accuracy_ensemble_10.append(evaluate_classifier(
        softmax_reg_bagging, X_test, y_test))

print("Accuracy of ensemble classifier with 10 classifiers: ",
      np.mean(accuracy_ensemble_10))

# Bagging ensemble of 50 classifiers
accuracy_ensemble_50 = []
for i in range(50):
    # Create a new instance of the Softmax Regression classifier for each bagging iteration
    softmax_reg_bagging = SoftmaxRegression()
    # Sample with replacement from the training data to create a new training set
    sample_indices = np.random.choice(
        X_train.shape[0], size=X_train.shape[0], replace=True)
    X_train_bagging = X_train[sample_indices]
    y_train_bagging = y_train[sample_indices]
    # Train the classifier on the new training set
    softmax_reg_bagging.fit(X_train_bagging, y_train_bagging)
    # Evaluate the classifier on the test data
    accuracy_ensemble_50.append(evaluate_classifier(
        softmax_reg_bagging, X_test, y_test))

print("Accuracy of ensemble classifier with 50 classifiers: ",
      np.mean(accuracy_ensemble_50))

# Bagging ensemble of 100 classifiers
accuracy_ensemble_100 = []
for i in range(100):
    # Create a new instance of the Softmax Regression classifier for each bagging iteration
    softmax_reg_bagging = SoftmaxRegression()
    # Sample with replacement from the training data to create a new training set
    sample_indices = np.random.choice(
        X_train.shape[0], size=X_train.shape[0], replace=True)
    X_train_bagging = X_train[sample_indices]
    y_train_bagging = y_train[sample_indices]
    # Train the classifier on the new training set
    softmax_reg_bagging.fit(X_train_bagging, y_train_bagging)
    # Evaluate the classifier on the test data
    accuracy_ensemble_100.append(evaluate_classifier(
        softmax_reg_bagging, X_test, y_test))

print("Accuracy of ensemble classifier with 100 classifiers: ",
      np.mean(accuracy_ensemble_100))
# Convert the data to a numpy array
# data = np.array(data_set)

# # Separate the features and label value
# X = data[:, :-1]
# X = X.astype(float)
# y = data[:, -1]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# Train a single classifier
# single_classifier = SoftmaxRegression()
# single_classifier.fit(X_train, y_train)
# single_accuracy = evaluate_classifier(single_classifier, X_test, y_test)

# clf = SoftmaxRegression()
# # Create three ensemble classifiers with different number of models
# models_10 = clf.bagging(X_train, y_train, num_models=10)
# y_pred_ensemble_10 = clf.predict_ensemble(models_10, X_test)
# accuracy_ensemble_10 = accuracy_score(y_test, y_pred_ensemble_10)

# models_50 = clf.bagging(X_train, y_train, num_models=50)
# y_pred_ensemble_50 = clf.predict_ensemble(models_50, X_test)
# accuracy_ensemble_50 = accuracy_score(y_test, y_pred_ensemble_50)

# models_100 = clf.bagging(X_train, y_train, num_models=100)
# y_pred_ensemble_100 = clf.predict_ensemble(models_100, X_test)
# accuracy_ensemble_100 = accuracy_score(y_test, y_pred_ensemble_100)

# # Print the accuracy scores
# print("Accuracy of a single classifier: {:.2f}%".format(single_accuracy * 100))
# print("Accuracy of the ensemble classifier with 10 models: {:.2f}%".format(
#     accuracy_ensemble_10 * 100))
# print("Accuracy of the ensemble classifier with 50 models: {:.2f}%".format(
#     accuracy_ensemble_50 * 100))
# print("Accuracy of the ensemble classifier with 100 models: {:.2f}%".format(
#     accuracy_ensemble_100 * 100))
