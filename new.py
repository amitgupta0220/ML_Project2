import random
import numpy as np
from sklearn.metrics import accuracy_score
from data import data_set


class SoftmaxRegression:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y, num_epochs=1000, learning_rate=0.01):
        num_examples, num_features = X.shape
        self.W = np.zeros((num_features, self.num_classes))
        self.b = np.zeros((1, self.num_classes))
        y_hot = self.one_hot(y)

        for epoch in range(num_epochs):
            scores = np.dot(X, self.W) + self.b
            prob = self.softmax(scores)
            loss = (-1 / num_examples) * np.sum(y_hot * np.log(prob))
            dW = (1 / num_examples) * np.dot(X.T, (prob - y_hot))
            db = (1 / num_examples) * \
                np.sum(prob - y_hot, axis=0, keepdims=True)
            self.W -= learning_rate * dW
            self.b -= learning_rate * db

    def predict(self, X):
        scores = np.dot(X, self.W) + self.b
        prob = self.softmax(scores)
        return np.argmax(prob, axis=1)

    def softmax(self, value):
        exp = np.exp(value - np.max(value, axis=1, keepdims=True))
        res = exp / np.sum(exp, axis=1, keepdims=True)
        return res

    def one_hot(self, y):
        unique_labels = ["Plastic", "Ceramic", "Metal"]
        num_examples = len(y)
        num_classes = len(unique_labels)
        y_hot = np.zeros((num_examples, num_classes))
        for i in range(num_examples):
            label_index = unique_labels.index(y[i])
            y_hot[i][label_index] = 1
        return y_hot


def bagging(X_train, y_train, num_models):
    models = []
    for i in range(num_models):
        X_bag, y_bag = resample(X_train, y_train)
        model = SoftmaxRegression(num_classes=3)
        model.fit(X_bag, y_bag)
        models.append(model)
    return models


def resample(X, y):
    num_examples = X.shape[0]
    indices = np.random.choice(num_examples, num_examples, replace=True)
    return X[indices], y[indices]


def train_test_split(data, test_size):
    np.random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    train_features = data[:split_index, :-1]
    train_labels = data[:split_index, -1]
    test_features = data[split_index:, :-1]
    test_labels = data[split_index:, -1]
    return train_features, train_labels, test_features, test_labels


X_train, y_train, X_test, y_test = train_test_split(np.array(data_set), 0.2)
X_train = X_train.astype(float)
X_test = X_test.astype(float)
num_models = [10, 50, 100]

for n in num_models:
    models = bagging(X_train, y_train, num_models=n)
    y_pred = []
    for model in models:
        y_pred.append(model.predict(X_test))
    y_pred = np.array(y_pred)
    y_pred_majority = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=y_pred)
    accuracy = accuracy_score(y_test, y_pred_majority)
    print(f"Bagging accuracy for {n} models: {accuracy:.3f}")
