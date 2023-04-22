import numpy as np
from sklearn.model_selection import train_test_split
from data import data_set


class SoftmaxRegression:
    def __init__(self, lr=0.001, epochs=10000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.weights = np.zeros((n_features, n_classes))

        for epoch in range(self.epochs):
            exp_scores = np.exp(X.dot(self.weights))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            y_onehot = np.zeros((n_samples, n_classes))
            y_onehot[np.arange(n_samples), y] = 1
            error = y_onehot - probs
            gradient = X.T.dot(error)
            self.weights += self.lr * gradient

    def predict(self, X):
        exp_scores = np.exp(X.dot(self.weights))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)


class AdaBoost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.full(n_samples, 1 / n_samples)

        for i in range(self.n_estimators):
            estimator = SoftmaxRegression(lr=0.1, epochs=5000)
            estimator.fit(X, y, sample_weight=weights)
            y_pred = estimator.predict(X)
            error = np.sum(weights * (y_pred != y))
            alpha = 0.5 * np.log((1 - error) / error)
            weights *= np.exp(-alpha * y * y_pred)
            weights /= np.sum(weights)
            self.estimators.append((alpha, estimator))

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for alpha, estimator in self.estimators:
            y_pred += alpha * estimator.predict(X)

        return y_pred.astype(int)


# Convert the data to a numpy array
data = np.array(data_set)

# split data into features and target variable
X = data[:, :-1].astype(float)
y = data[:, -1]

# encode target variable to integers
unique_classes = np.unique(y)
class_mapping = {label: idx for idx, label in enumerate(unique_classes)}
y = np.array([class_mapping[label] for label in y])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Evaluate error rates for single classifier
clf = SoftmaxRegression(lr=0.001, epochs=5000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
error_single = np.mean(y_pred != y_test)
print(f"Error rate for single classifier: {error_single}")
# Evaluate error rates for ensemble classifiers
for n_estimators in [10, 25, 50]:
    clf = AdaBoost(n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    error_ensemble = np.mean(y_pred != y_test)
    print(
        f"Error rate for {n_estimators} ensemble classifiers: {error_ensemble}")
