import numpy as np
from collections import Counter
from data import data_set
# Define the dataset
data = np.array(data_set)

# Extract the numerical data (first 4 columns)
X = data[:, :4].astype(float)


def k_means_clustering(X, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        # Compute the distance between each point and the centroids
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)

        # Assign each point to the closest centroid
        labels = np.argmin(distances, axis=1)

        # Update the centroids as the mean of the points assigned to each cluster
        for i in range(k):
            centroids[i] = np.mean(X[labels == i], axis=0)

    return labels, centroids


# Define the true labels
true_labels = data[:, 4]

# Compute the accuracy for each value of k
for k in [3, 6, 9]:
    # Cluster the data using K-Means
    labels, centroids = k_means_clustering(X, k)

    # Compute the accuracy for each cluster
    cluster_accuracies = []
    for i in range(k):
        # Get the labels for the data points in the cluster
        cluster_labels = true_labels[labels == i]

        # Compute the most common label in the cluster
        most_common_label = Counter(cluster_labels).most_common(1)[0][0]

        # Compute the accuracy for the cluster
        cluster_accuracy = sum(
            cluster_labels == most_common_label) / len(cluster_labels)
        cluster_accuracies.append(cluster_accuracy)

    # Compute the overall accuracy as the weighted sum of the accuracies
    overall_accuracy = np.average(cluster_accuracies, weights=[
                                  sum(labels == i) for i in range(k)])

    print(f"K={k}: Cluster accuracies = {cluster_accuracies}, Overall accuracy = {overall_accuracy}")
