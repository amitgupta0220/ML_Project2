import random
from data import trim_data

# Define the number of clusters
K_values = [3, 6, 9]

# Define the maximum number of iterations
max_iterations = 100

# Define a function to compute the Euclidean distance between two points


def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)-1):
        distance += (point1[i] - point2[i])**2
    return distance

# Define a function to assign each data point to the closest centroid


def assign_to_cluster(data, centroids):
    clusters = {}
    for point in data:
        distances = [euclidean_distance(point, centroid)
                     for centroid in centroids]
        closest_centroid_index = distances.index(min(distances))
        if closest_centroid_index not in clusters:
            clusters[closest_centroid_index] = []
        clusters[closest_centroid_index].append(point)
    return clusters

# Define a function to update the centroids


def update_centroids(clusters):
    centroids = []
    for cluster in clusters.values():
        centroid = []
        for i in range(len(cluster[0])-1):
            centroid.append(sum([point[i]
                            for point in cluster]) / len(cluster))
        centroids.append(centroid)
    return centroids

# Define a function to perform K-means clustering


def kmeans(data, K, max_iterations):
    # Initialize K centroids randomly
    centroids = random.sample(data, K)

    # Repeat the assignment and update steps for a fixed number of iterations
    for i in range(max_iterations):
        clusters = assign_to_cluster(data, centroids)
        centroids = update_centroids(clusters)

    # Compute the accuracy and overall accuracy
    cluster_accuracies = []
    overall_accuracy = 0.0
    for i in range(K):
        cluster = clusters.get(i, [])
        if len(cluster) == 0:
            continue
        label_counts = {}
        for point in cluster:
            label = point[-1]
            label_counts[label] = label_counts.get(label, 0) + 1
        max_count = max(label_counts.values())
        correct_count = label_counts.get(
            max(label_counts, key=label_counts.get), 0)
        accuracy = correct_count / len(cluster)
        cluster_accuracies.append(accuracy)
        overall_accuracy += accuracy * len(cluster) / len(data)
    return cluster_accuracies, overall_accuracy


# Compute the accuracy and overall accuracy for K=3, 6, and 9
for K in K_values:
    print(f"K={K}")
    cluster_accuracies, overall_accuracy = kmeans(trim_data, K, max_iterations)
    for i, accuracy in enumerate(cluster_accuracies):
        print(f"Cluster {i+1} accuracy: {accuracy:.2f}")
    print(f"Overall accuracy: {overall_accuracy:.2f}")
    print()
