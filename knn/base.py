def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    point1 : list or array-like
        Coordinates of the first point.
    point2 : list or array-like
        Coordinates of the second point.

    Returns:
    distance : float
        The Euclidean distance between the two points.
    """
    # Ensure both points have the same dimensions
    assert len(point1) == len(point2), "Points should have the same dimensions."
    
    # Compute the Euclidean distance
    distance = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5
    return distance


def find_neighbors(X_train, y_train, query_point, k):
    """
    Find the 'k' nearest neighbors of a query point within a dataset.

    Parameters:
    X_train : list or array-like
        Training dataset containing features.
    y_train : list or array-like
        Training dataset containing labels.
    query_point : list or array-like
        Coordinates of the query point.
    k : int
        Number of neighbors to find.

    Returns:
    neighbors : list
        List of indices of the 'k' nearest neighbors.
    """
    distances = []
    
    # Calculate distance from the query point to each point in the training set
    for i, data_point in enumerate(X_train):
        distance = euclidean_distance(query_point, data_point)
        distances.append((i, distance))
    
    # Sort distances in ascending order
    distances.sort(key=lambda x: x[1])
    
    # Get indices of the 'k' nearest neighbors
    neighbors = [index for index, _ in distances[:k]]
    return neighbors


def predict(X_train, y_train, query_point, k):
    """
    Predict the class of a query point based on the majority class among its nearest neighbors.

    Parameters:
    X_train : list or array-like
        Training dataset containing features.
    y_train : list or array-like
        Training dataset containing labels.
    query_point : list or array-like
        Coordinates of the query point.
    k : int
        Number of neighbors to consider.

    Returns:
    predicted_class : int or str
        Predicted class label for the query point.
    """
    neighbors = find_neighbors(X_train, y_train, query_point, k)
    neighbor_labels = [y_train[i] for i in neighbors]
    
    # Count occurrences of each label among neighbors
    label_counts = {}
    for label in neighbor_labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    # Get the label with the highest count
    predicted_class = max(label_counts, key=label_counts.get)
    return predicted_class


if __name__ == "__main__":
    # Sample dataset for demonstration
    X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Features of training data
    y_train = ['A', 'A', 'B', 'B']  # Corresponding labels

    # Sample query point for prediction
    query_point = [2.5, 3.5]

    # Predicting the class of the query point using KNN with 'k' neighbors
    k = 3  # Number of neighbors
    predicted_class = predict(X_train, y_train, query_point, k)
    print(f"The predicted class for the query point is: {predicted_class}")
